"""族 5 续集：agent_engine require_tool 服从性度量 — nudge_fire_rate.

设计与 [trajectory.py](trajectory.py) 同档（半通用、绑 agent_engine envelope schema、
纯函数 + 闭包工厂）。给 SFT baseline / Phase 5 复测提供"模型在 require_tool step
上的第一次服从率"信号——越低越好（少触发 nudge = 模型一次到位）。

核心定义：
  - 一个 **require_tool turn** = scenario.steps 里 `require_tool: <tool>` 字段被声明的
    一个展开后的 (agent, step) tuple
  - **nudge fire** = 该 turn 的第一次 attempt 没有产出 `(caller=agent, tool=required_tool)`
    的 artifact 事件，引擎打印 `🔁` 并发起重试
  - **nudge_fire_rate** = nudge_fire 数 / require_tool turn 总数 ∈ [0,1] ↓

数据契约（doc.metadata 标准 key，由 NudgeFireRate task 注入）：
  - `trajectory.transcript`            list[dict]   envelope 原样
  - `expected_require_tool_turns`      list[dict]   `[{turn_idx, agent, step_id, tool}, ...]`
                                                     由 process_docs / load_prediction 从
                                                     scenario YAML 自动派生
  - `scenario_path`                    str          仅用于报告 by_scenario breakdown 时取 id

转录规约（agent_engine.discussion._run_turn 写入 history 的形态）：
  - `{"type": "turn", "content": "turn N of M", "ts": ...}`           段分隔符
  - `{"speaker": <agent_name>, "content": <text>, "ts": ...}`         一次 attempt
  - `{"type": "artifact_event", "tool": ..., "caller": ..., ...}`     artifact 工具调用
  - `{"type": "tool_call", "tool": ..., "caller": ..., "ok": ...}`    非 artifact 工具调用

Phase 1 失败模式 taxonomy（1.C 引入；3 类）：
  - `missed`     第一次 attempt 该 caller 完全没有调任何工具
  - `wrong_tool` 第一次 attempt 该 caller 调了别的工具（非 required_tool）
  - `wrong_args` （**deferred to Phase 5**）调了正确工具但被 schema 拒——artifact
                  handler 目前在 error 路径不发 event，无法仅凭 transcript 判断；需
                  agent_engine 后续给 dispatch error 路径补 `{ok: false}` event 再启用。
                  当前实现下该桶恒为 0，文档诚实标注。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

from ..api import Doc, Response

# ---------- 1) scenario YAML 解析 → expected_require_tool_turns -----------

_FRONTMATTER_RE = re.compile(
    r"\A(?:[^\n]*\n)*?^---\s*\n(?P<meta>.*?)\n^---\s*\n?",
    re.DOTALL | re.MULTILINE,
)


def _split_frontmatter(text: str) -> str | None:
    """与 agent_engine.scenario._split_frontmatter 同形态——只解 frontmatter，body 不要."""
    m = _FRONTMATTER_RE.match(text)
    return m.group("meta") if m else None


def _resolve_who_to_agents(
    who: Any, agents: list[dict], roles: dict[str, str]
) -> list[str]:
    """与 agent_engine.discussion._resolve_who 等价——四种形态：
      - 'moderator' / 'member' / 'all' scalar role
      - list[name] 显式名单
    返回展开后的 agent 名字列表（保留 agent 声明顺序，与引擎一致）.
    """
    declared_order = [a["name"] for a in agents]
    if isinstance(who, str):
        if who == "all":
            return list(declared_order)
        return [n for n in declared_order if roles.get(n) == who]
    if isinstance(who, list):
        return [str(n) for n in who]
    raise ValueError(f"unsupported 'who' form in step: {who!r}")


def derive_expected_turns(scenario_path: str | Path) -> list[dict[str, Any]]:
    """解析 scenario YAML → `[{turn_idx, agent, step_id, tool}, ...]`.

    turn_idx 1-based，与 `discussion.run` 里的 `turn N of total` marker 对齐.
    expansion 顺序与引擎完全一致：steps 顺序 × 每 step 内 who 解析后的 agent 顺序.
    """
    path = Path(scenario_path)
    text = path.read_text(encoding="utf-8")
    meta_text = _split_frontmatter(text)
    if meta_text is None:
        raise ValueError(f"scenario {path} has no YAML frontmatter")
    meta = yaml.safe_load(meta_text)
    if not isinstance(meta, dict):
        raise ValueError(f"scenario {path} frontmatter is not a YAML mapping")

    agents = meta.get("agents") or []
    roles = {a["name"]: a.get("role", "member") for a in agents}
    steps = meta.get("steps") or []

    out: list[dict[str, Any]] = []
    turn_idx = 0
    for step in steps:
        who = step.get("who")
        expanded = _resolve_who_to_agents(who, agents, roles)
        require_tool = step.get("require_tool")
        step_id = step.get("id")
        for agent_name in expanded:
            turn_idx += 1
            if require_tool:
                out.append({
                    "turn_idx": turn_idx,
                    "agent": agent_name,
                    "step_id": step_id,
                    "tool": str(require_tool),
                })
    return out


# ---------- 2) transcript 切段 + nudge 检测 -------------------------------

def split_turns(transcript: list[dict]) -> list[list[dict]]:
    """把 transcript 按 `{"type": "turn"}` marker 切成段——段[N-1] 即 turn N 的内容.

    引擎在每个 (agent, step) 展开 turn 前 append 一个 turn marker（discussion.run L57-61）,
    所以段数 = 总 turn 数. marker 本身不进段; topic / 其它 pre-turn 杂物落在第 0 段
    （丢弃即可）.
    """
    segments: list[list[dict]] = []
    current: list[dict] = []
    started = False
    for entry in transcript:
        if isinstance(entry, dict) and entry.get("type") == "turn":
            if started:
                segments.append(current)
            current = []
            started = True
            continue
        if started:
            current.append(entry)
    if started:
        segments.append(current)
    return segments


def _split_attempts(segment: list[dict], agent: str) -> list[list[dict]]:
    """段内按"该 agent 的 speaker 入栈"切 attempt——每次 speaker 标志一次新 attempt
    的开始；attempt 包含其后到下一个该 agent speaker 之前的所有事件.

    规约（与引擎 _run_turn 对齐）：
      - 一个 turn 通常只属于一个 agent，所以"其他 speaker"很少出现；万一出现（旧
        日志或上游污染），会被作为前一 attempt 的 trailing 事件吞掉，不影响计数.
      - 没有任何 speaker entry 的段 → 0 attempts → 视为 missed（caller 完全沉默）.
    """
    attempts: list[list[dict]] = []
    current: list[dict] | None = None
    for entry in segment:
        if isinstance(entry, dict) and entry.get("speaker") == agent:
            if current is not None:
                attempts.append(current)
            current = []
        elif current is not None:
            current.append(entry)
    if current is not None:
        attempts.append(current)
    return attempts


def _attempt_called_required(events: list[dict], agent: str, tool: str) -> bool:
    """attempt 内是否有 `(caller=agent, tool=required_tool)` 事件——同 discussion._called_tool."""
    return any(
        isinstance(e, dict) and e.get("caller") == agent and e.get("tool") == tool
        for e in events
    )


def _attempt_called_any_tool(events: list[dict], agent: str) -> bool:
    """attempt 内该 agent 是否调过 *任何* 工具（artifact_event 或 tool_call 都算）."""
    return any(
        isinstance(e, dict)
        and e.get("caller") == agent
        and (e.get("type") in ("artifact_event", "tool_call") or e.get("tool"))
        for e in events
    )


def classify_failure_mode(
    first_attempt_events: list[dict], agent: str, required_tool: str
) -> str:
    """First attempt 没满足 require_tool 时的失败分类（Phase 1：missed / wrong_tool）.

    `wrong_args` 桶（要求工具但 schema 拒）当前不可检测——artifact dispatch 在 error
    路径不发 event，纯靠 transcript 看不出来；deferred 到 Phase 5（agent_engine 在
    error 路径补 `{ok: false}` event 后启用）. 当前 ≈ 模型从未出现该模式，安全归并.
    """
    if not _attempt_called_any_tool(first_attempt_events, agent):
        return "missed"
    return "wrong_tool"


# ---------- 3) 主入口：纯函数 compute_nudge_fire_rate --------------------

# Phase 1 supported failure modes（taxonomy 起手 2 类 + 1 deferred 占位 = 3 总桶；
# 见模块文档 wrong_args 注脚）.
FAILURE_MODES: tuple[str, ...] = ("missed", "wrong_tool", "wrong_args")


def compute_nudge_fire_rate(
    transcript: list[dict],
    expected_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """transcript + 期望表 → 度量字典.

    返回结构（doc 级；多 doc 聚合在 task.aggregation 里做）：
        {
            "nudge_fire_rate": float | None,    # fires / total（total=0 → None）
            "nudge_fire_count": int,
            "require_tool_total": int,
            "by_tool": {tool: {"fired": int, "total": int}, ...},
            "by_failure_mode": {mode: int, ...},   # 累计每种 mode 出现次数
            "per_turn": [
                {turn_idx, agent, step_id, tool, fired, mode | None, n_attempts}, ...
            ],
        }

    边界：
      - expected_turns 为空（如 brainstorm/debate/roundtable）→ rate=None, total=0,
        其它桶皆空——表示"该 doc 不参与 nudge 度量"（聚合时按 total 加权自然忽略）.
      - segments 数 < expected_turn.turn_idx（subprocess 中途崩 / scenario 截断）→
        该 turn 标 fired=True (mode='missed', n_attempts=0)，记入分母——保守计失败.
    """
    segments = split_turns(transcript)
    per_turn: list[dict[str, Any]] = []
    by_tool: dict[str, dict[str, int]] = {}
    mode_counter: Counter[str] = Counter()
    fire_count = 0

    for exp in expected_turns:
        idx = int(exp["turn_idx"]) - 1
        agent = str(exp["agent"])
        tool = str(exp["tool"])
        step_id = exp.get("step_id")

        bucket = by_tool.setdefault(tool, {"fired": 0, "total": 0})
        bucket["total"] += 1

        if idx >= len(segments):
            # turn 没跑到（subprocess 中途崩 / scenario 比 expected 短），算 missed
            fire_count += 1
            bucket["fired"] += 1
            mode_counter["missed"] += 1
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": True, "mode": "missed", "n_attempts": 0,
            })
            continue

        attempts = _split_attempts(segments[idx], agent)
        if not attempts:
            # 该 agent 在该 segment 完全没说话——算 missed
            fire_count += 1
            bucket["fired"] += 1
            mode_counter["missed"] += 1
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": True, "mode": "missed", "n_attempts": 0,
            })
            continue

        first_satisfied = _attempt_called_required(attempts[0], agent, tool)
        if first_satisfied:
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": False, "mode": None, "n_attempts": len(attempts),
            })
            continue

        # nudge 触发了
        fire_count += 1
        bucket["fired"] += 1
        mode = classify_failure_mode(attempts[0], agent, tool)
        mode_counter[mode] += 1
        per_turn.append({
            "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
            "tool": tool, "fired": True, "mode": mode, "n_attempts": len(attempts),
        })

    total = len(expected_turns)
    rate: float | None = (fire_count / total) if total > 0 else None

    # 把 FAILURE_MODES 三桶都显式列出，让 0 计数也可见——下游 breakdown 表格不缺列
    by_failure_mode = {m: int(mode_counter.get(m, 0)) for m in FAILURE_MODES}

    return {
        "nudge_fire_rate": rate,
        "nudge_fire_count": int(fire_count),
        "require_tool_total": int(total),
        "by_tool": by_tool,
        "by_failure_mode": by_failure_mode,
        "per_turn": per_turn,
    }


# ---------- 4) closure factories（与 trajectory.py 协议同形）-------------

def nudge_fire_rate_metric() -> Callable[[Doc, Response], float | None]:
    """工厂：(Doc, Response) → 该 doc 的 nudge_fire_rate.

    依赖 doc.metadata['trajectory']['transcript'] + doc.metadata['expected_require_tool_turns']
    都已被 process_docs / load_prediction 注入. 缺失字段时返 None（"未测得"）.
    """

    def _score(doc: Doc, _response: Response) -> float | None:
        traj = doc.metadata.get("trajectory", {}) or {}
        transcript = traj.get("transcript") or []
        expected = doc.metadata.get("expected_require_tool_turns") or []
        result = compute_nudge_fire_rate(transcript, expected)
        return result["nudge_fire_rate"]

    return _score
