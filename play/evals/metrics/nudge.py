"""族 5 续集：agent_engine require_tool 服从性度量 — nudge_fire_rate.

设计与 [trajectory.py](trajectory.py) 同档（半通用、绑 agent_engine envelope schema、
纯函数 + 闭包工厂）。给 SFT baseline / Phase 5 复测提供"模型在 require_tool step
上的第一次服从率"信号——越低越好（少触发 nudge = 模型一次到位）。

核心定义：
  - **require_tool turn** = scenario.steps 里 `require_tool: <tool>` 字段被声明的
    一个展开后的 (agent, step) tuple
  - **nudge fire** = 该 turn 的第一次 attempt 没有产出 `(caller=agent, tool=required_tool)`
    的 artifact 事件，引擎打印 `🔁` 并发起重试
  - **nudge_fire_rate** = nudge_fire 数 / require_tool turn 总数 ∈ [0,1] ↓

数据契约（doc.metadata 标准 key，由 NudgeFireRate task 注入）：
  - `trajectory`                       dict          envelope 形态 `{transcript, artifact,
                                                     warnings, success, usage}`（agent_engine
                                                     §16 typed entry / TokenUsage 规约）
  - `expected_require_tool_turns`      list[dict]   `[{turn_idx, agent, step_id, tool}, ...]`
                                                     由 process_docs / load_prediction 从
                                                     scenario YAML 自动派生
  - `scenario_path`                    str          仅用于报告 by_scenario breakdown 时取 id

Phase 1 失败模式 taxonomy（1.C 引入；3 类）：
  - `missed`     第一次 attempt 该 caller 完全没有调任何工具
  - `wrong_tool` 第一次 attempt 该 caller 调了别的工具（非 required_tool）
  - `wrong_args` （**deferred to Phase 5**）调了正确工具但被 schema 拒——artifact
                  handler 目前在 error 路径不发 event，无法仅凭 transcript 判断；需
                  agent_engine 后续给 dispatch error 路径补 `{ok: false}` event 再启用。
                  当前实现下该桶恒为 0，文档诚实标注。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .._ae_bridge import (
    ArtifactEventEntry,
    Result,
    Scenario,
    ToolCallEntry,
    TranscriptEntry,
)
from ..api import Doc, Response


def derive_expected_turns(scenario_path: str | Path) -> list[dict[str, Any]]:
    """解析 scenario → `[{turn_idx, agent, step_id, tool}, ...]`（仅含 require_tool 的 turn）.

    DECISIONS §13：内部委托 `agent_engine.Scenario.expanded_turns()`；turn_idx 1-based，
    与 `discussion.run` 写入的 `turn N of total` marker 对齐.
    """
    expanded = Scenario.from_yaml(str(scenario_path)).expanded_turns()
    return [
        {
            "turn_idx": e.turn_idx,
            "agent": e.agent,
            "step_id": e.step_id,
            "tool": str(e.require_tool),
        }
        for e in expanded
        if e.require_tool
    ]


# ---------- failure mode 分类（attempt → mode）---------------------------

def classify_failure_mode(
    first_attempt_events: list[TranscriptEntry],
    agent: str,
    required_tool: str,
) -> str:
    """First attempt 没满足 require_tool 时的失败分类（Phase 1：missed / wrong_tool）.

    `wrong_args` 桶（要求工具但 schema 拒）当前不可检测——artifact dispatch 在 error
    路径不发 event，纯靠 transcript 看不出来；deferred 到 Phase 5（agent_engine 在
    error 路径补 `{ok: false}` event 后启用）.

    `required_tool` 当前未被使用——保留参数以便 wrong_args 桶启用后区分"调对工具但
    args 不符"vs"调错工具"两种 wrong_tool 子类.
    """
    _ = required_tool  # reserved for wrong_args extension
    called_any_tool = any(
        isinstance(e, (ArtifactEventEntry, ToolCallEntry)) and e.caller == agent
        for e in first_attempt_events
    )
    return "wrong_tool" if called_any_tool else "missed"


# ---------- 主入口：compute_nudge_fire_rate ----------------------------

# Phase 1 supported failure modes（taxonomy 起手 2 类 + 1 deferred 占位 = 3 总桶；
# 见模块文档 wrong_args 注脚）.
FAILURE_MODES: tuple[str, ...] = ("missed", "wrong_tool", "wrong_args")


def compute_nudge_fire_rate(
    envelope: dict,
    expected_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """envelope dict + 期望表 → 度量字典.

    envelope 走 `Result.from_dict` typed 反序列化（§16）；`turns()` / `attempts()` 全
    typed dispatch.

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
    result = Result.from_dict(envelope)
    turns = result.turns()
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

        if idx >= len(turns):
            # turn 没跑到（subprocess 中途崩 / scenario 比 expected 短），算 missed
            fire_count += 1
            bucket["fired"] += 1
            mode_counter["missed"] += 1
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": True, "mode": "missed", "n_attempts": 0,
            })
            continue

        attempts = turns[idx].attempts(agent)
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

        first_satisfied = any(
            isinstance(e, (ArtifactEventEntry, ToolCallEntry))
            and e.caller == agent and e.tool == tool
            for e in attempts[0]
        )
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


# ---------- closure factories（与 trajectory.py 协议同形）---------------

def nudge_fire_rate_metric() -> Callable[[Doc, Response], float | None]:
    """工厂：(Doc, Response) → 该 doc 的 nudge_fire_rate.

    依赖 doc.metadata['trajectory']（envelope dict）+ doc.metadata['expected_require_tool_turns']
    都已被 process_docs / load_prediction 注入. 缺失字段时返 None（"未测得"）.
    """

    def _score(doc: Doc, _response: Response) -> float | None:
        envelope = doc.metadata.get("trajectory", {}) or {}
        expected = doc.metadata.get("expected_require_tool_turns") or []
        if not envelope or "transcript" not in envelope:
            return None
        result = compute_nudge_fire_rate(envelope, expected)
        return result["nudge_fire_rate"]

    return _score
