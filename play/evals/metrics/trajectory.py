"""族 5 agent trajectory metrics — 5 个 closure factory 返回 (Doc, Response) -> float.

设计要点：
  - **closure factory 协议**：与 judge_core / judge_rag / retrieval 同形态，
    返回 `(doc, response) -> float`；agent_traj.process_results 直接挂.
  - **无库**：5 个 metric 全在 ~200 行内手写完成（multiset F1 用 Counter,
    Levenshtein DP 空间优化到 O(n+m)），不引 python-Levenshtein / rapidfuzz——
    trajectory 长度 ≤ 50 步原生足够，引依赖性价比低.
  - **trajectory_match 命名**（不叫 edit_distance）：归一化 similarity
    `1 − Lev/max(len)` ∈ [0,1] ↑，与项目其它 metric 全 [0,1] higher-is-better
    约定一致；BFCL "trajectory_match" 同名（README C.5 同步更新）.

数据契约（doc.metadata 标准 key，由 AgentTraj.process_docs / load_prediction 注入）：
  - `trajectory.tool_seq`        list[str]               轨迹 tool_name 序列（含 artifact + non-artifact tool）
  - `trajectory.tool_calls`      list[{tool, caller, arguments}]
  - `trajectory.decision`        str | None              finalize_artifact 落定的 decision
  - `trajectory.transcript`      list[dict]              transcript 原样（speakers 提取用）
  - `trajectory.artifact`        dict[str, str]
  - `trajectory.success`         bool                    Result.success（warnings 空 = True）
  - `gold_tool_seq`              list[str]
  - `gold_tool_calls`            list[{tool, caller?, arguments?}]
  - `required_callers`           dict[tool, list[caller]]   coverage 'callers' kind
  - `expected_decision_options`  list[str]                  predicate_decision_in_options 用
  - `expected_speakers`          list[str]                  coverage 'speakers' kind / predicate

行业对标：
  - τ-bench (Anthropic 2024)：task_success 以 `verify(state) -> bool` 落，headline metric
  - BFCL (Berkeley Function-Calling Leaderboard)：tool_call_set_f1 + argument_correctness 是头部
  - inspect_ai trace match：trajectory_match 同源；学术 paper 多用原始 edit distance

显式不实现（README 和 plan 都已记）：
  - `tool_selection_accuracy`：与 trajectory_match 信号高度重合
  - `step_count_efficiency`：agent_engine steps scenario-pinned，恒为 ~1.0 无 signal
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Hashable, Sequence

from ..api import Doc, Response


# ---------- pure math（无 doc/response 依赖，最易单测）-------------------------

def multiset_f1(pred: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """两个 multiset 的 F1（Counter 取交集得 TP）.

    边界（与 IR 社区做法一致）：
      - 双方都空 → 1.0（vacuously matched）
      - 一方空一方非空 → 0.0（precision 或 recall 必为 0）
      - 都非空但交集为空 → 0.0（避免 0/0）
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    tp = sum((pc & gc).values())
    if tp == 0:
        return 0.0
    precision = tp / sum(pc.values())
    recall = tp / sum(gc.values())
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """编辑距离（insert / delete / substitute 各 cost 1）；O(n·m) 时间，O(m) 空间.

    序列元素只要 `__eq__` 就行——可以是 str / tuple / dict（虽然 dict 在我们的
    通路上一般已经 normalize 成 str/tuple 再喂进来）.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def normalized_lev_match(a: Sequence[Any], b: Sequence[Any]) -> float:
    """`1 − Levenshtein(a, b) / max(|a|, |b|)`，范围 [0,1] ↑.

    双空 → 1.0（vacuous）；其它情况下 max(|a|,|b|) ≥ Lev(a,b)，结果落 [0,1].
    与原始编辑距离互补：前者直观、与项目其它指标方向一致；后者保留绝对长度信息.
    """
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / n


# ---------- 数据提取 helpers --------------------------------------------------

def _traj(doc: Doc) -> dict:
    """容错读 doc.metadata['trajectory']；缺失时给空 dict 让下游优雅降级."""
    return doc.metadata.get("trajectory", {}) or {}


# ---------- closure factories（外部 API；与 judge_rag 协议同形）---------------

def task_success(predicate: Callable[[Doc], bool]) -> Callable[[Doc, Response], float]:
    """Outcome 类一等公民：把 `predicate(doc) -> bool` 包装成 0/1 metric.

    谓词通常读 `doc.metadata['trajectory']` 的 decision / artifact / warnings 等
    与 doc.metadata 的 gold 字段比对——具体逻辑由 task 提供，metric 只负责 0/1 规约.
    异常等价于失败（避免谓词 bug 把 batch 拉爆）.

    业界对标 τ-bench `verify(state) -> bool`：headline outcome metric.
    """

    def _score(doc: Doc, _response: Response) -> float:
        try:
            return 1.0 if predicate(doc) else 0.0
        except Exception:  # noqa: BLE001 — 谓词出错保守计 0
            return 0.0

    return _score


def tool_call_set_f1() -> Callable[[Doc, Response], float]:
    """Multiset F1 over `(tool_name, caller)` tuples.

    gold 来自 `doc.metadata['gold_tool_calls']`，pred 来自 `trajectory.tool_calls`.
    `(tool, caller)` 二元组回答"谁调了哪个工具"，与 `argument_correctness`（处理 args 侧）
    构成互补——args 含 LLM 生成的长文本（如 write_section 的 content），gold 无法在
    fixture 阶段固定，故 set F1 选不进 args，而由 argument_correctness 用 ⊆ 子集
    匹配关键参数. BFCL 'tool_call_set' 同源思路（args 严苛匹配是 BFCL function-call
    benchmark 场景；workshop 多 agent 自由生成场景下 (tool, caller) 信号更稳）.
    """

    def _key(call: dict) -> tuple[str, str]:
        return (str(call.get("tool", "")), str(call.get("caller", "")))

    def _score(doc: Doc, _response: Response) -> float:
        gold = doc.metadata.get("gold_tool_calls", []) or []
        pred = _traj(doc).get("tool_calls", []) or []
        return multiset_f1([_key(c) for c in pred], [_key(c) for c in gold])

    return _score


def argument_correctness() -> Callable[[Doc, Response], float]:
    """对每条 gold tool_call，看 pred 里是否有同名 tool 且 args ⊇ gold args.

    "⊇" 而非 "="：gold 通常只 pin 关键参数；额外 LLM 填的参数不应惩罚（如默认值、
    可选 description 字段）。返回命中率 ∈ [0,1].

    `gold_tool_calls` 缺失或空 → 1.0（无要求 = 满分；与 multiset_f1 双空规约一致）.
    pred 空但 gold 非空 → 0.0.
    """

    def _match(gold_call: dict, pred_call: dict) -> bool:
        if str(gold_call.get("tool", "")) != str(pred_call.get("tool", "")):
            return False
        gold_args = gold_call.get("arguments", {}) or {}
        pred_args = pred_call.get("arguments", {}) or {}
        for k, v in gold_args.items():
            if k not in pred_args or pred_args[k] != v:
                return False
        return True

    def _score(doc: Doc, _response: Response) -> float:
        gold = doc.metadata.get("gold_tool_calls", []) or []
        if not gold:
            return 1.0
        pred = _traj(doc).get("tool_calls", []) or []
        if not pred:
            return 0.0
        hits = sum(1 for g in gold if any(_match(g, p) for p in pred))
        return hits / len(gold)

    return _score


def trajectory_match() -> Callable[[Doc, Response], float]:
    """归一化 Levenshtein similarity on tool_name 序列.

    `1 − Lev(gold_seq, pred_seq) / max(len)`，范围 [0,1] ↑.
    gold 来自 `doc.metadata['gold_tool_seq']`，pred 来自 `trajectory.tool_seq`.

    BFCL trajectory_match / inspect_ai trace match 同源命名.
    """

    def _score(doc: Doc, _response: Response) -> float:
        gold = doc.metadata.get("gold_tool_seq", []) or []
        pred = _traj(doc).get("tool_seq", []) or []
        return normalized_lev_match(pred, gold)

    return _score


def trajectory_coverage(*, kind: str = "callers") -> Callable[[Doc, Response], float]:
    """Required ∩ Visited / |Required|；用于"每个 member 必须 cast_vote"等约束.

    `kind="callers"`（默认）：required = `doc.metadata['required_callers']` 中所有
        (tool, caller) 对；visited = pred trajectory 中实际出现的 (tool, caller).
    `kind="speakers"`：required = `expected_speakers`；visited = transcript 中
        实际说过话的 speaker. 用于 free-form 场景（brainstorm）的 fallback 度量.

    Required 为空 → 1.0（无约束 = 满分）.
    """
    if kind not in {"callers", "speakers"}:
        raise ValueError(f"trajectory_coverage: unknown kind={kind!r}")

    def _score_callers(doc: Doc) -> float:
        req_map = doc.metadata.get("required_callers", {}) or {}
        required: set[tuple[str, str]] = {
            (tool, caller) for tool, callers in req_map.items() for caller in callers
        }
        if not required:
            return 1.0
        pred = _traj(doc).get("tool_calls", []) or []
        visited = {
            (str(c.get("tool", "")), str(c.get("caller", ""))) for c in pred
        }
        return len(required & visited) / len(required)

    def _score_speakers(doc: Doc) -> float:
        required = set(doc.metadata.get("expected_speakers", []) or [])
        if not required:
            return 1.0
        transcript = _traj(doc).get("transcript", []) or []
        visited = {e["speaker"] for e in transcript if "speaker" in e}
        return len(required & visited) / len(required)

    fn = _score_callers if kind == "callers" else _score_speakers

    def _score(doc: Doc, _response: Response) -> float:
        return fn(doc)

    return _score


# ---------- ready-made predicates for task_success --------------------------

def predicate_decision_in_options(doc: Doc) -> bool:
    """panel 类场景：finalize_artifact 已落定 + decision ∈ expected_decision_options.

    artifact 缺失 / decision 空 / decision 不在白名单 → False.
    白名单本身缺失 → False（不做 default-allow，避免误判）.
    """
    traj = _traj(doc)
    if not traj.get("artifact"):
        return False
    decision = (traj.get("decision") or "").strip()
    if not decision:
        return False
    options = doc.metadata.get("expected_decision_options", []) or []
    return decision in options


def predicate_speakers_covered(doc: Doc) -> bool:
    """free-form 场景：所有 expected_speakers 都至少发言一次 + run success=True.

    expected_speakers 缺失 → 仅看 success（warnings 是否空）.
    """
    traj = _traj(doc)
    expected = doc.metadata.get("expected_speakers", []) or []
    if not traj.get("success", False):
        return False
    if not expected:
        return True
    transcript = traj.get("transcript", []) or []
    spoke = {e["speaker"] for e in transcript if "speaker" in e}
    return all(s in spoke for s in expected)
