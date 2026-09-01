"""Family 5 agent trajectory metrics — 5 closure factory returns (Doc, Response) -> float.

Design points:
  - **closure factory protocol**: the same form as judge_core / judge_rag / retrieval,
    Return `(doc, response) -> float`; agent_traj.process_results is hung directly.
  - **No library**: 5 metrics are all handwritten in ~200 lines (multiset F1 uses Counter,
    Levenshtein DP space is optimized to O(n+m)), without python-Levenshtein / rapidfuzz——
    Trajectory length ≤ 50 steps is natively sufficient, and the introduction cost is low.
  - **trajectory_match named** (not called edit_distance): normalized similarity
    `1 − Lev/max(len)` ∈ [0,1] ↑, and other metric of the project are all [0,1] higher-is-better
    The convention is consistent; BFCL "trajectory_match" has the same name (README C.5 will be updated simultaneously).

Data contract (doc.metadata standard key, injected by AgentTraj.process_docs / load_prediction):
  - `trajectory.tool_seq` list[str] trajectory tool_name sequence (including artifact + non-artifact tool)
  - `trajectory.tool_calls` list[{tool, caller, arguments}]
  - `trajectory.decision` str | None finalize_artifact finalized decision
  - `trajectory.transcript` list[dict] transcript as is (used for extracting speakers)
  - `trajectory.artifact` dict[str, str]
  - `trajectory.success` bool Result.success(warnings empty = True)
  - `gold_tool_seq` list[str]
  - `gold_tool_calls` list[{tool, caller?, arguments?}]
  - `required_callers` dict[tool, list[caller]] coverage 'callers' kind
  - `expected_decision_options` list[str] predicate_decision_in_options used
  - `expected_speakers` list[str] coverage 'speakers' kind / predicate

Industry Benchmarking:
  - τ-bench (Anthropic 2024): task_success ends with `verify(state) -> bool`, headline metric
  - BFCL (Berkeley Function-Calling Leaderboard): tool_call_set_f1 + argument_correctness is the header
  - inspect_ai trace match: trajectory_match has the same origin; academic paper uses the original edit distance

Explicitly not implemented (both README and plan are noted):
  - `tool_selection_accuracy`: highly coincident with trajectory_match signal
  - `step_count_efficiency`: agent_engine steps scenario-pinned, always ~1.0 without signal"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Hashable, Sequence

from ..api import Doc, Response


# ----------pure math (no doc/response dependency, easiest to test alone)-------------------------

def multiset_f1(pred: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """F1 of two multisets (Counter takes the intersection to get TP).

    Boundaries (consistent with IR community practice):
      - Both sides are empty → 1.0 (vacuously matched)
      - One side is empty and the other is not empty → 0.0 (precision or recall must be 0)
      - Neither is empty but the intersection is empty → 0.0 (avoiding 0/0)"""
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
    """Edit distance (insert / delete / substitute cost 1 each); O(n·m) time, O(m) space.

    Sequence elements only need to be `__eq__` - it can be str / tuple / dict (although dict is in our
    The path has generally been normalized into str/tuple and then fed in)."""
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
    """`1 − Levenshtein(a, b) / max(|a|, |b|)`, range [0,1] ↑.

    Double empty → 1.0 (vacuous); otherwise max(|a|,|b|) ≥ Lev(a,b), the result falls into [0,1].
    Complementary to the original edit distance: the former is intuitive and consistent with the direction of other project indicators; the latter retains absolute length information."""
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / n


# ---------- Data extraction helpers --------------------------------------------------

def _traj(doc: Doc) -> dict:
    """Fault-tolerant reading of doc.metadata['trajectory']; when missing, give an empty dict to allow downstream graceful degradation."""
    return doc.metadata.get("trajectory", {}) or {}


# ---------- closure factories (external API; identical to judge_rag protocol) ---------------

def task_success(predicate: Callable[[Doc], bool]) -> Callable[[Doc, Response], float]:
    """Outcome class first class citizen: wrap `predicate(doc) -> bool` into 0/1 metric.

    Predicates usually read `doc.metadata['trajectory']` of decision / artifact / warnings etc.
    Compare with the gold field of doc.metadata - the specific logic is provided by task, and metric is only responsible for the 0/1 specification.
    Exceptions are equivalent to failures (to avoid predicate bugs that blow up the batch).

    Industry benchmark τ-bench `verify(state) -> bool`: headline outcome metric."""

    def _score(doc: Doc, _response: Response) -> float:
        try:
            return 1.0 if predicate(doc) else 0.0
        except Exception:  # noqa: BLE001 — Predicate error count 0
            return 0.0

    return _score


def tool_call_set_f1() -> Callable[[Doc, Response], float]:
    """Multiset F1 over `(tool_name, caller)` tuples.

    gold comes from `doc.metadata['gold_tool_calls']`, pred comes from `trajectory.tool_calls`.
    `(tool, caller)` tuple answers "who called which tool", with `argument_correctness` (handling args side)
    Complementary - args contains long text generated by LLM (such as the content of write_section), gold cannot be used in
    The fixture stage is fixed, so set F1 cannot select args, but argument_correctness uses the ⊆ subset
    Match key parameters. BFCL 'tool_call_set' homologous idea (args strict matching is BFCL function-call
    Benchmark scenario; workshop multi-agent free generation scenario (tool, caller) signal is more stable)."""

    def _key(call: dict) -> tuple[str, str]:
        return (str(call.get("tool", "")), str(call.get("caller", "")))

    def _score(doc: Doc, _response: Response) -> float:
        gold = doc.metadata.get("gold_tool_calls", []) or []
        pred = _traj(doc).get("tool_calls", []) or []
        return multiset_f1([_key(c) for c in pred], [_key(c) for c in gold])

    return _score


def argument_correctness() -> Callable[[Doc, Response], float]:
    """For each gold tool_call, check whether there is a tool with the same name in pred and args ⊇ gold args.

    "⊇" instead of "=": gold usually only pins key parameters; additional LLM-filled parameters should not be penalized (such as default values,
    optional description field). Return hit rate ∈ [0,1].

    `gold_tool_calls` missing or empty → 1.0 (no requirement = full score; consistent with multiset_f1 double-empty convention).
    pred is empty but gold is not empty → 0.0."""

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
    """Normalized Levenshtein similarity on tool_name sequence.

    `1 − Lev(gold_seq, pred_seq) / max(len)`, range [0,1] ↑.
    gold comes from `doc.metadata['gold_tool_seq']`, pred comes from `trajectory.tool_seq`.

    BFCL trajectory_match / inspect_ai trace match homologous naming."""

    def _score(doc: Doc, _response: Response) -> float:
        gold = doc.metadata.get("gold_tool_seq", []) or []
        pred = _traj(doc).get("tool_seq", []) or []
        return normalized_lev_match(pred, gold)

    return _score


def trajectory_coverage(*, kind: str = "callers") -> Callable[[Doc, Response], float]:
    """Required ∩ Visited / |Required|; used for constraints such as "each member must cast_vote".

    `kind="callers"` (default): required = all in `doc.metadata['required_callers']`
        (tool, caller) pair; visited = actual occurrence of (tool, caller) in pred trajectory.
    `kind="speakers"`: required = `expected_speakers`; visited = transcript
        The speaker that actually spoke. Fallback metric for free-form scenarios (brainstorm).

    Required is empty → 1.0 (no constraint = full score)."""
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
        # transcript is a list[dict] deserialized by envelope (starting from agent_engine §16, each entry contains
        # Explicit type field; speaker entry i.e. type=="speaker").
        visited = {e["speaker"] for e in transcript if e.get("type") == "speaker"}
        return len(required & visited) / len(required)

    fn = _score_callers if kind == "callers" else _score_speakers

    def _score(doc: Doc, _response: Response) -> float:
        return fn(doc)

    return _score


# ---------- ready-made predicates for task_success --------------------------

def predicate_decision_in_options(doc: Doc) -> bool:
    """Panel class scenario: finalize_artifact settled + decision ∈ expected_decision_options.

    artifact is missing / decision is empty / decision is not in the whitelist → False.
    The whitelist itself is missing → False (do not use default-allow to avoid misjudgment)."""
    traj = _traj(doc)
    if not traj.get("artifact"):
        return False
    decision = (traj.get("decision") or "").strip()
    if not decision:
        return False
    options = doc.metadata.get("expected_decision_options", []) or []
    return decision in options


def predicate_speakers_covered(doc: Doc) -> bool:
    """free-form scenario: all expected_speakers speak at least once + run success=True.

    expected_speakers missing → only look at success (whether warnings is empty)."""
    traj = _traj(doc)
    expected = doc.metadata.get("expected_speakers", []) or []
    if not traj.get("success", False):
        return False
    if not expected:
        return True
    transcript = traj.get("transcript", []) or []
    spoke = {e["speaker"] for e in transcript if e.get("type") == "speaker"}
    return all(s in spoke for s in expected)
