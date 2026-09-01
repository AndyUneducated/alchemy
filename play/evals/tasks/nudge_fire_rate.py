"""Phase 1 baseline main indicator: require_tool compliance.

Measure the "first time placement rate" of the model in require_tool step on `play/agent_engine/scenarios/*.md`
(= 1 - nudge_fire_rate). Same as `agent_traj` in envelope subprocess mode + same as expectations
gold.jsonl schema (`scenario_path` is the only required metadata), only the measurement function is different:

  | task | metric axis | signal |
  |---|---|---|
  | agent_traj | trajectory overall (task_success / tool F1 / coverage / ...) | final state right or wrong |
  | **nudge_fire_rate** | First response behavior of require_tool step | Process compliance |

Design points:
  - **output_type='none'**: Same as agent_traj; runner jumps LM call; agent_engine
    subprocess runs full-link LLM.
  - **expected_require_tool_turns automatically derived**: parsed from scenario YAML frontmatter,
    Avoid gold.jsonl hand maintenance and scenario drift. See [`metrics/nudge.derive_expected_turns`].
  - **Failure Mode taxonomy** (DECISIONS Phase 1 ADR §6): missed/wrong_tool/wrong_args
    Three buckets; wrong_args is currently deferred (artifact handler does not send events in the error path).
  - **Aggregation**: top-level weighted average by require_tool_total, by_scenario / by_tool /
    by_failure_mode Three breakdown dictionaries are written to aggregated.

Backward compatibility: `run_fn=None` default constructor available for score path (trajectory reads from predictions)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.nudge import (
    FAILURE_MODES,
    compute_nudge_fire_rate,
    derive_expected_turns,
)
from ..registry import register_task
from .base import Task

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "nudge_fire_rate" / "gold.jsonl"
)

# scenarios path resolution root (same origin as agent_engine_run.make_run_fn)
PLAY_DIR = Path(__file__).resolve().parents[2]

RunFn = Callable[[str], dict[str, Any]]


@register_task("nudge_fire_rate")
class NudgeFireRate(Task):
    """Agent require_tool compliance task.

    Construction:
      - `run_fn=None` → only score path is available (envelope read from predictions JSONL)
      - `run_fn=callable` → run path process_docs hook automatic subprocess run agent_engine"""

    name: ClassVar[str] = "nudge_fire_rate"
    output_type: ClassVar[str] = "none"

    def __init__(self, run_fn: RunFn | None = None) -> None:
        self.data_path = DATA_PATH
        self._run_fn = run_fn

    # ---- ABC implementations -------------------------------------------------

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(
                    id=row["id"],
                    input=row.get("input", ""),
                    target=row.get("target"),
                    metadata=dict(row.get("metadata", {})),
                )

    def doc_to_text(self, doc: Doc) -> str:
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """Run path: subprocess runs envelope, then derives expected_turns and pins to metadata."""
        if self._run_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            scenario_path = d.metadata.get("scenario_path")
            if not scenario_path:
                raise ValueError(
                    f"nudge_fire_rate doc id={d.id!r} missing 'scenario_path' in metadata"
                )
            envelope = self._run_fn(scenario_path)
            out.append(_pin_envelope(d, envelope))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: envelope within predictions JSONL row → metadata['trajectory']
        + Derive expected_turns; Response placeholder.

        Predictions JSONL is written by the run path, including §16 envelope and all 5 fields (including typed
        transcript entry + usage list)."""
        envelope = {
            "transcript": row["transcript"],
            "artifact": row["artifact"],
            "warnings": row["warnings"],
            "success": row["success"],
            "usage": row["usage"],
        }
        enriched = _pin_envelope(doc, envelope)
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        traj = doc.metadata.get("trajectory", {}) or {}
        expected = doc.metadata.get("expected_require_tool_turns") or []

        result = compute_nudge_fire_rate(traj, expected)

        # SampleResult.metrics only holds scalars (one-level nested constraints + intuitive aggregation); breakdown details
        # Enter artifacts for aggregation drill-down.
        metrics: dict[str, float | None | dict[str, float | None]] = {
            "nudge_fire_rate": result["nudge_fire_rate"],
            "nudge_fire_count": float(result["nudge_fire_count"]),
            "require_tool_total": float(result["require_tool_total"]),
        }

        artifacts: dict[str, Any] = {
            "scenario_path": doc.metadata.get("scenario_path"),
            "by_tool": result["by_tool"],
            "by_failure_mode": result["by_failure_mode"],
            "per_turn": result["per_turn"],
        }

        return SampleResult(
            doc_id=doc.id,
            prediction="",
            target=doc.target or "",
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        return {
            "nudge_fire_rate": _weighted_rate,
            "nudge_fire_count": _sum_metric("nudge_fire_count"),
            "require_tool_total": _sum_metric("require_tool_total"),
            "by_scenario": _by_scenario,
            "by_tool": _by_tool,
            "by_failure_mode": _by_failure_mode,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            # nudge_fire_rate The lower the better - a reverse metric, contrary to the "high = good" convention of other project metrics.
            # Flag False to make the show/sort UI handle it correctly; breakdown dicts are not included in higher_is_better.
            "nudge_fire_rate": False,
            "nudge_fire_count": False,
            "require_tool_total": True,  # Dimension - more turns count as the denominator, not more is better but
                                          # There is no reverse semantics of "less is better"; putting True in neutral is not misleading.
        }


# ---------- module-level helpers --------------------------------------------

def _pin_envelope(doc: Doc, envelope: dict[str, Any]) -> Doc:
    """envelope + scenario_path → metadata['trajectory'] + ['expected_require_tool_turns'].

    envelope schema is the same as agent_engine.result.Result (§16, 5 fields):
      {transcript, artifact, warnings, success, usage}
    Strictly transparent transmission - missing fields will directly raise KeyError, aligned with `Result.from_dict`."""
    trajectory = {
        "transcript": list(envelope["transcript"]),
        "artifact": dict(envelope["artifact"]),
        "warnings": list(envelope["warnings"]),
        "success": bool(envelope["success"]),
        "usage": list(envelope["usage"]),
    }
    scenario_path = doc.metadata.get("scenario_path")
    expected: list[dict[str, Any]] = []
    if scenario_path:
        sp = Path(scenario_path)
        if not sp.is_absolute():
            sp = (PLAY_DIR / sp).resolve()
        if sp.exists():
            expected = derive_expected_turns(sp)
        # If the file does not exist, it will not be thrown - the stub fixture of the score path may use a fictitious path; the doc takes "no
        # require_tool turns" path (rate=None), no require_tool like brainstorm etc.
        # scenario behaves consistently.

    new_meta = {
        **doc.metadata,
        "trajectory": trajectory,
        "expected_require_tool_turns": expected,
    }
    return replace(doc, metadata=new_meta)


# ---------- aggregation closures --------------------------------------------

def _weighted_rate(srs: list[SampleResult]) -> float | None:
    """"Global" nudge_fire_rate = Σ fires / Σ totals across docs.

    Explicitly weighted by require_tool_total - more accurate than simple average doc-level rate: each require_tool
    turn is an independent Bernoulli test weighted with the SE tightened by a factor of √N."""
    total = sum(int(s.metrics.get("require_tool_total") or 0) for s in srs)
    fires = sum(int(s.metrics.get("nudge_fire_count") or 0) for s in srs)
    if total == 0:
        return None
    return fires / total


def _sum_metric(key: str) -> Callable[[list[SampleResult]], float]:
    """Sum per-sample metric key - for nudge_fire_count / require_tool_total."""
    def _agg(srs: list[SampleResult]) -> float:
        return float(sum(float(s.metrics.get(key) or 0.0) for s in srs))
    _agg.__name__ = f"sum_{key}"
    return _agg


def _by_scenario(srs: list[SampleResult]) -> dict[str, float | None]:
    """{doc.id: nudge_fire_rate of that doc}.

    doc.id is scenario id (gold.jsonl line order convention, same origin as agent_traj). No require_tool
    The turn scenario is None here - rendering <n/a> in the breakdown table."""
    out: dict[str, float | None] = {}
    for s in srs:
        rate = s.metrics.get("nudge_fire_rate")
        out[s.doc_id] = float(rate) if isinstance(rate, (int, float)) else None
    return out


def _by_tool(srs: list[SampleResult]) -> dict[str, float | None]:
    """{tool_name: Weighted average fire rate across docs}.

    same weighting strategy as _weighted_rate (weighted by the number of turns, not by the number of docs)."""
    bucket: dict[str, dict[str, int]] = {}
    for s in srs:
        per_tool = s.artifacts.get("by_tool", {}) or {}
        for tool, counts in per_tool.items():
            b = bucket.setdefault(tool, {"fired": 0, "total": 0})
            b["fired"] += int(counts.get("fired", 0))
            b["total"] += int(counts.get("total", 0))
    out: dict[str, float | None] = {}
    for tool, b in bucket.items():
        out[tool] = (b["fired"] / b["total"]) if b["total"] > 0 else None
    return out


def _by_failure_mode(srs: list[SampleResult]) -> dict[str, int]:
    """{mode: cumulative count across doc}. All three buckets are listed explicitly (including wrong_args=0)."""
    counter = {m: 0 for m in FAILURE_MODES}
    for s in srs:
        per_mode = s.artifacts.get("by_failure_mode", {}) or {}
        for m, n in per_mode.items():
            if m in counter:
                counter[m] += int(n)
    return counter
