"""Phase 5 vertical slice: Family 5 agent trajectory task.

3 trajectory eval docs + 4 stubs for `play/agent_engine/scenarios/*.md`
predictions (perfect/partial/wrong_decision/garbage). Core of teaching narrative:
"Look at the agent behavior quality ladder in process metric and outcome metric"——

  | prediction | task_success | tool_call_set_f1 | trajectory_match | coverage | story |
  |---|---|---|---|---|---|
  | perfect | 1.0 | ~1.0 | ~1.0 | ~1.0 | upper bound sanity |
  | partial | 0.0 | ~0.6 | ~0.6 | ~0.6 | tools partial / not finalize → failed (**core narrative**: process > 0 but outcome=0) |
  | wrong_decision | 0.0 | ~1.0 | ~1.0 | ~1.0 | The tools are fully adjusted but the decision is wrong (**reverse narrative**: tool call pair ≠ task pair) |
  | garbage | 0.0 | 0.0 | 0 | 0.0 | lower bound sanity |

Design points:
  - **output_type='none'** (literal introduced in phase 4): runner skips LM calls. agent_engine
    The internal complete LLM link is run in the subprocess, and there is no LM-side output on the evals side.
  - **gold.jsonl row order is sorted by "smoke → medium → heavy" (DECISIONS §7.1.3)**:
    brainstorm (2 steps ~10-30s) → example (kitchen sink ~1-2 minutes) → panel (5 characters × 11 steps several minutes).
    `--limit 1` naturally hits the brainstorm single run to do CI-friendly smoke testing, explicitly with tests/conftest.py
    Select brainstorm's strategy alignment. The rearranged data itself will keep the bytes unchanged.
  - **process_docs injected into trajectory** (run path): via `models.agent_engine_run.make_run_fn`
    fork agent_engine subprocess + parse JSON envelope (cli.py --save-result-json writes out),
    Inject `{transcript, artifact, warnings, success}` into `doc.metadata['trajectory']`.
  - **load_prediction injection trajectory** (score path): extract the same envelope from row and inject it
    `doc.metadata['trajectory']`, path B+C dual paths with the same address.
  - **judge_lm optional**: None → only 4 non-judge metrics; hang plan_quality when giving judge_lm
    (Reuse judge_core.g_eval directly, do not duplicate it in metrics/trajectory.py).
  - **task_success predicate** is installed by task based on doc.metadata in process_results:
    panel-style → `predicate_decision_in_options` (finalize + decision ∈ whitelist);
    free-form → `predicate_speakers_covered` (all expected_speakers speak + success=True).

Backward compatibility: This task can also work normally in the score path through the default construction of `run_fn=None`; the run path is required
Run_fn must be injected (default cli.py injection)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

from .._ae_bridge import Result
from ..api import Doc, Response, SampleResult
from ..metrics.judge_core import g_eval
from ..metrics.trajectory import (
    argument_correctness,
    predicate_decision_in_options,
    predicate_speakers_covered,
    task_success,
    tool_call_set_f1,
    trajectory_coverage,
    trajectory_match,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_traj" / "gold.jsonl"

PLAN_QUALITY_TEMPLATE = (
    "Rate the agent run on the dimension '{dimension}' from 1-5 "
    "(1=poor, 5=excellent).\n"
    "Topic: {input}\n"
    "Reference outcome (if any): {reference}\n"
    "Final agent transcript summary + tools used:\n{response}\n"
    "Score (1-5):"
)

PLAN_QUALITY_DIMENSIONS = ("plan_structure", "tool_choice", "completeness")

RunFn = Callable[[str], dict[str, Any]]


@register_task("agent_traj")
class AgentTraj(Task):
    """Agent trajectory eval: 5 metric + optional plan_quality (judge).

    Construction:
      - `run_fn=None` → only the score path is available (trajectory reads from predictions)
      - `run_fn=callable` → run path process_docs hook automatic subprocess run agent_engine
      - `judge_lm=None` → only 4 non-judge metrics
      - `judge_lm=lm` → add plan_quality (multidimensional G-Eval takes mean)"""

    name: ClassVar[str] = "agent_traj"
    output_type: ClassVar[str] = "none"  # phase 4 literal: runner jumps lm.generate_until

    def __init__(
        self,
        run_fn: RunFn | None = None,
        judge_lm: LM | None = None,
    ) -> None:
        self.data_path = DATA_PATH
        self._run_fn = run_fn
        self._judge_lm = judge_lm
        if judge_lm is not None:
            self._judge_plan = g_eval(
                judge_lm,
                dimensions=PLAN_QUALITY_DIMENSIONS,
                prompt_template=PLAN_QUALITY_TEMPLATE,
            )
        else:
            self._judge_plan = None

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
        """The runner is not adjusted when output_type='none'; the reserved method is only satisfied by ABC."""
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """Run path: Before the LM step, the subprocess runs all scenarios at once, and the trajectory injects metadata."""
        if self._run_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            scenario_path = d.metadata.get("scenario_path")
            if not scenario_path:
                raise ValueError(
                    f"agent_traj doc id={d.id!r} missing 'scenario_path' in metadata"
                )
            envelope = self._run_fn(scenario_path)
            out.append(_pin_trajectory(d, envelope))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: envelope field in row → doc.metadata['trajectory']; Response placeholder.

        Predictions JSONL is written by the run path, including §16 envelope and all 5 fields (including typed
        transcript entry + usage list)."""
        envelope = {
            "transcript": row["transcript"],
            "artifact": row["artifact"],
            "warnings": row["warnings"],
            "success": row["success"],
            "usage": row["usage"],
        }
        enriched = _pin_trajectory(doc, envelope)
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        predicate = self._select_predicate(doc)

        ts = task_success(predicate)
        f1 = tool_call_set_f1()
        ac = argument_correctness()
        tm = trajectory_match()
        coverage_kind = doc.metadata.get("coverage_kind", "callers")
        cov = trajectory_coverage(kind=coverage_kind)

        metrics: dict[str, float | None] = {
            "task_success": float(ts(doc, response)),
            "tool_call_set_f1": float(f1(doc, response)),
            "argument_correctness": float(ac(doc, response)),
            "trajectory_match": float(tm(doc, response)),
            "trajectory_coverage": float(cov(doc, response)),
        }

        if self._judge_plan is not None:
            judge_resp = _trajectory_summary_response(doc)
            dim_scores = self._judge_plan(doc, judge_resp)
            # DECISIONS §X wave 4: g_eval now returns dict[str, float | None] - all single-dimensional parse fails
            # The dimension is None; plan_quality mean takes the valid subset; all are None → plan_quality does not write the key.
            # aggregator (_mean_metric) natural filtering; consistent with phase 7 P2 style.
            valid = [v for v in dim_scores.values() if v is not None]
            if valid:
                metrics["plan_quality"] = sum(valid) / len(valid)
            for dim, score in dim_scores.items():
                # Subdimensions are smuggled with keys ('_' prefix), not on the aggregation panel, only for drill-down;
                # None directly drops None (dropping JSON null) retains the drill-down value.
                metrics[f"_plan_{dim}"] = float(score) if score is not None else None

        traj = doc.metadata.get("trajectory", {}) or {}
        artifacts: dict[str, Any] = {
            "scenario_path": doc.metadata.get("scenario_path"),
            "tool_seq": list(traj.get("tool_seq", [])),
            "tool_calls": list(traj.get("tool_calls", [])),
            "decision": traj.get("decision"),
            "warnings": list(traj.get("warnings", [])),
        }

        return SampleResult(
            doc_id=doc.id,
            prediction="",  # output_type='none', no LM-side output
            target=doc.target or "",
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        agg: dict[str, Callable[[list[SampleResult]], float | None]] = {
            "task_success": _mean_metric("task_success"),
            "tool_call_set_f1": _mean_metric("tool_call_set_f1"),
            "argument_correctness": _mean_metric("argument_correctness"),
            "trajectory_match": _mean_metric("trajectory_match"),
            "trajectory_coverage": _mean_metric("trajectory_coverage"),
        }
        if self._judge_lm is not None:
            agg["plan_quality"] = _mean_metric("plan_quality")
        return agg

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3: Pull LM call records from g_eval closure's _recorder."""
        if self._judge_plan is None:
            return [], None
        rec = getattr(self._judge_plan, "_recorder", None)
        if rec is None:
            return [], None
        return list(rec.responses), rec.model_label

    def higher_is_better(self) -> dict[str, bool]:
        out = {
            "task_success": True,
            "tool_call_set_f1": True,
            "argument_correctness": True,
            "trajectory_match": True,
            "trajectory_coverage": True,
        }
        if self._judge_lm is not None:
            out["plan_quality"] = True
        return out

    # ---- predicate selection ---------------------------------------------------------------

    @staticmethod
    def _select_predicate(doc: Doc) -> Callable[[Doc], bool]:
        """Select predicate according to metadata; explicit is better than implicit: the author can declare it directly in gold.jsonl
        `success_predicate: "decision_in_options" | "speakers_covered"`, otherwise press whether
        Declare `expected_decision_options` automatic fallback:
          - with expected_decision_options → decision_in_options
          - none / declares only expected_speakers → speakers_covered"""
        kind = doc.metadata.get("success_predicate")
        if kind == "decision_in_options":
            return predicate_decision_in_options
        if kind == "speakers_covered":
            return predicate_speakers_covered
        # automatic fallback
        if doc.metadata.get("expected_decision_options"):
            return predicate_decision_in_options
        return predicate_speakers_covered


# ---------- module-level helpers --------------------------------------------

def _pin_trajectory(doc: Doc, envelope: dict[str, Any]) -> Doc:
    """Derive tool_calls / tool_seq / decision from envelope, write back doc.metadata['trajectory'].

    envelope must look like `Result.asdict()` (§16, field 5): `{transcript, artifact,
    warnings, success, usage}`. `Result.from_dict` Strict deserialization (missing field KeyError).

    `transcript` / `usage` in the `trajectory` dictionary are reserialized into list[dict] form
    For evals measurement layer ([`metrics/trajectory.py`]) to consume by dict - metadata passes predictions
    The same type is maintained when JSONL is written to disk + read back."""
    result = Result.from_dict(envelope)
    tool_calls = [
        {"tool": c.tool, "caller": c.caller, "arguments": dict(c.arguments)}
        for c in result.tool_calls()
    ]
    trajectory = {
        "transcript": [dataclasses.asdict(e) for e in result.transcript],
        "artifact": dict(result.artifact),
        "warnings": list(result.warnings),
        "success": bool(result.success),
        "usage": [dataclasses.asdict(u) for u in result.usage],
        "tool_calls": tool_calls,
        "tool_seq": [c["tool"] for c in tool_calls],
        "decision": result.find_finalize_decision(),
    }
    return replace(doc, metadata={**doc.metadata, "trajectory": trajectory})


def _trajectory_summary_response(doc: Doc) -> Response:
    """Flatten the trajectory into short text and feed it to the {response} placeholder of g_eval.

    G-Eval's prompt expects a readable response, here the tool sequence + key artifact summary
    Make a Chinese-friendly text so that the judge can make a score based on this paragraph."""
    traj = doc.metadata.get("trajectory", {}) or {}
    tool_seq = traj.get("tool_seq", []) or []
    decision = traj.get("decision")
    warnings = traj.get("warnings", []) or []
    artifact = traj.get("artifact") or {}
    parts = [
        f"Tools called (in order): {', '.join(tool_seq) if tool_seq else '(none)'}.",
        f"Final decision: {decision if decision else '(no finalize)'}.",
    ]
    if warnings:
        parts.append(f"Warnings: {' | '.join(warnings)}.")
    if artifact:
        sections = "; ".join(f"{k}: {v[:60]}" for k, v in artifact.items() if v)
        if sections:
            parts.append(f"Artifact sections: {sections}.")
    return Response(doc_id=doc.id, text=" ".join(parts))


def _mean_metric(key: str) -> Callable[[list[SampleResult]], float | None]:
    """Factory: aggregation closure that averages SampleResult.metrics[key].

    DECISIONS §X wave 4: None occupancy "not measured" - key is missing / value=None are filtered;
    The set is empty → None (identical to safety / qa_open)."""

    def _agg(srs: list[SampleResult]) -> float | None:
        if not srs:
            return None
        vals = [
            s.metrics[key]
            for s in srs
            if key in s.metrics and s.metrics[key] is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    _agg.__name__ = f"mean_{key}"
    return _agg
