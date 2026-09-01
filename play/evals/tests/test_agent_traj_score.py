"""agent_traj × 4 stub predictions × 3 docs = 12 sample matrices: Numerical gatekeepers for core instructional narratives.

Run evaluate_score end-to-end: load gold.jsonl + pred.jsonl → metrics → aggregated
(unquote LM; judge_lm=None). Locked Four State Story:
  - perfect → full 1.0
  - partial → process positive or negative score / outcome=0
  - **wrong_decision** → process has full marks but outcome=0 (**phase 5 teaching core**)
  - garbage → process drops to 0 (except brainstorm vacuous 1/3)

Absolute decimals are not locked (the values may drift with fine-tuning of the fixture), only:
  ① 0/1 distribution of 4 states on task_success
  ② process metrics of wrong_decision ≈ process metrics of perfect (key narrative)
  ③ garbage task_success / coverage all 0 + tool_call set f1 strict < perfect"""

from __future__ import annotations

from pathlib import Path

from evals.runner import evaluate_score
from evals.tasks.agent_traj import AgentTraj

PRED_ROOT = Path(__file__).resolve().parent.parent / "data" / "agent_traj" / "predictions"


def _score(pred_name: str) -> dict[str, float]:
    """Run 4 metric aggregated of a certain pred (no judge)."""
    task = AgentTraj()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return dict(result.aggregated)


def test_docs_smoke_friendly_ordering():
    """DECISIONS §7.1.3 Lock: gold.jsonl row order is smoke → medium → rearranged,
    `--limit 1` hits brainstorm instead of panel (panel 5 characters × 11 steps will exceed 600s).
    Align with tests/conftest.py live test to select brainstorm's CI-friendly strategy."""
    docs = list(AgentTraj().docs())
    assert [d.id for d in docs] == ["brainstorm", "example", "panel"]


# ---------- Upper and lower bounds sanity (2 items) -----------------------------------------------

def test_perfect_all_metrics_are_one():
    agg = _score("perfect")
    for k in (
        "task_success",
        "tool_call_set_f1",
        "argument_correctness",
        "trajectory_match",
        "trajectory_coverage",
    ):
        assert agg[k] == 1.0, f"{k}={agg[k]}"


def test_garbage_outcome_and_coverage_are_zero():
    agg = _score("garbage")
    assert agg["task_success"] == 0.0
    assert agg["trajectory_coverage"] == 0.0
    # process metric and brainstorm's vacuous 1/3 residual value (gold empty vs pred empty = 1.0)
    # But the absolute value must be < perfect
    assert agg["tool_call_set_f1"] < 1.0
    assert agg["trajectory_match"] < 1.0


# ---------- partial: process positive or negative score / outcome=0 (2 items) -------------------

def test_partial_outcome_zero():
    agg = _score("partial")
    assert agg["task_success"] == 0.0


def test_partial_process_in_middle_band():
    """The process metric of partial should be between (0, 1) - neither lower nor upper bound."""
    agg = _score("partial")
    for k in ("tool_call_set_f1", "argument_correctness", "trajectory_match", "trajectory_coverage"):
        assert 0.0 < agg[k] < 1.0, f"{k}={agg[k]}"


# ---------- wrong_decision: phase 5 teaching core (3 items) -----------------------

def test_wrong_decision_outcome_is_zero():
    """All tools are in place + decision is not in the whitelist → task_success=0."""
    agg = _score("wrong_decision")
    assert agg["task_success"] == 0.0


def test_wrong_decision_process_metrics_match_perfect():
    """wrong_decision of (tool_call_set_f1 / trajectory_match / coverage) ≈ perfect.

    This is the reverse narrative of phase 5: "Tool calls all pairs ≠ task pairs" - process metric will be obtained if you look at it alone
    Completely wrong conclusion; only by showing the outcome metric together can we see through the "right program, right decision, wrong" type of agent."""
    perfect = _score("perfect")
    wrong = _score("wrong_decision")
    for k in ("tool_call_set_f1", "trajectory_match", "trajectory_coverage"):
        assert wrong[k] == perfect[k], (
            f"{k}: wrong={wrong[k]} vs perfect={perfect[k]}; "
            "wrong_decision 故事核心要求两者在 process 维度完全等价"
        )


def test_wrong_decision_outcome_breaks_process_correlation():
    """Also locked: wrong_decision in process dimension = perfect, but outcome dimension ≠ perfect.

    This is a single assertion that simultaneously verifies both sides of the "reverse narrative" - the process is right and the result is wrong."""
    perfect = _score("perfect")
    wrong = _score("wrong_decision")
    assert wrong["tool_call_set_f1"] == perfect["tool_call_set_f1"]
    assert wrong["trajectory_match"] == perfect["trajectory_match"]
    assert wrong["task_success"] != perfect["task_success"]
    assert wrong["task_success"] == 0.0


# ---------- Matrix monotonicity (1 item)---------------------------------------------

def test_metric_ordering_perfect_above_partial_above_garbage():
    """The phase 5 story matrix requires a rough staircase in the process dimension: perfect > partial >= garbage.

    Use trajectory_match as representative (the most IR-friendly and stable); do not lock absolute decimals to avoid fixture drift."""
    perfect = _score("perfect")["trajectory_match"]
    partial = _score("partial")["trajectory_match"]
    garbage = _score("garbage")["trajectory_match"]
    assert perfect > partial >= garbage


# ---------- single-doc dimension drill-down (1 item) -------------------------------

def test_brainstorm_partial_speakers_coverage():
    """brainstorm/partial only 1/3 speakers speak → coverage(speakers) = 1/3.

    Expand SampleResult.metrics via evaluate_score to lock per-doc values (not just aggregated)."""
    task = AgentTraj()
    result = evaluate_score(task, str(PRED_ROOT / "partial.jsonl"))
    # per_sample order by gold.jsonl line order: brainstorm/example/panel (DECISIONS §7.1.3, smoke → heavy)
    brainstorm = next(s for s in result.per_sample if s.doc_id == "brainstorm")
    assert abs(brainstorm.metrics["trajectory_coverage"] - 1 / 3) < 1e-9
    assert brainstorm.metrics["task_success"] == 0.0  # Only 1 speaker, speakers_covered fail
