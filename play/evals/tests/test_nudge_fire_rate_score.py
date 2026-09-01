"""nudge_fire_rate × 3 stub × 7 docs = 21 sample matrix: task end-to-end goalkeeper.

Run evaluate_score end-to-end: load gold.jsonl + pred.jsonl → metrics → aggregated.
Lock three-state story + breakdown dictionary + reverse metric direction:

  | prediction | nudge_fire_rate | by_failure_mode main bucket | story |
  |---|---|---|---|
  | perfect | 0.0 | All three buckets are 0 | Upper bound sanity (the model is 100% in place for the first time) |
  | all_nudged | 1.0 | missed dominance | sanity (the model is completely silent) |
  | mixed | (0,1) | missed + wrong_tool each | intermediate state + failure classification signal |

Absolute decimals are not locked (fixture fine-tuning may drift), only:
  ① Strict upper and lower bounds (perfect=0 / all_nudged=1)
  ② mixed strictly ∈ (0, 1)
  ③ by_scenario dimension brainstorm/debate/roundtable always None (vacuous)
  ④ by_tool dictionary contains cast_vote / append_section / retrieve_docs (the latter comes from 1.B
     New require_tool intensive scenario, dependent on agent_engine DECISIONS §12 lifted
     "require_tool artifact tool only" restriction)
  ⑤ by_failure_mode All three buckets are listed (including the deferred placeholder of wrong_args=0)
  ⑥Total number of require_tool turns = 20 (panel 4 + example 3 + tool_chain 5 + code_review 8)"""

from __future__ import annotations

from pathlib import Path

from evals.runner import evaluate_score
from evals.tasks.nudge_fire_rate import NudgeFireRate

PRED_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "nudge_fire_rate" / "predictions"
)


def _score(pred_name: str) -> dict:
    """Run the aggregated dictionary of a certain pred (including nested by_* breakdowns)."""
    task = NudgeFireRate()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return dict(result.aggregated)


def _per_sample_rates(pred_name: str) -> dict[str, float | None]:
    """Run a pred's per-doc nudge_fire_rate dictionary (drill-down lock by_scenario)."""
    task = NudgeFireRate()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return {s.doc_id: s.metrics["nudge_fire_rate"] for s in result.per_sample}


# ---------- Upper and lower bounds sanity (4 items) -----------------------------------------------

def test_perfect_aggregated_rate_is_zero():
    agg = _score("perfect")
    assert agg["nudge_fire_rate"] == 0.0
    assert agg["nudge_fire_count"] == 0.0


def test_perfect_failure_mode_buckets_all_zero():
    agg = _score("perfect")
    by_mode = agg["by_failure_mode"]
    assert by_mode == {"missed": 0, "wrong_tool": 0, "wrong_args": 0}


def test_all_nudged_aggregated_rate_is_one():
    agg = _score("all_nudged")
    assert agg["nudge_fire_rate"] == 1.0
    # 20 require_tool turns (example 3 + panel 4 + tool_chain 5 + code_review 8)
    # The latter two were introduced by agent_sft phase 1.B and rely on agent_engine DECISIONS §12 to lift scope restrictions.
    assert agg["nudge_fire_count"] == 20.0
    assert agg["require_tool_total"] == 20.0


def test_all_nudged_dominated_by_missed_mode():
    agg = _score("all_nudged")
    by_mode = agg["by_failure_mode"]
    assert by_mode["missed"] == 20
    assert by_mode["wrong_tool"] == 0
    assert by_mode["wrong_args"] == 0


# ---------- mixed intermediate state (3 items) -----------------------------------------------

def test_mixed_aggregated_rate_in_open_interval():
    agg = _score("mixed")
    rate = agg["nudge_fire_rate"]
    assert 0.0 < rate < 1.0


def test_mixed_failure_modes_have_both_missed_and_wrong_tool():
    """Mixed stub is designed to count both missed + wrong_tool (verify that both buckets are active)."""
    agg = _score("mixed")
    by_mode = agg["by_failure_mode"]
    assert by_mode["missed"] > 0
    assert by_mode["wrong_tool"] > 0
    assert by_mode["wrong_args"] == 0  # Phase 1 deferred, constant 0


def test_mixed_strictly_between_perfect_and_all_nudged():
    """Monotonicity: perfect ≤ mixed < all_nudged ∈ {0, x ∈ (0,1), 1}."""
    perfect = _score("perfect")["nudge_fire_rate"]
    mixed = _score("mixed")["nudge_fire_rate"]
    nudged = _score("all_nudged")["nudge_fire_rate"]
    assert perfect == 0.0
    assert nudged == 1.0
    assert perfect < mixed < nudged


# ---------- by_scenario / by_tool breakdown (3 items) -----------------------

def test_by_scenario_vacuous_scenarios_are_none():
    """3 scenarios without require_tool Always None (vacuous) → breakdown explicitly rendered."""
    agg = _score("all_nudged")
    by_scn = agg["by_scenario"]
    assert by_scn["brainstorm"] is None
    assert by_scn["debate"] is None
    assert by_scn["roundtable"] is None
    # The 4 scenarios with require_tool are all 1.0
    assert by_scn["example"] == 1.0
    assert by_scn["panel"] == 1.0
    assert by_scn["tool_chain"] == 1.0
    assert by_scn["code_review"] == 1.0


def test_by_tool_breakdown_contains_three_tools():
    """all_nudged has three keys by_tool: append_section + cast_vote + retrieve_docs.
    retrieve_docs is a non-artifact tool and relies on DECISIONS §12 to enter the require_tool inspection interface."""
    agg = _score("all_nudged")
    by_tool = agg["by_tool"]
    assert set(by_tool.keys()) == {"append_section", "cast_vote", "retrieve_docs"}
    assert by_tool["append_section"] == 1.0
    assert by_tool["cast_vote"] == 1.0
    assert by_tool["retrieve_docs"] == 1.0


def test_perfect_per_sample_rates_correctly_split():
    """per-sample drill-down: perfect next 4 require_tool scenario rate=0, 3 vacuous None."""
    rates = _per_sample_rates("perfect")
    assert rates == {
        "brainstorm": None, "debate": None, "roundtable": None,
        "example": 0.0, "panel": 0.0, "tool_chain": 0.0, "code_review": 0.0,
    }


# ---------- Data contract sanity (2 items) ------------------------------------------

def test_docs_loading_order_matches_gold():
    """gold.jsonl line order: vacuous first (smoke friendly) → require_tool after; require_tool
    Internally sorted in ascending order by turn number (example 3 → panel 4 → tool_chain 5 → code_review 8).
    --limit 1 hits brainstorm (fastest), --limit 7 runs all."""
    docs = list(NudgeFireRate().docs())
    assert [d.id for d in docs] == [
        "brainstorm", "debate", "roundtable", "example", "panel",
        "tool_chain", "code_review",
    ]


def test_higher_is_better_marks_rate_as_lower_better():
    """nudge_fire_rate is an inverse metric (lower is better), contrary to the high = good convention for other project metrics."""
    h = NudgeFireRate().higher_is_better()
    assert h["nudge_fire_rate"] is False
    assert h["nudge_fire_count"] is False
