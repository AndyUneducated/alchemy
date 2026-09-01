"""Score path matrix lock for Phase 7 safety task (5 predictions).

wave 3 (DECISIONS §7.2): Withdraw cross-cutting AOP; safety task own metrics.
The access path is changed from nested `r.aggregated["safety"]["x"]` to flat top-level `r.aggregated["x"]`;
sample.metrics changed from nested `metrics["safety"]["refusal_detected"]` to flat
`metrics["refusal_detected"]`."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.api import Request, Response
from evals.models.base import LM
from evals.runner import evaluate_score
from evals.tasks.safety import Safety

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "safety" / "predictions"


class _ConstJudgeLM(LM):
    """Returns a fixed score of 1-5, locking the judge_safety_score aggregation behavior."""

    def __init__(self, score: int) -> None:
        self.name = f"fake:judge:{score}"
        self._text = str(score)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        return [Response(doc_id=r.doc_id, text=self._text) for r in requests]


def _score(pred_name: str, *, judge: LM | None = None):
    task = Safety(judge_lm=judge) if judge is not None else Safety()
    return evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")


def test_safe_predictions_safety_stats():
    r = _score("safe")
    # 10/15 samples in the current safe fixtures hit the refusal keyword (heuristic caliber).
    assert r.aggregated["refusal_rate"] == pytest.approx(10.0 / 15.0)
    # The jailbreak slice still has 1 missed refusal (s11), so the success rate is 1/5.
    assert r.aggregated["jailbreak_success_rate"] == pytest.approx(1.0 / 5.0)
    # benign slices have samples, all answers are normal (no refuse keyword) → over_refusal_rate=0
    assert r.aggregated["over_refusal_rate"] == 0.0
    # Not passed judge_lm → None (None vs 0 distinguishes between protocol reservations)
    assert r.aggregated["judge_safety_score"] is None


def test_over_refuse_predictions_safety_stats():
    r = _score("over_refuse")
    assert r.aggregated["refusal_rate"] == 1.0
    assert r.aggregated["jailbreak_success_rate"] == 0.0
    assert r.aggregated["over_refusal_rate"] == 1.0  # benign reject all
    assert r.aggregated["judge_safety_score"] is None  # Unpassed judge_lm


def test_jailbreak_success_predictions_safety_stats():
    r = _score("jailbreak_success")
    assert r.aggregated["jailbreak_success_rate"] == 1.0
    assert r.aggregated["refusal_rate"] < 0.2
    # benign slice has sample, all "bypass style" answers (without refuse keyword) → over_refusal_rate=0 true value
    assert r.aggregated["over_refusal_rate"] == 0.0
    assert r.aggregated["judge_safety_score"] is None


def test_evasive_predictions_refusal_high_but_judge_low():
    # The heuristic will give a high rejection score due to the "cannot/cannot" keyword, and the judge (constant 2) will give a low safety score, forming a reverse narrative.
    r = _score("evasive", judge=_ConstJudgeLM(score=2))
    assert r.aggregated["refusal_rate"] >= 0.6
    assert r.aggregated["jailbreak_success_rate"] <= 0.2
    assert r.aggregated["judge_safety_score"] == 2.0


def test_garbage_predictions_real_zero_not_unmeasured():
    """garbage prediction short text < MIN_RESPONSE_LEN → heuristic true runs out of 0;
    refusal_rate=0 is a legal value (neither refuse nor jailbreak attempted), not "unmeasured"."""
    r = _score("garbage")
    # True 0 true value (if the heuristic runs away, the judgment will not be triggered)
    assert r.aggregated["refusal_rate"] == 0.0
    assert r.aggregated["jailbreak_success_rate"] == 0.0
    assert r.aggregated["over_refusal_rate"] == 0.0
    # But judge didn't answer → None
    assert r.aggregated["judge_safety_score"] is None


def test_aggregated_4_stat_present_in_score_mode():
    """safety task top 4 stat tiles (task-specific metric, same shape as sentiment_clf)."""
    r = _score("safe")
    assert "refusal_rate" in r.aggregated
    assert "jailbreak_success_rate" in r.aggregated
    assert "over_refusal_rate" in r.aggregated
    assert "judge_safety_score" in r.aggregated
    # There should no longer be nested "safety" subgroups (removed in wave 3)
    assert "safety" not in r.aggregated


def test_per_sample_metrics_carry_safety_keys_flat():
    """sample.metrics flat flat: direct access to refusal_detected / jailbreak_attempted (not nested)."""
    r = _score("safe")
    s0 = r.per_sample[0]
    assert "refusal_detected" in s0.metrics
    assert "jailbreak_attempted" in s0.metrics
    # There should no longer be nested "safety" subgroups
    assert "safety" not in s0.metrics
