"""Run path lock for Phase 7 safety task.

wave 3 (DECISIONS §7.2): safety = standalone task; aggregated top level flat tile
(refusal_rate/jailbreak_success_rate/over_refusal_rate/judge_safety_score),
No longer nested under `aggregated["safety"]` subgroup."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evals.api import Request, Response
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.safety import Safety

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "safety" / "predictions"

# 4 stat top-level key collection of safety task (flat tiles from wave 3)
_SAFETY_4_STAT_KEYS = {
    "refusal_rate",
    "jailbreak_success_rate",
    "over_refusal_rate",
    "judge_safety_score",
}


class _ConstJudgeLM(LM):
    def __init__(self, score: int) -> None:
        self.name = f"fake:judge:{score}"
        self._text = str(score)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        return [Response(doc_id=r.doc_id, text=self._text) for r in requests]


def test_run_mock_gold_parity_perfect_predictions():
    """safety task parity: run(mock:gold) is equal to score(perfect predictions generated on the fly by gold)."""
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "prediction": d.target}, ensure_ascii=False) + "\n")
        path = Path(f.name)
    r_score = evaluate_score(Safety(), path)

    # 4 stat exists at the top level of tiles
    assert _SAFETY_4_STAT_KEYS <= r_run.aggregated.keys()
    assert _SAFETY_4_STAT_KEYS <= r_score.aggregated.keys()

    # task-specific metrics are equal in dual mode
    for k in _SAFETY_4_STAT_KEYS:
        assert r_run.aggregated[k] == r_score.aggregated[k], f"parity mismatch on {k!r}"

    # There are many run paths efficiency (call class); score none
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated


def test_run_with_judge_outputs_judge_score():
    docs = list(Safety().docs())
    task = Safety(judge_lm=_ConstJudgeLM(score=4))
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert r.aggregated["judge_safety_score"] == 4.0


def test_run_safety_4_stat_present_both_modes():
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    assert _SAFETY_4_STAT_KEYS <= r_run.aggregated.keys()
    assert _SAFETY_4_STAT_KEYS <= r_score.aggregated.keys()


def test_safety_aggregated_subgroup_no_longer_present():
    """wave 3 decisive lock: aggregated no more nested 'safety' subgroups (task-specific tiling)."""
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    assert "safety" not in r_run.aggregated
    assert "safety" not in r_score.aggregated


def test_safety_self_handles_long_answer_correctly():
    """A1 wave 3 fixes the core: safety task runs heuristic by itself (it has nothing to do with cross-cutting AOP deletion),
    Long answers still work normally within the safety task refusal_detected / jailbreak_attempted - this lock
    The function of the safety task itself is not affected by the deletion of cross-cutting in wave 3."""
    r = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    # There are at least some samples refusal_detected=1 (safe.jsonl contains refusal samples)
    refusal_set = {s.metrics.get("refusal_detected") for s in r.per_sample}
    assert 1.0 in refusal_set
