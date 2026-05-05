"""Phase 7 safety task 的 run 路径锁。"""

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


class _ConstJudgeLM(LM):
    def __init__(self, score: int) -> None:
        self.name = f"fake:judge:{score}"
        self._text = str(score)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        return [Response(doc_id=r.doc_id, text=self._text) for r in requests]


def test_run_mock_gold_parity_perfect_predictions():
    """safety task parity：run(mock:gold) 与 score(由 gold 即时生成的 perfect predictions) 相等."""
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "prediction": d.target}, ensure_ascii=False) + "\n")
        path = Path(f.name)
    r_score = evaluate_score(Safety(), path)

    assert "safety" in r_run.aggregated
    assert "safety" in r_score.aggregated
    assert r_run.aggregated["safety"] == r_score.aggregated["safety"]
    # run 路径多 efficiency（call class）；score 无
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated


def test_run_with_judge_outputs_judge_score():
    docs = list(Safety().docs())
    task = Safety(judge_lm=_ConstJudgeLM(score=4))
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert r.aggregated["safety"]["judge_safety_score"] == 4.0


def test_run_safety_subgroup_present_both_modes():
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    assert "safety" in r_run.aggregated
    assert "safety" in r_score.aggregated
