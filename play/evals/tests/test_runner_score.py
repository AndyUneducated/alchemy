"""score path end-to-end: The values of each of the four predictions fall within the expected interval.

These values ​​are also the teaching narrative of the README - test green = README is not lying."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"


def _score(name: str) -> dict[str, float]:
    task = SentimentClf()
    r = evaluate_score(task, PRED_DIR / f"{name}.jsonl")
    assert r.mode == "score"
    assert r.n == 30
    return r.aggregated


def test_score_perfect():
    agg = _score("perfect")
    assert agg["accuracy"] == 1.0
    assert agg["f1_macro"] == 1.0
    assert agg["cohens_kappa"] == 1.0


def test_score_constant_neutral():
    """constant_neutral is the core case of chance-corrected teaching."""
    agg = _score("constant_neutral")
    # 10/30 samples are neutral → accuracy = 1/3
    assert abs(agg["accuracy"] - 1 / 3) < 1e-9
    # macro-F1 = recall=0 → F1_c=0 for the other two categories, only neutral F1=0.5 → macro = 0.5/3
    assert abs(agg["f1_macro"] - 1 / 6) < 1e-9
    # p_o = p_e → kappa is exactly 0 (chance-corrected teaching core)
    assert agg["cohens_kappa"] == 0.0


def test_score_noisy_03_deterministic():
    """seed 0 fixed → value must be fully reproducible."""
    agg = _score("noisy_0.3")
    # Actual values ​​under noise=0.3, seed=0 (10 items per category, total 30)
    assert abs(agg["accuracy"] - 0.8333333333333334) < 1e-9
    assert abs(agg["f1_macro"] - 0.8293460925039872) < 1e-9
    assert abs(agg["cohens_kappa"] - 0.75) < 1e-9
    # kappa < accuracy: part of accuracy is "luck points", and kappa eliminates it
    assert agg["cohens_kappa"] < agg["accuracy"]


def test_score_keyword_rule_middle_ground():
    agg = _score("keyword_rule")
    # Weak baseline: stronger than constant, weaker than noisy
    assert 0.45 <= agg["accuracy"] <= 0.60
    assert 0.20 <= agg["cohens_kappa"] <= 0.40
    assert agg["cohens_kappa"] < agg["accuracy"]


def test_score_limit_parameter():
    task = SentimentClf()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl", limit=10)
    assert r.n == 10
    assert r.aggregated["accuracy"] == 1.0  # perfect is still correct


def test_score_missing_prediction_raises(tmp_path):
    """doc_id is missing in predictions → strictly report an error (phase 1 default behavior)."""
    task = SentimentClf()
    # Only put 1 pred (and the id is deliberately not in gold), 30 gold → the first lookup should not be able to hit
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"id": "sNONE", "prediction": "neutral"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        evaluate_score(task, partial)
