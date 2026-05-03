"""score 路径 end-to-end：四份 predictions 各自的数值落在预期区间.

这些数值也是 README 的教学叙事——test 绿 = README 没说谎。
"""

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
    """constant_neutral 是 chance-corrected 的教学核心案例."""
    agg = _score("constant_neutral")
    # 10/30 样本是 neutral → accuracy = 1/3
    assert abs(agg["accuracy"] - 1 / 3) < 1e-9
    # macro-F1 = 其它两类 recall=0 → F1_c=0，只有 neutral 的 F1=0.5 → macro = 0.5/3
    assert abs(agg["f1_macro"] - 1 / 6) < 1e-9
    # p_o = p_e → kappa 精确为 0（chance-corrected 的教学核心）
    assert agg["cohens_kappa"] == 0.0


def test_score_noisy_03_deterministic():
    """seed 0 固定 → 数值必须完全可复现."""
    agg = _score("noisy_0.3")
    # noise=0.3，seed=0 下实际数值（每类 10 条，总 30）
    assert abs(agg["accuracy"] - 0.8333333333333334) < 1e-9
    assert abs(agg["f1_macro"] - 0.8293460925039872) < 1e-9
    assert abs(agg["cohens_kappa"] - 0.75) < 1e-9
    # kappa < accuracy：accuracy 里有一部分是"运气分"，kappa 剔除了
    assert agg["cohens_kappa"] < agg["accuracy"]


def test_score_keyword_rule_middle_ground():
    agg = _score("keyword_rule")
    # 弱基线：比 constant 强，比 noisy 弱
    assert 0.45 <= agg["accuracy"] <= 0.60
    assert 0.20 <= agg["cohens_kappa"] <= 0.40
    assert agg["cohens_kappa"] < agg["accuracy"]


def test_score_limit_parameter():
    task = SentimentClf()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl", limit=10)
    assert r.n == 10
    assert r.aggregated["accuracy"] == 1.0  # perfect 下仍全对


def test_score_missing_prediction_raises(tmp_path):
    """predictions 里缺 doc_id → 严格报错（phase 1 默认行为）."""
    task = SentimentClf()
    # 只放 1 条 pred（且 id 故意不在 gold 里），gold 30 条 → 第一个 lookup 就应该命中不了
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"id": "sNONE", "prediction": "neutral"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        evaluate_score(task, partial)
