"""Phase 8 iaa_nominal task 的 score 路径矩阵锁（4 份 predictions）.

核心叙事 — kappa paradox：
  - constant_majority：accuracy=0.9 但 cohens_kappa=0；gwet_ac1≈0.89 仍诚实地高
  - noisy_diverging：cohens_kappa mid (~0.26)，但 fleiss_kappa < 0（多 rater 拉平）
  - garbage：所有 kappa 系列 < 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.iaa_nominal import IaaNominal

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "iaa_nominal" / "predictions"


def _score(pred_name: str):
    return evaluate_score(IaaNominal(), PRED_DIR / f"{pred_name}.jsonl")


# ---------- perfect: 上界 sanity ----------

def test_perfect_all_metrics_one():
    r = _score("perfect")
    assert r.aggregated["accuracy"] == 1.0
    assert r.aggregated["balanced_accuracy"] == 1.0
    assert r.aggregated["mcc"] == pytest.approx(1.0)
    assert r.aggregated["f1_macro"] == 1.0
    assert r.aggregated["precision_spam"] == 1.0
    assert r.aggregated["recall_spam"] == 1.0
    assert r.aggregated["f1_spam"] == 1.0
    assert r.aggregated["cohens_kappa"] == pytest.approx(1.0)
    assert r.aggregated["scott_pi"] == pytest.approx(1.0)
    assert r.aggregated["gwet_ac1"] == pytest.approx(1.0)
    assert r.aggregated["fleiss_kappa"] == pytest.approx(1.0)
    assert r.aggregated["krippendorff_alpha"] == pytest.approx(1.0)


# ---------- constant_majority: kappa paradox 主战场 ----------

def test_constant_majority_kappa_paradox_acc_high_kappa_zero():
    """核心断言：acc=0.9 高（看似良好）但 cohens_kappa=0（实际 = 多数类基线）."""
    r = _score("constant_majority")
    assert r.aggregated["accuracy"] == pytest.approx(0.9)
    assert r.aggregated["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)


def test_constant_majority_gwet_ac1_paradox_antidote():
    """kappa paradox 解药 1：gwet_ac1≈0.89 仍诚实地高（与 cohens_kappa=0 形成对比）."""
    r = _score("constant_majority")
    assert r.aggregated["gwet_ac1"] == pytest.approx(0.805 / 0.905, abs=1e-9)
    assert r.aggregated["gwet_ac1"] > 0.85


def test_constant_majority_minority_class_collapse():
    """kappa paradox 副产品：少数类（spam）的 precision/recall/f1 全 0；
    balanced_accuracy / mcc / f1_macro 也跌到 0/0.5（揭示真实失明）."""
    r = _score("constant_majority")
    assert r.aggregated["precision_spam"] == 0.0
    assert r.aggregated["recall_spam"] == 0.0
    assert r.aggregated["f1_spam"] == 0.0
    assert r.aggregated["mcc"] == pytest.approx(0.0, abs=1e-9)
    assert r.aggregated["balanced_accuracy"] == pytest.approx(0.5)


def test_constant_majority_multi_rater_low():
    """3 raters 全押多数类 → fleiss / krippendorff 也接近 0（与 cohens_kappa 同源失明）."""
    r = _score("constant_majority")
    assert abs(r.aggregated["fleiss_kappa"]) < 0.05
    assert abs(r.aggregated["krippendorff_alpha"]) < 0.05


# ---------- noisy_diverging: 多 rater 拉平叙事 ----------

def test_noisy_diverging_cohen_mid_fleiss_negative():
    """反向叙事：2-rater (gold vs pred) cohens_kappa mid (~0.26)，
    但 4 ratings (gold + 3 raters) fleiss_kappa <0 — 多 rater 暴露内部分歧."""
    r = _score("noisy_diverging")
    assert 0.15 < r.aggregated["cohens_kappa"] < 0.35
    assert 0.55 < r.aggregated["gwet_ac1"] < 0.75
    assert r.aggregated["fleiss_kappa"] < 0
    assert r.aggregated["krippendorff_alpha"] < 0


def test_noisy_diverging_accuracy_around_077():
    """21 ham 对 + 2 spam 对 = 23/30，acc ≈ 0.767."""
    r = _score("noisy_diverging")
    assert r.aggregated["accuracy"] == pytest.approx(23.0 / 30.0, abs=1e-9)


# ---------- garbage: 下界 sanity ----------

def test_garbage_acc_low_all_kappas_negative():
    """garbage：30% 准确 + 全 kappa 系列 < 0（盲眼或反向预测）."""
    r = _score("garbage")
    assert r.aggregated["accuracy"] == pytest.approx(0.3, abs=1e-9)
    assert r.aggregated["cohens_kappa"] < 0
    assert r.aggregated["scott_pi"] < 0
    assert r.aggregated["gwet_ac1"] < 0
    assert r.aggregated["fleiss_kappa"] < 0
    assert r.aggregated["krippendorff_alpha"] < 0


# ---------- 结构断言（整套指标齐 + confusion matrix 形态）----------

def test_aggregated_has_15_stats():
    """aggregation 返 15 个键（9 classification + 3 agreement 2-rater + 2 multi-rater
    + 1 _confusion_matrix）—— 防 stat 丢失退化."""
    r = _score("perfect")
    expected = {
        "accuracy", "balanced_accuracy", "mcc",
        "f1_micro", "f1_macro", "f_beta_2",
        "precision_spam", "recall_spam", "f1_spam",
        "cohens_kappa", "scott_pi", "gwet_ac1",
        "fleiss_kappa", "krippendorff_alpha",
        "_confusion_matrix",
    }
    assert expected.issubset(r.aggregated.keys())


def test_confusion_matrix_nested_form():
    """`_confusion_matrix` 嵌套 dict 形态：{gold: {pred: count}}（诊断辅助）."""
    r = _score("constant_majority")
    cm = r.aggregated["_confusion_matrix"]
    # 27 ham gold → 全部预测 ham；3 spam gold → 全部预测 ham
    assert cm["ham"]["ham"] == 27
    assert cm["ham"]["spam"] == 0
    assert cm["spam"]["ham"] == 3
    assert cm["spam"]["spam"] == 0
