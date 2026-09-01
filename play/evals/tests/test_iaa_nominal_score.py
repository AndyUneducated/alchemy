"""Phase 8 iaa_nominal task's score path matrix lock (4 predictions).

Core narrative — kappa paradox:
  - constant_majority: accuracy=0.9 but cohens_kappa=0; gwet_ac1≈0.89 is still honestly high
  - noisy_diverging: cohens_kappa mid (~0.26), but fleiss_kappa < 0 (multi rater leveling)
  - garbage: all kappa series < 0"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.iaa_nominal import IaaNominal

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "iaa_nominal" / "predictions"


def _score(pred_name: str):
    return evaluate_score(IaaNominal(), PRED_DIR / f"{pred_name}.jsonl")


# ---------- perfect: upper realm sanity ----------

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


# ---------- constant_majority: kappa paradox main battlefield ----------

def test_constant_majority_kappa_paradox_acc_high_kappa_zero():
    """Core assertion: acc=0.9 is high (seems good) but cohens_kappa=0 (actual = majority class baseline)."""
    r = _score("constant_majority")
    assert r.aggregated["accuracy"] == pytest.approx(0.9)
    assert r.aggregated["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)


def test_constant_majority_gwet_ac1_paradox_antidote():
    """Antidote 1 to the kappa paradox: gwet_ac1≈0.89 is still honestly high (in contrast to cohens_kappa=0)."""
    r = _score("constant_majority")
    assert r.aggregated["gwet_ac1"] == pytest.approx(0.805 / 0.905, abs=1e-9)
    assert r.aggregated["gwet_ac1"] > 0.85


def test_constant_majority_minority_class_collapse():
    """By-product of kappa paradox: precision/recall/f1 of minority class (spam) are all 0;
    balanced_accuracy / mcc / f1_macro also dropped to 0/0.5 (revealing true blindness)."""
    r = _score("constant_majority")
    assert r.aggregated["precision_spam"] == 0.0
    assert r.aggregated["recall_spam"] == 0.0
    assert r.aggregated["f1_spam"] == 0.0
    assert r.aggregated["mcc"] == pytest.approx(0.0, abs=1e-9)
    assert r.aggregated["balanced_accuracy"] == pytest.approx(0.5)


def test_constant_majority_multi_rater_low():
    """3 raters all-in majority class → fleiss / krippendorff also close to 0 (cohens_kappa cognate blindness)."""
    r = _score("constant_majority")
    assert abs(r.aggregated["fleiss_kappa"]) < 0.05
    assert abs(r.aggregated["krippendorff_alpha"]) < 0.05


# ---------- noisy_diverging: Multiple raters to level the narrative ----------

def test_noisy_diverging_cohen_mid_fleiss_negative():
    """Reverse narrative: 2-rater (gold vs pred) cohens_kappa mid (~0.26),
    But 4 ratings (gold + 3 raters) fleiss_kappa <0 — multiple raters expose internal disagreements."""
    r = _score("noisy_diverging")
    assert 0.15 < r.aggregated["cohens_kappa"] < 0.35
    assert 0.55 < r.aggregated["gwet_ac1"] < 0.75
    assert r.aggregated["fleiss_kappa"] < 0
    assert r.aggregated["krippendorff_alpha"] < 0


def test_noisy_diverging_accuracy_around_077():
    """21 ham pairs + 2 spam pairs = 23/30, acc ≈ 0.767."""
    r = _score("noisy_diverging")
    assert r.aggregated["accuracy"] == pytest.approx(23.0 / 30.0, abs=1e-9)


# ---------- garbage: lower bound sanity ----------

def test_garbage_acc_low_all_kappas_negative():
    """garbage: 30% accurate + full kappa series < 0 (blind or backward prediction)."""
    r = _score("garbage")
    assert r.aggregated["accuracy"] == pytest.approx(0.3, abs=1e-9)
    assert r.aggregated["cohens_kappa"] < 0
    assert r.aggregated["scott_pi"] < 0
    assert r.aggregated["gwet_ac1"] < 0
    assert r.aggregated["fleiss_kappa"] < 0
    assert r.aggregated["krippendorff_alpha"] < 0


# ---------- Structural assertion (complete set of indicators + confusion matrix form) ----------

def test_aggregated_has_15_stats():
    """aggregation returns 15 keys (9 classification + 3 agreement 2-rater + 2 multi-rater
    + 1 _confusion_matrix) - to prevent stat loss and degradation."""
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
    """`_confusion_matrix` Nested dict shape: {gold: {pred: count}} (diagnostic aid)."""
    r = _score("constant_majority")
    cm = r.aggregated["_confusion_matrix"]
    # 27 ham gold → all predictions ham; 3 spam gold → all predictions ham
    assert cm["ham"]["ham"] == 27
    assert cm["ham"]["spam"] == 0
    assert cm["spam"]["ham"] == 3
    assert cm["spam"]["spam"] == 0
