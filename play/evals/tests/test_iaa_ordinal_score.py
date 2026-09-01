"""Phase 8 iaa_ordinal task's score path matrix lock (4 predictions).

core narrative — ordinal-aware metric rescue nominal κ blindness:
  - off_by_one: accuracy=0 + cohens_kappa=-0.25 (nominal total blindness); but
    weighted_kappa_quadratic≈0.71 + pearson≈0.83 + lins_ccc≈0.71 (ordinal-aware rescue)
  - garbage: pred = 6−gold (perfect inverse) → weighted_quad=-1, pearson=-1, ccc=-1
    (ordinal-aware directly captures the reverse, and Cohens_kappa is still 0 paradox replica)"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.iaa_ordinal import IaaOrdinal

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "iaa_ordinal" / "predictions"


def _score(pred_name: str):
    return evaluate_score(IaaOrdinal(), PRED_DIR / f"{pred_name}.jsonl")


# ---------- perfect: upper realm sanity ----------

def test_perfect_all_metrics_one():
    r = _score("perfect")
    for k in [
        "accuracy", "cohens_kappa",
        "weighted_kappa_linear", "weighted_kappa_quadratic",
        "pearson_r", "spearman_rho", "kendall_tau", "lins_ccc",
        "fleiss_kappa", "krippendorff_alpha_ordinal",
        "krippendorff_alpha_interval", "icc_1_1",
    ]:
        assert r.aggregated[k] == pytest.approx(1.0), f"{k}={r.aggregated[k]} != 1.0"


# ---------- off_by_one: core narrative (nominal blindness / ordinal rescue) ----------

def test_off_by_one_nominal_failure():
    """exact accuracy = 0 + nominal cohens_kappa = -0.25 (exact and nominal κ are all blind)."""
    r = _score("off_by_one")
    assert r.aggregated["accuracy"] == 0.0
    assert r.aggregated["cohens_kappa"] == pytest.approx(-0.25, abs=1e-9)


def test_off_by_one_ordinal_aware_rescue():
    """ordinal-aware metric rescue: weighted_quad≈0.71 / pearson≈0.83 / spearman≈0.82 /
    kendall≈0.74 / ccc≈0.71."""
    r = _score("off_by_one")
    assert r.aggregated["weighted_kappa_quadratic"] == pytest.approx(0.7058823529411764, abs=1e-9)
    assert r.aggregated["weighted_kappa_linear"] > 0.3
    assert r.aggregated["pearson_r"] > 0.8
    assert r.aggregated["spearman_rho"] > 0.8
    assert r.aggregated["kendall_tau"] > 0.7
    assert r.aggregated["lins_ccc"] == pytest.approx(0.7058823529411764, abs=1e-9)


def test_off_by_one_multi_rater_ordinal_high():
    """Raters and prediction synchronization bias 1 → multiple rater ordinal/interval/ICC is still high
    (The raters are consistent among themselves, only biased by 1 with gold)."""
    r = _score("off_by_one")
    assert r.aggregated["fleiss_kappa"] > 0.3
    assert r.aggregated["krippendorff_alpha_ordinal"] > 0.8
    assert r.aggregated["krippendorff_alpha_interval"] > 0.8
    assert r.aggregated["icc_1_1"] > 0.8


# ---------- random: lower bound sanity ----------

def test_random_near_zero_correlation():
    """random: accuracy ≈ 1/5 + all kappa/correlation/ccc are close to 0 (no signal baseline)."""
    r = _score("random")
    assert r.aggregated["accuracy"] == pytest.approx(0.2, abs=1e-9)
    assert abs(r.aggregated["cohens_kappa"]) < 0.1
    assert abs(r.aggregated["weighted_kappa_quadratic"]) < 0.1
    assert abs(r.aggregated["pearson_r"]) < 0.15
    assert abs(r.aggregated["spearman_rho"]) < 0.15
    assert abs(r.aggregated["lins_ccc"]) < 0.1
    assert abs(r.aggregated["krippendorff_alpha_ordinal"]) < 0.1


# ---------- garbage: extreme reverse sanity (perfect inverse paradox) ----------

def test_garbage_inverse_ordinal_aware_catches_negative():
    """garbage = 6−gold (perfect inverse): ordinal-aware directly captures the perfect negative
    (weighted_quad=-1 / pearson=-1 / spearman=-1 / lins_ccc=-1)."""
    r = _score("garbage")
    assert r.aggregated["weighted_kappa_quadratic"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["weighted_kappa_linear"] == pytest.approx(-0.5, abs=1e-9)
    assert r.aggregated["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["spearman_rho"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["kendall_tau"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["lins_ccc"] == pytest.approx(-1.0, abs=1e-9)


def test_garbage_cohens_kappa_paradox_replay():
    """nominal cohens_kappa = 0 (paradox inverse — even with perfect inverse, nominal is still blind);
    accuracy = 1/5 (gold=3 → pred=3 self-matching)."""
    r = _score("garbage")
    assert r.aggregated["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)
    assert r.aggregated["accuracy"] == pytest.approx(0.2, abs=1e-9)


def test_garbage_multi_rater_negative():
    """Random raters → multiple raters < 0 (no consistent signal among raters)."""
    r = _score("garbage")
    assert r.aggregated["krippendorff_alpha_ordinal"] < 0
    assert r.aggregated["krippendorff_alpha_interval"] < 0
    assert r.aggregated["icc_1_1"] < 0


# ---------- Structure assertion ----------

def test_aggregated_has_12_stats():
    """12 stat：1 exact + 3 agreement (nominal/linear/quadratic) + 3 corr +
    1 ccc + 4 multi-rater (fleiss + krip×2 + icc11)."""
    r = _score("perfect")
    expected = {
        "accuracy", "cohens_kappa",
        "weighted_kappa_linear", "weighted_kappa_quadratic",
        "pearson_r", "spearman_rho", "kendall_tau", "lins_ccc",
        "fleiss_kappa", "krippendorff_alpha_ordinal",
        "krippendorff_alpha_interval", "icc_1_1",
    }
    assert expected.issubset(r.aggregated.keys())
