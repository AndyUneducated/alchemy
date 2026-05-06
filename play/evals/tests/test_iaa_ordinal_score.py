"""Phase 8 iaa_ordinal task 的 score 路径矩阵锁（4 份 predictions）.

核心叙事 — ordinal-aware metric 救场 nominal κ 失明：
  - off_by_one：accuracy=0 + cohens_kappa=-0.25 (nominal 全失明)；但
    weighted_kappa_quadratic≈0.71 + pearson≈0.83 + lins_ccc≈0.71 (ordinal-aware 救场)
  - garbage：pred = 6−gold (perfect inverse) → weighted_quad=-1, pearson=-1, ccc=-1
    （ordinal-aware 直接抓出反向，cohens_kappa 仍是 0 paradox 复刻）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.iaa_ordinal import IaaOrdinal

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "iaa_ordinal" / "predictions"


def _score(pred_name: str):
    return evaluate_score(IaaOrdinal(), PRED_DIR / f"{pred_name}.jsonl")


# ---------- perfect: 上界 sanity ----------

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


# ---------- off_by_one: 核心叙事（nominal 失明 / ordinal 救场）----------

def test_off_by_one_nominal_failure():
    """exact accuracy = 0 + nominal cohens_kappa = -0.25（exact 与 nominal κ 全部失明）."""
    r = _score("off_by_one")
    assert r.aggregated["accuracy"] == 0.0
    assert r.aggregated["cohens_kappa"] == pytest.approx(-0.25, abs=1e-9)


def test_off_by_one_ordinal_aware_rescue():
    """ordinal-aware metric 救场：weighted_quad≈0.71 / pearson≈0.83 / spearman≈0.82 /
    kendall≈0.74 / ccc≈0.71."""
    r = _score("off_by_one")
    assert r.aggregated["weighted_kappa_quadratic"] == pytest.approx(0.7058823529411764, abs=1e-9)
    assert r.aggregated["weighted_kappa_linear"] > 0.3
    assert r.aggregated["pearson_r"] > 0.8
    assert r.aggregated["spearman_rho"] > 0.8
    assert r.aggregated["kendall_tau"] > 0.7
    assert r.aggregated["lins_ccc"] == pytest.approx(0.7058823529411764, abs=1e-9)


def test_off_by_one_multi_rater_ordinal_high():
    """raters 与 prediction 同步偏 1 → 多 rater ordinal/interval/ICC 仍高
    (raters 之间一致，仅与 gold 偏 1)."""
    r = _score("off_by_one")
    assert r.aggregated["fleiss_kappa"] > 0.3
    assert r.aggregated["krippendorff_alpha_ordinal"] > 0.8
    assert r.aggregated["krippendorff_alpha_interval"] > 0.8
    assert r.aggregated["icc_1_1"] > 0.8


# ---------- random: 下界 sanity ----------

def test_random_near_zero_correlation():
    """random：accuracy ≈ 1/5 + 全 kappa/correlation/ccc 都接近 0 (无信号 baseline)."""
    r = _score("random")
    assert r.aggregated["accuracy"] == pytest.approx(0.2, abs=1e-9)
    assert abs(r.aggregated["cohens_kappa"]) < 0.1
    assert abs(r.aggregated["weighted_kappa_quadratic"]) < 0.1
    assert abs(r.aggregated["pearson_r"]) < 0.15
    assert abs(r.aggregated["spearman_rho"]) < 0.15
    assert abs(r.aggregated["lins_ccc"]) < 0.1
    assert abs(r.aggregated["krippendorff_alpha_ordinal"]) < 0.1


# ---------- garbage: 极端反向 sanity（perfect inverse 复刻 paradox）----------

def test_garbage_inverse_ordinal_aware_catches_negative():
    """garbage = 6−gold (perfect inverse)：ordinal-aware 直接抓出 perfect negative
    (weighted_quad=-1 / pearson=-1 / spearman=-1 / lins_ccc=-1)."""
    r = _score("garbage")
    assert r.aggregated["weighted_kappa_quadratic"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["weighted_kappa_linear"] == pytest.approx(-0.5, abs=1e-9)
    assert r.aggregated["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["spearman_rho"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["kendall_tau"] == pytest.approx(-1.0, abs=1e-9)
    assert r.aggregated["lins_ccc"] == pytest.approx(-1.0, abs=1e-9)


def test_garbage_cohens_kappa_paradox_replay():
    """nominal cohens_kappa = 0 (paradox 复刻 — 即使 perfect inverse，nominal 仍盲)；
    accuracy = 1/5 (gold=3 → pred=3 自匹配)."""
    r = _score("garbage")
    assert r.aggregated["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)
    assert r.aggregated["accuracy"] == pytest.approx(0.2, abs=1e-9)


def test_garbage_multi_rater_negative():
    """随机 raters → 多 rater 都 < 0 (无 rater 间一致信号)."""
    r = _score("garbage")
    assert r.aggregated["krippendorff_alpha_ordinal"] < 0
    assert r.aggregated["krippendorff_alpha_interval"] < 0
    assert r.aggregated["icc_1_1"] < 0


# ---------- 结构断言 ----------

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
