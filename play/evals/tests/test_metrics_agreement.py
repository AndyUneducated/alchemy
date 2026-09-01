"""metrics/agreement.py unit lock: 4 hand calculations + 1 shared helper.

Coverage:
  - scott_pi / gwet_ac1: textbook + kappa paradox two-way narrative (Gwet is still normal when Cohen κ ≈ 0)
  - lins_ccc: perfect / shift penalty / negative / degeneracy constant
  - icc_1_1: perfect / anti-correlation / intermediate value + uneven boundary
  - build_rater_matrix: with/without gold + missing raters fail-loud"""

from __future__ import annotations

import pytest

from evals.api import SampleResult
from evals.metrics.agreement import (
    build_rater_matrix,
    gwet_ac1,
    icc_1_1,
    lins_ccc,
    scott_pi,
)


# ---------- scott_pi ----------

def test_scott_pi_perfect_agreement():
    assert scott_pi([1, 1, 0, 0, 1], [1, 1, 0, 0, 1]) == pytest.approx(1.0)


def test_scott_pi_textbook_binary():
    """5 samples binary, Po=0.6, combined margin q_1=0.6 q_0=0.4 → Pe=0.52, π=0.08/0.48."""
    y1 = [1, 1, 1, 0, 0]
    y2 = [1, 1, 0, 1, 0]
    assert scott_pi(y1, y2) == pytest.approx(0.08 / 0.48, abs=1e-9)


def test_scott_pi_kappa_paradox_near_zero():
    """Highly imbalanced 90/10 + constant majority → π close to 0 (paradox blindness)."""
    gold = ["a"] * 9 + ["b"]
    pred = ["a"] * 10
    pi = scott_pi(gold, pred)
    # Po=0.9, q_a=0.95, q_b=0.05, Pe=0.905, π≈-0.0526
    assert pi < 0.1


def test_scott_pi_single_class_returns_one():
    """Boundary: Both parties are all-in on the same category → Pe=1 and degenerates back to 1.0 (ZeroDivisionError is not thrown)."""
    assert scott_pi(["x"] * 5, ["x"] * 5) == 1.0


def test_scott_pi_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        scott_pi([1, 2], [1, 2, 3])


def test_scott_pi_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        scott_pi([], [])


# ---------- gwet_ac1 ----------

def test_gwet_ac1_perfect_agreement():
    assert gwet_ac1([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_gwet_ac1_kappa_paradox_high():
    """Kappa paradox antidote 1: Same as 90/10 imbalanced data, AC1 is still honest ≈ 0.89."""
    gold = ["a"] * 9 + ["b"]
    pred = ["a"] * 10
    # Po=0.9, K=2, q_a=0.95, q_b=0.05, Pe=(0.95*0.05+0.05*0.95)/1=0.095
    # AC1 = (0.9 - 0.095)/(1 - 0.095) = 0.805/0.905 ≈ 0.8895
    ac1 = gwet_ac1(gold, pred)
    assert ac1 == pytest.approx(0.805 / 0.905, abs=1e-9)
    assert ac1 > 0.7


def test_gwet_ac1_binary_textbook():
    """Same as scott_pi binary example: K=2, q_1=0.6, q_0=0.4 → Pe=0.48, AC1=0.12/0.52."""
    y1 = [1, 1, 1, 0, 0]
    y2 = [1, 1, 0, 1, 0]
    assert gwet_ac1(y1, y2) == pytest.approx(0.12 / 0.52, abs=1e-9)


def test_gwet_ac1_single_class_returns_one():
    """Single class (K=1) → returns Po (theoretically Po=1, consistent with the convention of Gwet paper)."""
    assert gwet_ac1(["x"] * 5, ["x"] * 5) == 1.0


def test_gwet_ac1_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        gwet_ac1([1, 2], [1, 2, 3])


# ---------- lins_ccc ----------

def test_lins_ccc_perfect_match():
    assert lins_ccc([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_lins_ccc_shift_penalty():
    """Perfect linear relationship + shift 1 → CCC < 1 (while pearson r still = 1)."""
    y1 = [1, 2, 3, 4, 5]
    y2 = [2, 3, 4, 5, 6]
    # mean1=3, mean2=4, pop var=2 each, cov=2
    # CCC = 2*2 / (2 + 2 + 1) = 0.8
    assert lins_ccc(y1, y2) == pytest.approx(0.8, abs=1e-9)


def test_lins_ccc_negative_correlation():
    """Perfect anticorrelation → CCC = -1.0."""
    y1 = [1, 2, 3, 4, 5]
    y2 = [5, 4, 3, 2, 1]
    # mean1=mean2=3, var=2 each, cov=-2
    # CCC = -4/4 = -1.0
    assert lins_ccc(y1, y2) == pytest.approx(-1.0, abs=1e-9)


def test_lins_ccc_constant_equal_returns_one():
    """Both sides have the same constant → CCC=1.0 (denom=0 degenerate protocol)."""
    assert lins_ccc([3, 3, 3, 3], [3, 3, 3, 3]) == pytest.approx(1.0)


def test_lins_ccc_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        lins_ccc([1.0, 2.0], [1.0])


def test_lins_ccc_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        lins_ccc([], [])


# ---------- icc_1_1 ----------

def test_icc_1_1_perfect_agreement():
    """Perfect agreement: both raters score equally for each subject → BMS>0, WMS=0, ICC=1."""
    matrix = [[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
    assert icc_1_1(matrix) == pytest.approx(1.0)


def test_icc_1_1_perfect_disagreement_negative():
    """The two raters of each subject are completely opposite, but the subject mean value=3 → BMS=0, WMS>0, ICC=-1."""
    matrix = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]]
    assert icc_1_1(matrix) == pytest.approx(-1.0, abs=1e-9)


def test_icc_1_1_intermediate_textbook():
    """Partial agreement: 4.1/4.9 ≈ 0.8367 (hand calculation BMS=4.5 / WMS=0.4 / k=2)."""
    matrix = [[5, 4], [4, 5], [3, 3], [2, 1], [1, 2]]
    icc = icc_1_1(matrix)
    assert icc == pytest.approx(4.1 / 4.9, abs=1e-9)


def test_icc_1_1_constant_returns_one():
    """All values ​​are equal → denom=0 degenerates back to 1.0."""
    assert icc_1_1([[3, 3], [3, 3], [3, 3]]) == 1.0


def test_icc_1_1_uneven_rows_raises():
    with pytest.raises(ValueError, match="row .* length"):
        icc_1_1([[1, 2, 3], [4, 5]])


def test_icc_1_1_too_few_subjects_raises():
    with pytest.raises(ValueError, match="subjects"):
        icc_1_1([[1, 2]])


def test_icc_1_1_too_few_raters_raises():
    with pytest.raises(ValueError, match="raters"):
        icc_1_1([[1], [2], [3]])


# ---------- build_rater_matrix ----------

def _sr(doc_id: str, target: str, raters: list[str] | None) -> SampleResult:
    artifacts: dict = {}
    if raters is not None:
        artifacts["raters"] = raters
    return SampleResult(doc_id=doc_id, prediction="", target=target, metrics={}, artifacts=artifacts)


def test_build_rater_matrix_with_gold():
    """include_gold=True: Each line [target, *raters], width K+1."""
    srs = [_sr("a", "1", ["1", "1", "0"]), _sr("b", "0", ["0", "0", "1"])]
    m = build_rater_matrix(srs, include_gold=True)
    assert m == [["1", "1", "1", "0"], ["0", "0", "0", "1"]]


def test_build_rater_matrix_without_gold():
    """include_gold=False: only raters per line, K wide."""
    srs = [_sr("a", "1", ["1", "1", "0"])]
    m = build_rater_matrix(srs, include_gold=False)
    assert m == [["1", "1", "0"]]


def test_build_rater_matrix_empty_returns_empty():
    assert build_rater_matrix([]) == []


def test_build_rater_matrix_missing_raters_raises():
    """Contract violation (artifacts missing raters) → fail-loud (lm-eval philosophy)."""
    sr = _sr("x", "1", raters=None)
    with pytest.raises(ValueError, match="raters"):
        build_rater_matrix([sr])


def test_build_rater_matrix_uneven_raters_raises():
    """The lengths of each sample raters are inconsistent → ValueError."""
    srs = [_sr("a", "1", ["1", "1", "0"]), _sr("b", "0", ["0", "1"])]
    with pytest.raises(ValueError, match="length"):
        build_rater_matrix(srs)
