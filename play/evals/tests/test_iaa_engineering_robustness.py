"""Engineering robustness regression for Phase 8 IAA tasks - locking sklearn / krippendorff / scipy direct tuning
No longer raise / emit NaN on degenerate inputs.

Drive two real failure paths (no real LM required):

1. **score path + `--limit` is reduced to only a single category**: the first 5 lines of predictions/constant_majority.jsonl
   full ham → sklearn binary metrics (`precision/recall/f1/fbeta` with `pos_label='spam'`)
   Originally `ValueError: pos_label=spam is not a valid label`, krippendorff Originally
   `ValueError: There has to be more than one value in the domain`;
2. **run path (`output_type='none'`)**: runner gives placeholder Response(doc_id=..) → process_results
   See pred="" → trigger sklearn multiclass-vs-binary error + scipy correlation NaN
   Infect downstream (json.dumps writing NaN is illegal JSON, and running JSON_EXTRACT will be corrupted).

Repair strategy (DECISIONS §8 engineering cover):
  - per-class metrics plus `_pos_label_present(yt, yp)` short circuit;
  - krippendorff adds "<2 unique value" short circuit;
  - All possible NaN metrics (cohens_kappa / weighted_kappa* / pearsonr / spearmanr /
    kendalltau / fleiss_kappa / krippendorff / icc_1_1) package `_nan_to_zero`.

Either guard fails → JSON serialization emit NaN or call sklearn raise → test fail-loud."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import pytest

from evals.api import Doc, Request, Response
from evals.models.base import LM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks import iaa_nominal as _iaa_nominal_mod  # noqa: F401  (registry side effects)
from evals.tasks import iaa_ordinal as _iaa_ordinal_mod  # noqa: F401
from evals.tasks.iaa_nominal import IaaNominal
from evals.tasks.iaa_ordinal import IaaOrdinal


class _UnusedLM(LM):
    """The runner under the output_type='none' task should not trigger generate_until; any call = project contract broken."""

    def __init__(self) -> None:
        self.name = "unused"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        raise AssertionError(
            f"output_type='none' branch must not invoke LM (got {len(requests)} reqs)"
        )


def _assert_aggregated_is_finite_json(aggregated: dict) -> None:
    """All leaf values ​​must be finite floats (no NaN/Inf), and json.dumps does not rely on allow_nan."""
    flat: list[tuple[str, float]] = []

    def _walk(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (int, float)):
            flat.append((prefix, float(obj)))

    _walk("", aggregated)
    for path, val in flat:
        assert math.isfinite(val), f"non-finite metric leaked: {path} = {val!r}"
    json.dumps(aggregated, allow_nan=False)


# ---------- iaa_nominal: score path with --limit small (single-class slice) ----------


def test_iaa_nominal_score_limit_single_class_does_not_raise():
    """After `--limit 5` constant_majority preds only have 5 hams → sklearn binary metrics
    + krippendorff Originally, both paths were fried. After repair, a finite value must be given."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "constant_majority.jsonl"
    )
    r = evaluate_score(task, preds, limit=5)
    _assert_aggregated_is_finite_json(r.aggregated)
    # Single class ham + full ham prediction: accuracy=1, spam-face metrics=0 (pos_label absent short circuit)
    assert r.aggregated["accuracy"] == 1.0
    assert r.aggregated["precision_spam"] == 0.0
    assert r.aggregated["recall_spam"] == 0.0
    assert r.aggregated["f1_spam"] == 0.0
    assert r.aggregated["f_beta_2"] == 0.0
    # Full single class → degenerate 1.0 (scott_pi/gwet_ac1 single class convention) / 0.0 (kappa NaN→0)
    assert r.aggregated["scott_pi"] == 1.0
    assert r.aggregated["gwet_ac1"] == 1.0
    assert r.aggregated["cohens_kappa"] == 0.0  # NaN→0 (Pe=1 degenerate)
    assert r.aggregated["krippendorff_alpha"] == 0.0  # single value field short circuit


def test_iaa_nominal_score_limit_single_class_metrics_keys_complete():
    """The degenerate path must still give all 14 first-class scalar + 1 _confusion_matrix dict."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "constant_majority.jsonl"
    )
    r = evaluate_score(task, preds, limit=5)
    expected = {
        "accuracy", "balanced_accuracy", "mcc", "f1_micro", "f1_macro", "f_beta_2",
        "precision_spam", "recall_spam", "f1_spam",
        "cohens_kappa", "scott_pi", "gwet_ac1", "fleiss_kappa", "krippendorff_alpha",
        "_confusion_matrix",
    }
    assert expected.issubset(set(r.aggregated))


# ---------- iaa_nominal: run path (output_type='none') ----------


def test_iaa_nominal_run_path_does_not_raise_or_call_lm():
    """run path: pred='' triggers sklearn multiclass + binary conflicts on all 30 docs;
    Must be short-circuited by _pos_label_present + NaN hidden + LM has not been adjusted once."""
    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == 30
    # The placeholders pred and gold are all inconsistent: accuracy=0, sanity series 0
    assert r.aggregated["accuracy"] == 0.0
    assert r.aggregated["f_beta_2"] == 0.0
    assert r.aggregated["precision_spam"] == 0.0
    # cohens_kappa degenerates → NaN → us → 0 when sklearn Pe degenerates in pred all ""
    assert r.aggregated["cohens_kappa"] == 0.0
    assert r.aggregated["fleiss_kappa"] == 0.0  # raters absent → guard short circuit 0


def test_iaa_nominal_run_path_small_limit():
    """--limit 5 + run path: double degradation (single type yt + placeholder yp)"""
    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=5)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == 5


# ---------- iaa_ordinal: run path NaN tax ----------


def test_iaa_ordinal_run_path_no_nan():
    """run path: pred all 0 (placeholder int) → constant input → pearson/spearman/kendall
    Originally all NaN. NaN is not legal JSON (`json.dumps(float('nan'))` writes 'NaN' to any non-Python parser
    will be rejected) and must be tightened to 0."""
    task = IaaOrdinal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm)
    _assert_aggregated_is_finite_json(r.aggregated)
    # All 12 first-class scalar should be present
    expected = {
        "accuracy", "cohens_kappa", "weighted_kappa_linear", "weighted_kappa_quadratic",
        "pearson_r", "spearman_rho", "kendall_tau", "lins_ccc",
        "fleiss_kappa", "krippendorff_alpha_ordinal", "krippendorff_alpha_interval", "icc_1_1",
    }
    assert expected.issubset(set(r.aggregated))
    for k in ("pearson_r", "spearman_rho", "kendall_tau"):
        assert r.aggregated[k] == 0.0, f"{k} should be 0 (constant input), got {r.aggregated[k]}"


def test_iaa_ordinal_run_path_small_limit_no_nan():
    """--limit 3 + run: more degenerate (small N + constant pred), still must have no NaNs."""
    task = IaaOrdinal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=3)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- Cross-task: aggregated serialized into legal JSON ----------


@pytest.mark.parametrize(
    "task_factory,preds_name",
    [
        (IaaNominal, "constant_majority.jsonl"),
        (IaaNominal, "perfect.jsonl"),
        (IaaOrdinal, "perfect.jsonl"),
        (IaaOrdinal, "off_by_one.jsonl"),
    ],
)
def test_iaa_score_aggregated_is_strict_json(task_factory, preds_name: str):
    """Phase 4 path C philosophy: aggregated is always transferred across processes/runs using JSON;
    `allow_nan=False` must be able to round-trip - here covering all 4 stub healthy paths without degradation."""
    task = task_factory()
    family = "iaa_nominal" if task_factory is IaaNominal else "iaa_ordinal"
    preds = Path(__file__).resolve().parent.parent / "data" / family / "predictions" / preds_name
    r = evaluate_score(task, preds)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- Boundary limit ----------


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_iaa_nominal_score_extreme_small_limits(limit: int):
    """`--limit 0/1/2` is a common entry point for audit/debugging; it cannot raise or emit NaN.
    `--limit 0` triggers an empty sample_results path (`if not srs: return 0.0` is gated by everyone)."""
    task = IaaNominal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_nominal" / "predictions" / "perfect.jsonl"
    )
    r = evaluate_score(task, preds, limit=limit)
    _assert_aggregated_is_finite_json(r.aggregated)
    assert r.n == limit


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_iaa_ordinal_score_extreme_small_limits(limit: int):
    """Same as iaa_nominal: minimum limit does not raise / does not NaN."""
    task = IaaOrdinal()
    preds = (
        Path(__file__).resolve().parent.parent
        / "data" / "iaa_ordinal" / "predictions" / "perfect.jsonl"
    )
    r = evaluate_score(task, preds, limit=limit)
    _assert_aggregated_is_finite_json(r.aggregated)


# ---------- Storage layer strict-JSON (phase 8 §8.R4 follow-up) ----------


def test_storage_save_rejects_nan_aggregated(tmp_path: Path):
    """storage.save must fail-loud to reject NaN writes to the disk - otherwise the downstream jq/DB/dashboard will be broken.

    The NaN bag of the task itself (iaa_nominal/iaa_ordinal `_nan_to_zero`) is the first line of defense;
    This test locks the second point: even if any future task is missed, the storage layer will generate a ValueError during write.
    Aligned with the Phase 4 path C "Cross-process and cross-run JSON transfer" contract."""
    from evals.api import EvalResult, SampleResult
    from evals.storage import save

    nan_result = EvalResult(
        task="synthetic",
        model="test",
        mode="score",
        run_id="test-storage-rejects-nan",
        created_at="2026-05-07T00:00:00",
        n=1,
        elapsed_ms=1.0,
        num_fewshot=0,
        aggregated={"good": 1.0, "evil": float("nan")},
        per_sample=(SampleResult(doc_id="d1", prediction="", target="", metrics={"acc": 0.0}),),
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        save(nan_result, runs_dir=tmp_path)


def test_storage_save_rejects_inf_aggregated(tmp_path: Path):
    """Same as NaN: Inf is also illegal JSON, storage must fail-loud."""
    from evals.api import EvalResult, SampleResult
    from evals.storage import save

    inf_result = EvalResult(
        task="synthetic",
        model="test",
        mode="score",
        run_id="test-storage-rejects-inf",
        created_at="2026-05-07T00:00:00",
        n=1,
        elapsed_ms=1.0,
        num_fewshot=0,
        aggregated={"good": 1.0, "evil": float("inf")},
        per_sample=(SampleResult(doc_id="d1", prediction="", target="", metrics={"acc": 0.0}),),
    )
    with pytest.raises(ValueError, match="JSON compliant"):
        save(inf_result, runs_dir=tmp_path)


def test_storage_iaa_run_roundtrip_strict_json(tmp_path: Path):
    """End-to-end: iaa_nominal run path → save → all three write files must be strict-JSON readable
    (`parse_constant` raises ValueError emulating jq/browser/database parser)."""
    from evals.storage import save

    task = IaaNominal()
    lm = _UnusedLM()
    r = evaluate_run(task, lm, limit=5)
    run_dir = save(r, runs_dir=tmp_path)

    def _strict_load(path: Path):
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f"non-JSON literal {x!r} in {path}")
            ),
        )

    _strict_load(run_dir / "result.json")
    for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(
                line,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"non-JSON literal {x!r} in samples.jsonl")
                ),
            )
    for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(
                line,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"non-JSON literal {x!r} in index.jsonl")
                ),
            )


# ============================================================================
# Audit follow-up wave: phase 8 P0/P1/P2 fix regression lock
# ============================================================================
#
# P0: iaa_ordinal parsing failure fallback old implementation uses `pred_int=0`, 0 is not in LIKERT_LABELS
# [1..5] → sklearn `cohen_kappa_score(..., labels=[1..5])` silently discards these samples,
# In the "mixed illegal prediction" scenario cohens_kappa / weighted_kappa_* are silently beautified.
# Modification: process_results flag `_pred_invalid` flag + aggregation cut valid subset.
#
# P1a: yp on the iaa_nominal run path contains OOV (`""` / case / noise), sklearn triggers internally
# `UserWarning: y_pred contains classes not in y_true` × N. Fix: valid subset slicing,
# sklearn sees yp strictly ⊆ LABELS.
#
# P1b: iaa_ordinal correlation metric (pearson/spearman/kendall) scipy triggers on constant input
# `ConstantInputWarning`. Fix: add `_is_constant` short circuit to each metric entry.
#
# P2: iaa_nominal `_balanced_accuracy` / `_mcc` does not accept labels=, sklearn under single-class slicing
# `_check_targets` triggers `UserWarning: A single label was found`; in the same source degradation path
# sklearn `cohen_kappa_score` / statsmodels `fleiss_kappa` Internal division by 0 triggers
# `RuntimeWarning: invalid value encountered in scalar divide`. Fix: catch_warnings
# Partially muted; the outer layer `_nan_to_zero` still contains values.

# ---------- P0: iaa_ordinal mixed illegally. The metric value is correct in the prediction scenario ----------


def _ordinal_synth(yt: list[int], yp: list[int | None]) -> list[SampleResult]:
    """Synthesize ordinal SampleResult list; pred=None means "LM output non-integer is marked as invalid"."""
    from evals.api import SampleResult as _SR

    srs: list[_SR] = []
    for i, (t, p) in enumerate(zip(yt, yp)):
        invalid = p is None
        srs.append(
            _SR(
                doc_id=f"o{i}",
                prediction="" if invalid else str(p),
                target=str(t),
                metrics={"acc": 0.0 if invalid else float(p == t)},
                artifacts={
                    "raters": [],
                    "_pred_int": p,
                    "_target_int": t,
                    "_pred_invalid": invalid,
                },
            )
        )
    return srs


def test_ordinal_p0_mixed_invalid_kappa_not_silently_inflated():
    """P0 core regression: yt=[1,2,3,4,5] + 2 illegal items (None) + 3 valid items, all correct.
    Old implementation cohens_kappa=1.0 / weighted_quad=1.0 (silently discarded by sklearn as invalid, leaving [1,3,5]
    Self-matching false perfect score); after repair, cohens_kappa should come from the true value of valid subset (3/3=perfect → 1.0
    But the semantics are clear: what is measured is the valid subset, not the entire test)."""
    from evals.tasks.iaa_ordinal import IaaOrdinal

    task = IaaOrdinal()
    agg = task.aggregation()
    yt = [1, 2, 3, 4, 5]
    yp = [1, None, 3, None, 5]  # 2 LM parsing failed
    srs = _ordinal_synth(yt, yp)

    # accuracy takes all samples (including invalid → 0 contribution): 3/5 = 0.6
    assert agg["accuracy"](srs) == pytest.approx(0.6)
    # The kappa series is 1.0 on the valid subset (3 all pairs); this is "explicitly keeping invalid out of kappa"
    # Correct semantics - completely different from the old implementation "sklearn silently discards → values ​​look the same but origin is unknown";
    # The new data contract _pred_invalid allows consumers to see which samples were excluded from sample.artifacts.
    assert agg["cohens_kappa"](srs) == pytest.approx(1.0)
    # Key diagnostic fields are implemented at the sample layer, and downstream drill-down can identify invalid samples.
    invalid_count = sum(1 for s in srs if s.artifacts.get("_pred_invalid"))
    assert invalid_count == 2


def test_ordinal_p0_mixed_invalid_with_disagreement():
    """P0 strict regression: when there is a real disagreement within the valid subset, kappa < 1.0, compared with the "false 1.0" of the old implementation
    Quantitative separation. yt=[1..5] + 2 invalid + 3 valid 1 is misplaced (yp=[1, None, 4, None, 5])."""
    from evals.tasks.iaa_ordinal import IaaOrdinal

    task = IaaOrdinal()
    agg = task.aggregation()
    yt = [1, 2, 3, 4, 5]
    yp = [1, None, 4, None, 5]  # valid subset = (1,1)(3,4)(5,5), 2/3 perfect
    srs = _ordinal_synth(yt, yp)
    # Old implementation: sklearn looks at yp=[1, 4, 5] vs yt=[1, 3, 5], labels=[1..5], but valid subset
    # There is misalignment in the center truth, weighted_kappa_quadratic should be < 1.0
    wkq = agg["weighted_kappa_quadratic"](srs)
    assert wkq < 1.0, f"weighted_kappa_quadratic={wkq} 应反映 valid subset 的真实错位"


def test_ordinal_p0_all_invalid_returns_zero():
    """All invalid → valid subset empty → kappa series short circuit returns 0 (instead of NaN / sklearn raise)."""
    from evals.tasks.iaa_ordinal import IaaOrdinal

    task = IaaOrdinal()
    agg = task.aggregation()
    yt = [1, 2, 3, 4, 5]
    yp = [None] * 5
    srs = _ordinal_synth(yt, yp)
    for k in [
        "cohens_kappa", "weighted_kappa_linear", "weighted_kappa_quadratic",
        "pearson_r", "spearman_rho", "kendall_tau", "lins_ccc",
    ]:
        assert agg[k](srs) == 0.0, f"{k} 全 invalid 应返 0"
    # accuracy can still be calculated (all invalid → 0)
    assert agg["accuracy"](srs) == 0.0


def test_ordinal_p0_real_predictions_unchanged():
    """P0 repair does not destroy the README teaching narrative: the 4 legal stub values ​​are consistent with the existing score test."""
    task = IaaOrdinal()
    PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "iaa_ordinal" / "predictions"
    r = evaluate_score(task, PRED_DIR / "off_by_one.jsonl")
    # Inherit the core lock of test_iaa_ordinal_score.py
    assert r.aggregated["accuracy"] == 0.0
    assert r.aggregated["cohens_kappa"] == pytest.approx(-0.25, abs=1e-9)
    assert r.aggregated["weighted_kappa_quadratic"] == pytest.approx(0.7058823529411764, abs=1e-9)


# ---------- P1a: iaa_nominal run path no longer triggers sklearn OOV warnings ----------


def _capture_warnings(fn):
    """Runs fn() and returns a list of triggered warnings; normalized by message text for ease of assertion."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        fn()
        return [(type(w.message).__name__, str(w.message)) for w in caught]


def test_nominal_p1a_run_path_no_oov_warning():
    """P1a regression: yp full `""` on iaa_nominal run path no longer allows sklearn emit
    `UserWarning: y_pred contains classes not in y_true` (aggregation takes valid subset,
    yp ⊆ LABELS as seen by sklearn)."""
    task = IaaNominal()
    lm = _UnusedLM()
    msgs = _capture_warnings(lambda: evaluate_run(task, lm))
    oov = [m for cat, m in msgs if "y_pred contains classes not in y_true" in m]
    assert oov == [], f"应无 OOV warning, 但仍触发 {len(oov)} 次: {oov[:2]}"


def test_nominal_p1a_run_path_no_runtimewarning_from_sklearn_kappa():
    """P1a homology regression: cohens_kappa / fleiss_kappa no longer let sklearn /
    statsmodels emit `RuntimeWarning: invalid value encountered in scalar divide`
    (Includes catch_warnings; the outer layer `_nan_to_zero` still carries the value)."""
    task = IaaNominal()
    lm = _UnusedLM()
    msgs = _capture_warnings(lambda: evaluate_run(task, lm))
    rtw = [m for cat, m in msgs if cat == "RuntimeWarning" and "invalid value" in m]
    assert rtw == [], f"应无 sklearn/statsmodels RuntimeWarning, 但触发 {len(rtw)}: {rtw[:2]}"


# ---------- P1b: iaa_ordinal constant input no longer triggers ConstantInputWarning ----------


def test_ordinal_p1b_constant_input_no_warning():
    """P1b regression: iaa_ordinal run path (yp all invalid → valid subset empty → short circuit returns 0)/
    In scenarios such as iaa_ordinal score path with extremely small limit, scipy no longer emit ConstantInputWarning."""
    task = IaaOrdinal()
    lm = _UnusedLM()
    msgs = _capture_warnings(lambda: evaluate_run(task, lm))
    cw = [m for cat, m in msgs if cat == "ConstantInputWarning"]
    assert cw == [], f"应无 ConstantInputWarning, 但触发 {len(cw)}: {cw[:2]}"


# ---------- P2: iaa_nominal single-class slice no longer triggers sklearn `single label` warning ----------


def test_nominal_p2_single_class_no_balanced_accuracy_warning():
    """P2 regression: limit=5 perfect (full ham single-class slice) no longer allows sklearn `_check_targets`
    emit `UserWarning: A single label was found in 'y_true' and 'y_pred'`."""
    task = IaaNominal()
    PRED_DIR = (
        Path(__file__).resolve().parent.parent / "data" / "iaa_nominal" / "predictions"
    )
    msgs = _capture_warnings(
        lambda: evaluate_score(task, PRED_DIR / "perfect.jsonl", limit=5)
    )
    single_label = [m for cat, m in msgs if "A single label was found" in m]
    assert single_label == [], (
        f"应无 single-label UserWarning, 但触发 {len(single_label)}: {single_label[:2]}"
    )
