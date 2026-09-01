"""Phase 8 vertical slice: Family 1 second half IAA ordinal task — ordinal-aware vs nominal teaching.

25 1-5 likert ratings (uniform 5×5) + 4 stub predictions + 3 raters/sample performances
"ordinal-aware metric rescue nominal kappa blindness" two-way narrative:

  | prediction | accuracy | cohens_kappa | weighted_quad | pearson | lins_ccc | story |
  |---|---|---|---|---|---|---|
  | perfect | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | upper bound sanity |
  | off_by_one | **0.00** | **-0.25** | **0.71** | **0.83**| **0.71** | **Core narrative**: partial 1 → exact / nominal total blindness; ordinal-aware (weighted κ + corr + ccc) rescue |
  | random | ~0.20 | ~0 | ~0 | ~0 | ~0 | lower bound sanity |
  | garbage | ~0.20 | 0 (paradox) | -1.00 | -1.00 | -1.00 | Extreme inverse: ordinal-aware directly captures the perfect inverse; nominal cohen is still lost (paradox reprint) |

Design points (DECISIONS §8):
  - **output_type='none'**: Same type as iaa_nominal / rag_retrieval, runner jumps to LM call;
    The score main path welds all teaching narratives, the run path completes the teaching deferred (DECISIONS §8 explicit concession)
  - **target as int**: gold/prediction JSONL stores the string `"4"`, and converts it to `int()` in process_results
  - **Library direct adjustment and decentralization + hand calculation lins_ccc**: sklearn cohen_kappa_score (with weights=...) +
    scipy.stats {pearsonr, spearmanr, kendalltau} + statsmodels.fleiss_kappa +
    krippendorff.alpha (ordinal/interval) import in all tasks; lins_ccc goes
    metrics/agreement.py hand calculation (no library available + simple formula)
  - **12 stat aggregation**: exact (1) + agreement nominal/ordinal (3) + corr (3)
    + ccc (1) + multi-rater (4 = fleiss + krip×2 level + icc11)"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

import krippendorff
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from ..api import Doc, Response, SampleResult
from ..metrics.agreement import build_rater_matrix, icc_1_1, lins_ccc
from ..registry import register_task
from .base import Task

LIKERT_LABELS = (1, 2, 3, 4, 5)  # Integer ordinal label

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iaa_ordinal" / "gold.jsonl"


@register_task("iaa_ordinal")
class IaaOrdinal(Task):
    """Family 1 second half IAA oral task: ordinal-aware metric vs nominal κ blindness comparison.

    Data contract: predictions JSONL row = `{id, prediction: str-of-int, raters: list[str-of-int]}`,
    `int()` conversion within process_results."""

    name: ClassVar[str] = "iaa_ordinal"
    output_type: ClassVar[str] = "none"  # phase 4 literal: runner jumps lm.generate_until

    data_path: Path = DATA_PATH

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(id=row["id"], input=row["input"], target=row["target"])

    def doc_to_text(self, doc: Doc) -> str:
        # When output_type='none' it will not be called by the runner; the reserved method satisfies ABC
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: row['raters'] inject metadata; prediction → Response.text."""
        raters = list(row.get("raters", []))
        enriched = replace(doc, metadata={**doc.metadata, "raters": raters})
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        """string → int; illegal prediction flag `_pred_invalid` filtered by aggregation.

        History (audit follow-up): Old implementation fallsback illegal prediction to 0, but 0 is not there
        LIKERT_LABELS=[1..5] within → sklearn `cohen_kappa_score(..., labels=[1..5])`
        Treat these samples as out-of-labels and silently discard them → in the "mixed illegal prediction" scenario
        The kappa series is silently beautified (actually measured when yt=[1,2,3,4,5] yp=[1,0,3,0,5] cohens_kappa=1.0
        instead of real about 0.6). Change to explicit `_pred_invalid` flag + aggregation filtering to avoid
        Double layer silent superposition."""
        pred_str = (response.text or "").strip()
        try:
            pred_int: int | None = int(pred_str)
            pred_invalid = False
        except (TypeError, ValueError):
            pred_int = None
            pred_invalid = True

        target = doc.target or ""
        try:
            target_int = int(target)
        except (TypeError, ValueError):
            target_int = 0

        # raters: score path is already in the form of list[str], converted to int; illegal items fallback 0 (multiple raters
        # The matrix does not participate in the sklearn label silently discarding path, 0 fallback affects only fleiss/krippendorff
        # The statistical smoothness of P0 is much smaller than the metric error of P0)
        raw_raters = list(doc.metadata.get("raters", []))
        rater_ints: list[int] = []
        for r in raw_raters:
            try:
                rater_ints.append(int(r))
            except (TypeError, ValueError):
                rater_ints.append(0)

        acc = float(pred_int is not None and pred_int == target_int)

        return SampleResult(
            doc_id=doc.id,
            # Keep raw pred_str (drill-down to see the real output of LM is more diagnostic than 'None')
            prediction=pred_str if pred_invalid else str(pred_int),
            target=str(target_int),
            metrics={"acc": acc},
            artifacts={
                "raters": rater_ints,
                "_pred_int": pred_int,  # Can be None
                "_target_int": target_int,
                "_pred_invalid": pred_invalid,
            },
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        labels = list(LIKERT_LABELS)

        def _is_invalid(sr: SampleResult) -> bool:
            return bool(sr.artifacts.get("_pred_invalid", False))

        def _y_int_valid(srs: list[SampleResult]) -> tuple[list[int], list[int]]:
            """Only (yt, yp) of valid pred are taken; invalid pred is filtered (audit P0 fixed).

            After filtering, yp ⊆ LIKERT_LABELS is strictly established, sklearn `cohen_kappa_score`
            `labels=[1..5]` no longer silently discards out-of-class samples → the kappa series has correct values in mixed illegal scenarios."""
            valid = [s for s in srs if not _is_invalid(s)]
            return (
                [int(s.artifacts["_target_int"]) for s in valid],
                [int(s.artifacts["_pred_int"]) for s in valid],
            )

        def _nan_to_zero(x: float) -> float:
            # NaN propagates from constant-input correlations and degenerate kappa cases.
            # NaN is non-portable JSON (`json.dumps(NaN)` → invalid for any non-Python
            # parser); collapse to 0.0 sanity per plan contract.
            return 0.0 if x != x else float(x)

        def _is_constant(xs: list[int]) -> bool:
            # scipy.stats correlations emit ConstantInputWarning + return NaN on
            # constant input (audit P1b)；pre-empt to keep stderr clean.
            return len(set(xs)) < 2

        def _accuracy(srs: list[SampleResult]) -> float:
            """All samples (including invalid): invalid pred is naturally not equal to target → 0 contribution.

            Go for `metrics["acc"]` average instead of sklearn `accuracy_score`, avoid sentinel None
            The path into sklearn; consistent with the sample layer acc field definition."""
            if not srs:
                return 0.0
            return float(sum(s.metrics.get("acc", 0.0) for s in srs) / len(srs))

        def _cohens_kappa(srs: list[SampleResult]) -> float:
            """Interpretation of nominal: 1-5 as 5 unordered categories (demonstration of ordinal when nominal is blind)."""
            if not srs:
                return 0.0
            yt, yp = _y_int_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                # Pe=1 degeneracy (small limit single class slices) makes sklearn internally divide by 0 emit RuntimeWarning;
                # `_nan_to_zero` has the value, and the noise is removed here (audit P2 has the same origin).
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels))

        def _weighted_kappa_linear(srs: list[SampleResult]) -> float:
            """ordinal-aware: distance by |i-j| linear discount disagreement."""
            if not srs:
                return 0.0
            yt, yp = _y_int_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels, weights="linear"))

        def _weighted_kappa_quadratic(srs: list[SampleResult]) -> float:
            """ordinal-aware: Distance by (i-j)² secondary discount (off-by-1 is much lighter than off-by-3).

            **Core narrative metric**: ≈ 0.71 in off_by_one scenario, compared to cohens_kappa = -0.25
            It is the most intuitive demonstration of ordinal-aware rescue."""
            if not srs:
                return 0.0
            yt, yp = _y_int_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels, weights="quadratic"))

        def _pearson_r(srs: list[SampleResult]) -> float:
            """Continuous correlation: treat likert as interval scale; off_by_one scenario ≈ 0.83."""
            yt, yp = _y_int_valid(srs)
            if len(yt) < 2 or _is_constant(yt) or _is_constant(yp):
                return 0.0
            r, _p = pearsonr(yt, yp)
            return _nan_to_zero(r)

        def _spearman_rho(srs: list[SampleResult]) -> float:
            """rank correlation: invariant to monotonic transformations; off_by_one scenario ≈ 0.82."""
            yt, yp = _y_int_valid(srs)
            if len(yt) < 2 or _is_constant(yt) or _is_constant(yp):
                return 0.0
            rho, _p = spearmanr(yt, yp)
            return _nan_to_zero(rho)

        def _kendall_tau(srs: list[SampleResult]) -> float:
            """concordance-based rank corr: small samples are more stable; off_by_one scenario ≈ 0.74."""
            yt, yp = _y_int_valid(srs)
            if len(yt) < 2 or _is_constant(yt) or _is_constant(yp):
                return 0.0
            tau, _p = kendalltau(yt, yp)
            return _nan_to_zero(tau)

        def _lins_ccc(srs: list[SampleResult]) -> float:
            """concordance correlation: simultaneous penalty shift + scale; off_by_one scenario ≈ 0.71
            (Synchronized with weighted_kappa_quadratic to reflect ordinal saves)."""
            if not srs:
                return 0.0
            yt, yp = _y_int_valid(srs)
            if not yt:
                return 0.0
            return float(lins_ccc(yt, yp))

        def _fleiss_kappa(srs: list[SampleResult]) -> float:
            """gold + N raters → statsmodels fleiss_kappa (nominal interpretation, multi-rater version Cohen)."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            arr = np.asarray(matrix, dtype=int)
            agg, _cats = aggregate_raters(arr)
            import warnings as _warnings

            with _warnings.catch_warnings():
                # Single class degradation lets statsmodels internal Pe=1 divided by 0 emit RuntimeWarning;
                # `_nan_to_zero` has the value, and the noise is removed here (audit P2 has the same origin).
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(fleiss_kappa(agg))

        def _krippendorff_alpha_ordinal(srs: list[SampleResult]) -> float:
            """ordinal level: rank distance weight, but ignore the interval assumption (5−1 is the same distance as 4−2)."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            rd = np.asarray(matrix, dtype=int).T
            # `<2 unique value` must be determined after dtype=int conversion (target is str("1") raters is
            # int(1), unique false=2 before converting to int, but krippendorff raises after seeing single-domain)
            if len(np.unique(rd)) < 2:
                return 0.0
            return _nan_to_zero(
                krippendorff.alpha(reliability_data=rd, level_of_measurement="ordinal")
            )

        def _krippendorff_alpha_interval(srs: list[SampleResult]) -> float:
            """interval level: (i-j)² distance (same idea as weighted_kappa_quadratic),
            Demonstrates the impact of level selection on alpha - multiple rater origins."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            rd = np.asarray(matrix, dtype=int).T
            if len(np.unique(rd)) < 2:
                return 0.0
            return _nan_to_zero(
                krippendorff.alpha(reliability_data=rd, level_of_measurement="interval")
            )

        def _icc_1_1(srs: list[SampleResult]) -> float:
            """ICC(1,1) one-way random: Assume that raters are randomly selected from the population of raters, and single raters are evaluated individually for reliability."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2 or len(matrix) < 2:
                return 0.0
            return _nan_to_zero(icc_1_1(matrix))

        return {
            "accuracy": _accuracy,
            "cohens_kappa": _cohens_kappa,
            "weighted_kappa_linear": _weighted_kappa_linear,
            "weighted_kappa_quadratic": _weighted_kappa_quadratic,
            "pearson_r": _pearson_r,
            "spearman_rho": _spearman_rho,
            "kendall_tau": _kendall_tau,
            "lins_ccc": _lins_ccc,
            "fleiss_kappa": _fleiss_kappa,
            "krippendorff_alpha_ordinal": _krippendorff_alpha_ordinal,
            "krippendorff_alpha_interval": _krippendorff_alpha_interval,
            "icc_1_1": _icc_1_1,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "accuracy": True,
            "cohens_kappa": True,
            "weighted_kappa_linear": True,
            "weighted_kappa_quadratic": True,
            "pearson_r": True,
            "spearman_rho": True,
            "kendall_tau": True,
            "lins_ccc": True,
            "fleiss_kappa": True,
            "krippendorff_alpha_ordinal": True,
            "krippendorff_alpha_interval": True,
            "icc_1_1": True,
        }
