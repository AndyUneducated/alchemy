"""Phase 8 vertical slice: Family 1 second half IAA nominal task — kappa paradox main stage.

30 highly imbalanced binary spam/ham (27 ham + 3 spam, ~90/10) + 4 stub predictions
+ 3 raters/sample perform kappa paradox three narrative scenes:

  | prediction | accuracy | cohens_kappa | gwet_ac1 | fleiss_kappa | story |
  |---|---|---|---|---|---|
  | perfect | 1.00 | 1.00 | 1.00 | 1.00 | upper bound sanity |
  | constant_majority | **0.90** | **~0.0** | **~0.89**| ~0.0 | **Core paradox**: All-in majority class, acc high but cohens_kappa dead; gwet_ac1 still honestly high (paradox antidote 1) |
  | noisy_diverging | ~0.77 | mid (~0.26) | mid (~0.67)| <0 | Multi rater divergence, fleiss/krippendorff flattened to negative numbers (reverse narrative) |
  | garbage | 0.30 | <0 | <0 | <0 | lower bound sanity |

Design points (DECISIONS §8):
  - **output_type='none'**: Same type as rag_retrieval, runner jumps to LM call; score main path welds all
    Teaching narrative, run path complete teaching deferred (DECISIONS §8 explicit concession from the same source phase 5)
  - **load_prediction injects raters**: score path, row['raters'] into doc.metadata;
    process_results transcribe raters to SampleResult.artifacts
  - **Library direct adjustment and decentralization**: sklearn classification metrics + statsmodels.fleiss_kappa +
    krippendorff.alpha import in all tasks; metrics/agreement.py only installs manual calculation + shared helper
  - **15 stat aggregation**: classification (9) + agreement 2-rater (3) + multi-rater (2)
    + diagnostic confusion matrix (1, `_` prefix is treated as non-first-class indicator)"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

import krippendorff
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from ..api import Doc, Response, SampleResult
from ..metrics.agreement import build_rater_matrix, gwet_ac1, scott_pi
from ..registry import register_task
from .base import Task

LABELS = ("ham", "spam")
POSITIVE_CLASS = "spam"  # imbalanced minority class — report the target class of precision/recall/f1/f_beta

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iaa_nominal" / "gold.jsonl"


@register_task("iaa_nominal")
class IaaNominal(Task):
    """Family 1 second half IAA nominal task: kappa paradox teaching main stage.

    Data contract (same path B+C as rag_retrieval):
      - score path: predictions JSONL row schema = `{id, prediction, raters: list[str]}`
      - run path: runner gives placeholder Response (doc.metadata has no raters) → aggregation gives sanity 0"""

    name: ClassVar[str] = "iaa_nominal"
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
        # When output_type='none', it will not be adjusted by the runner; the reserved method can satisfy ABC
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: row['raters'] injects doc.metadata; prediction goes to Response.text."""
        raters = list(row.get("raters", []))
        enriched = replace(doc, metadata={**doc.metadata, "raters": raters})
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        """pred not in LABELS → label `_pred_invalid` filtered by aggregation.

        History (audit follow-up): The old implementation uses OOV pred (including run path placeholder `""` / LM output
        `'Spam'` / `'Label: spam'` and other noise) are directly fed to sklearn metric → internal trigger
        `UserWarning: y_pred contains classes not in y_true` × N + degenerate paths
        `RuntimeWarning: invalid value encountered in scalar divide`, stderr is polluted.
        Change to `_pred_invalid` flag + aggregation slicing: yp as seen by sklearn strict ⊆ LABELS,
        The warnings disappear, accuracy / multi-rater still takes all samples without affecting the values."""
        pred = (response.text or "").strip()
        target = doc.target or ""
        pred_invalid = pred not in LABELS
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics={"acc": float(pred == target)},
            artifacts={
                "raters": list(doc.metadata.get("raters", [])),
                "_pred_invalid": pred_invalid,
            },
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        labels = list(LABELS)

        def _is_invalid(sr: SampleResult) -> bool:
            return bool(sr.artifacts.get("_pred_invalid", False))

        def _y(srs: list[SampleResult]) -> tuple[list[str], list[str]]:
            """All samples (including invalid); used for accuracy / confusion_matrix
            Wait for the metric of "OOV pred naturally misses without polluting the value"."""
            return [s.target for s in srs], [s.prediction for s in srs]

        def _y_valid(srs: list[SampleResult]) -> tuple[list[str], list[str]]:
            """Only valid pred’s (yt, yp); used internally in sklearn and will be triggered by OOV pred
            metric for `UserWarning: y_pred contains classes not in y_true`
            (audit P1a fix)."""
            valid = [s for s in srs if not _is_invalid(s)]
            return [s.target for s in valid], [s.prediction for s in valid]

        def _pos_label_present(yt: list[str], yp: list[str]) -> bool:
            # History preservation: after valid subset comes in, yp ⊆ labels usually holds, but yt ∪ yp may not contain
            # POSITIVE_CLASS (e.g. limit=5 full ham slice) - still needs to be short-circuited to avoid sklearn raises.
            seen = set(yt) | set(yp)
            return POSITIVE_CLASS in seen and seen.issubset(set(labels))

        def _nan_to_zero(x: float) -> float:
            # NaN propagates from degenerate kappa cases (Pe=1 when --limit 5 leaves
            # only one class). NaN is non-portable JSON (`json.dumps(NaN)` → invalid
            # for any non-Python parser); collapse to 0.0 sanity per plan contract.
            return 0.0 if x != x else float(x)

        def _accuracy(srs: list[SampleResult]) -> float:
            """All samples: OOV pred is naturally not equal to target → 0 contribution, and the sklearn path is not adjusted."""
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(accuracy_score(yt, yp))

        def _balanced_accuracy(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=UserWarning)
                return float(balanced_accuracy_score(yt, yp))

        def _mcc(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=UserWarning)
                return float(matthews_corrcoef(yt, yp))

        def _f1_micro(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            return float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0))

        def _f1_macro(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            return float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))

        def _f_beta_2(srs: list[SampleResult]) -> float:
            """F_β=2: recall weighted 4× precision; "more recall" trade-off for imbalanced tasks."""
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(fbeta_score(yt, yp, beta=2.0, pos_label=POSITIVE_CLASS, zero_division=0))

        def _precision_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(precision_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _recall_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(recall_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _f1_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(f1_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _cohens_kappa(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            import warnings as _warnings

            with _warnings.catch_warnings():
                # Pe=1 degeneracy (single-class slicing) lets sklearn internal `expected = ... / np.sum(sum0)`
                # In addition to 0 emit RuntimeWarning; the outer layer `_nan_to_zero` has the value and is denoised here.
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels))

        def _scott_pi(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            return float(scott_pi(yt, yp))

        def _gwet_ac1(srs: list[SampleResult]) -> float:
            """Antidote 1 to the kappa paradox: still reflect consistency honestly under highly imbalanced margins."""
            if not srs:
                return 0.0
            yt, yp = _y_valid(srs)
            if not yt:
                return 0.0
            return float(gwet_ac1(yt, yp))

        def _fleiss_kappa(srs: list[SampleResult]) -> float:
            """gold + N raters → statsmodels fleiss_kappa; run path raters missing → 0."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            arr = np.asarray(matrix)
            agg, _cats = aggregate_raters(arr)
            import warnings as _warnings

            with _warnings.catch_warnings():
                # Pe=1 degenerate (single class slicing) let statsmodels `(p_mean - p_mean_exp) / (1 - p_mean_exp)`
                # In addition to 0 emit RuntimeWarning; the outer layer `_nan_to_zero` has the value and is denoised here.
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                return _nan_to_zero(fleiss_kappa(agg))

        def _krippendorff_alpha(srs: list[SampleResult]) -> float:
            """gold + N raters → krippendorff alpha (nominal level)."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            # krippendorff alpha is undefined when the domain has <2 distinct values
            # (degenerate single-class subset; e.g. --limit 5 over a ham-only slice).
            if len({v for row in matrix for v in row}) < 2:
                return 0.0
            # krippendorff desired shape (raters, subjects) — N×K transpose of build_rater_matrix
            rd = np.asarray(matrix).T
            return float(krippendorff.alpha(reliability_data=rd, level_of_measurement="nominal"))

        def _confusion(srs: list[SampleResult]) -> dict[str, dict[str, int]]:
            """{gold_label: {pred_label: count}} (diagnostic aid, not a single scalar; `_` prefix avoidance
            higher_is_better sort/cross-run comparison expects scalar by default)."""
            if not srs:
                return {}
            yt, yp = _y(srs)
            cm = confusion_matrix(yt, yp, labels=labels)
            return {
                labels[i]: {labels[j]: int(cm[i][j]) for j in range(len(labels))}
                for i in range(len(labels))
            }

        return {
            "accuracy": _accuracy,
            "balanced_accuracy": _balanced_accuracy,
            "mcc": _mcc,
            "f1_micro": _f1_micro,
            "f1_macro": _f1_macro,
            "f_beta_2": _f_beta_2,
            "precision_spam": _precision_spam,
            "recall_spam": _recall_spam,
            "f1_spam": _f1_spam,
            "cohens_kappa": _cohens_kappa,
            "scott_pi": _scott_pi,
            "gwet_ac1": _gwet_ac1,
            "fleiss_kappa": _fleiss_kappa,
            "krippendorff_alpha": _krippendorff_alpha,
            "_confusion_matrix": _confusion,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "accuracy": True,
            "balanced_accuracy": True,
            "mcc": True,
            "f1_micro": True,
            "f1_macro": True,
            "f_beta_2": True,
            "precision_spam": True,
            "recall_spam": True,
            "f1_spam": True,
            "cohens_kappa": True,
            "scott_pi": True,
            "gwet_ac1": True,
            "fleiss_kappa": True,
            "krippendorff_alpha": True,
        }
