"""Phase 8 vertical slice：族 1 后半 IAA nominal task — kappa paradox 主舞台.

30 条 highly imbalanced binary spam/ham (27 ham + 3 spam, ~90/10) + 4 份 stub predictions
+ 3 raters/sample 演 kappa paradox 三场叙事：

  | 预测              | accuracy | cohens_kappa | gwet_ac1 | fleiss_kappa | 故事 |
  |---|---|---|---|---|---|
  | perfect           | 1.00     | 1.00         | 1.00     | 1.00         | 上界 sanity |
  | constant_majority | **0.90** | **~0.0**     | **~0.89**| ~0.0         | **核心 paradox**：全押多数类，acc 高但 cohens_kappa 失效；gwet_ac1 仍诚实地高 (paradox 解药 1) |
  | noisy_diverging   | ~0.77    | mid (~0.26)  | mid (~0.67)| <0          | 多 rater 分歧，fleiss/krippendorff 拉平到负数 (反向叙事) |
  | garbage           | 0.30     | <0           | <0       | <0           | 下界 sanity |

设计要点（DECISIONS §8）：
  - **output_type='none'**：与 rag_retrieval 同型，runner 跳 LM 调用；score 主路径焊死全部
    教学叙事，run 路径完整教学 deferred (DECISIONS §8 显式让步同源 phase 5)
  - **load_prediction 注入 raters**：score 路径，row['raters'] 进 doc.metadata；
    process_results 把 raters 转写到 SampleResult.artifacts
  - **库直调下放**：sklearn classification metrics + statsmodels.fleiss_kappa +
    krippendorff.alpha 全部 task 内 import；metrics/agreement.py 仅装手算 + 共享 helper
  - **15 stat aggregation**：classification (9) + agreement 2-rater (3) + multi-rater (2)
    + diagnostic confusion matrix (1, `_` 前缀视为非 first-class 指标)
"""

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
POSITIVE_CLASS = "spam"  # imbalanced minority class — 报 precision/recall/f1/f_beta 的目标类

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iaa_nominal" / "gold.jsonl"


@register_task("iaa_nominal")
class IaaNominal(Task):
    """族 1 后半 IAA nominal task：kappa paradox 教学主舞台.

    数据契约（与 rag_retrieval 同 path B+C）：
      - score 路径：predictions JSONL 行 schema = `{id, prediction, raters: list[str]}`
      - run 路径：runner 给占位 Response（doc.metadata 无 raters）→ aggregation 给 sanity 0
    """

    name: ClassVar[str] = "iaa_nominal"
    output_type: ClassVar[str] = "none"  # phase 4 literal：runner 跳 lm.generate_until

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
        # output_type='none' 时不会被 runner 调；保留方法满足 ABC 即可
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：row['raters'] 注入 doc.metadata；prediction 走 Response.text."""
        raters = list(row.get("raters", []))
        enriched = replace(doc, metadata={**doc.metadata, "raters": raters})
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = (response.text or "").strip()
        target = doc.target or ""
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics={"acc": float(pred == target)},
            artifacts={"raters": list(doc.metadata.get("raters", []))},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        labels = list(LABELS)

        def _y(srs: list[SampleResult]) -> tuple[list[str], list[str]]:
            return [s.target for s in srs], [s.prediction for s in srs]

        def _pos_label_present(yt: list[str], yp: list[str]) -> bool:
            # sklearn binary scorers (precision/recall/f1/fbeta with pos_label) raise when
            # pos_label is absent from y_true ∪ y_pred OR when the union is multiclass
            # (e.g. {ham, spam, ""} on the run path where placeholder pred = ""). Both
            # scenarios mean "no signal for the positive class" — short-circuit to 0.0
            # rather than propagating sklearn's internal validation errors.
            seen = set(yt) | set(yp)
            return POSITIVE_CLASS in seen and seen.issubset(set(labels))

        def _nan_to_zero(x: float) -> float:
            # NaN propagates from degenerate kappa / Fleiss cases (e.g. Pe=1 when --limit
            # 5 leaves only one class, or run-path placeholder pred outside `labels`).
            # NaN is non-portable JSON (`json.dumps(NaN)` → invalid for any non-Python
            # parser); collapse to 0.0 sanity per plan contract.
            return 0.0 if x != x else float(x)

        def _accuracy(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(accuracy_score(yt, yp))

        def _balanced_accuracy(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(balanced_accuracy_score(yt, yp))

        def _mcc(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(matthews_corrcoef(yt, yp))

        def _f1_micro(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(f1_score(yt, yp, labels=labels, average="micro", zero_division=0))

        def _f1_macro(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))

        def _f_beta_2(srs: list[SampleResult]) -> float:
            """F_β=2：recall 加权 4× precision；imbalanced 任务的"宁多召回"权衡."""
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(fbeta_score(yt, yp, beta=2.0, pos_label=POSITIVE_CLASS, zero_division=0))

        def _precision_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(precision_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _recall_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(recall_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _f1_spam(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            if not _pos_label_present(yt, yp):
                return 0.0
            return float(f1_score(yt, yp, pos_label=POSITIVE_CLASS, zero_division=0))

        def _cohens_kappa(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels))

        def _scott_pi(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(scott_pi(yt, yp))

        def _gwet_ac1(srs: list[SampleResult]) -> float:
            """kappa paradox 解药 1：高度不均衡边际下仍诚实反映一致性."""
            if not srs:
                return 0.0
            yt, yp = _y(srs)
            return float(gwet_ac1(yt, yp))

        def _fleiss_kappa(srs: list[SampleResult]) -> float:
            """gold + N raters → statsmodels fleiss_kappa；run 路径 raters 缺失 → 0."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            arr = np.asarray(matrix)
            agg, _cats = aggregate_raters(arr)
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
            # krippendorff 期望形状 (raters, subjects) — 对 build_rater_matrix 的 N×K 转置
            rd = np.asarray(matrix).T
            return float(krippendorff.alpha(reliability_data=rd, level_of_measurement="nominal"))

        def _confusion(srs: list[SampleResult]) -> dict[str, dict[str, int]]:
            """{gold_label: {pred_label: count}}（诊断辅助，非单标量；`_` 前缀避开
            higher_is_better 排序 / cross-run 比较默认期待 scalar）."""
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
