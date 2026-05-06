"""Phase 8 vertical slice：族 1 后半 IAA ordinal task — ordinal-aware vs nominal 教学.

25 条 1-5 likert ratings (uniform 5×5) + 4 份 stub predictions + 3 raters/sample 演
"ordinal-aware metric 救场 nominal kappa 失明" 双向叙事：

  | 预测       | accuracy | cohens_kappa | weighted_quad | pearson | lins_ccc | 故事 |
  |---|---|---|---|---|---|---|
  | perfect    | 1.00     | 1.00         | 1.00          | 1.00    | 1.00     | 上界 sanity |
  | off_by_one | **0.00** | **-0.25**    | **0.71**      | **0.83**| **0.71** | **核心叙事**：偏 1 → exact / nominal 全失明；ordinal-aware (weighted κ + corr + ccc) 救场 |
  | random     | ~0.20    | ~0           | ~0            | ~0      | ~0       | 下界 sanity |
  | garbage    | ~0.20    | 0 (paradox)  | -1.00         | -1.00   | -1.00    | 极端反向：ordinal-aware 直接抓出 perfect inverse；nominal cohen 仍迷失 (paradox 复刻) |

设计要点（DECISIONS §8）：
  - **output_type='none'**：与 iaa_nominal / rag_retrieval 同型，runner 跳 LM 调用；
    score 主路径焊死全部教学叙事，run 路径完整教学 deferred (DECISIONS §8 显式让步)
  - **target as int**：gold/prediction JSONL 存字符串 `"4"`，process_results 内 `int()` 转
  - **库直调下放 + 手算 lins_ccc**：sklearn cohen_kappa_score (with weights=...) +
    scipy.stats {pearsonr, spearmanr, kendalltau} + statsmodels.fleiss_kappa +
    krippendorff.alpha (ordinal/interval) 全部 task 内 import；lins_ccc 走
    metrics/agreement.py 手算 (无库可用 + 公式简单)
  - **12 stat aggregation**：exact (1) + agreement nominal/ordinal (3) + corr (3)
    + ccc (1) + multi-rater (4 = fleiss + krip×2 level + icc11)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar

import krippendorff
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from ..api import Doc, Response, SampleResult
from ..metrics.agreement import build_rater_matrix, icc_1_1, lins_ccc
from ..registry import register_task
from .base import Task

LIKERT_LABELS = (1, 2, 3, 4, 5)  # 整数序数标签

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iaa_ordinal" / "gold.jsonl"


@register_task("iaa_ordinal")
class IaaOrdinal(Task):
    """族 1 后半 IAA ordinal task：ordinal-aware metric vs nominal κ 失明对比.

    数据契约：predictions JSONL 行 = `{id, prediction: str-of-int, raters: list[str-of-int]}`,
    process_results 内 `int()` 转换。
    """

    name: ClassVar[str] = "iaa_ordinal"
    output_type: ClassVar[str] = "none"  # phase 4 literal: runner 跳 lm.generate_until

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
        # output_type='none' 时不会被 runner 调；保留方法满足 ABC
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score 路径：row['raters'] 注入 metadata；prediction → Response.text."""
        raters = list(row.get("raters", []))
        enriched = replace(doc, metadata={**doc.metadata, "raters": raters})
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        """字符串 → int；非法 prediction 落 0 占位（accuracy 自动 0；correlation 仍可算）."""
        pred_str = (response.text or "").strip()
        try:
            pred_int = int(pred_str)
        except (TypeError, ValueError):
            pred_int = 0
        target = doc.target or ""
        try:
            target_int = int(target)
        except (TypeError, ValueError):
            target_int = 0

        # raters：score 路径已是 list[str] 形式，转 int
        raw_raters = list(doc.metadata.get("raters", []))
        rater_ints: list[int] = []
        for r in raw_raters:
            try:
                rater_ints.append(int(r))
            except (TypeError, ValueError):
                rater_ints.append(0)

        return SampleResult(
            doc_id=doc.id,
            prediction=str(pred_int),
            target=str(target_int),
            metrics={"acc": float(pred_int == target_int)},
            artifacts={"raters": rater_ints, "_pred_int": pred_int, "_target_int": target_int},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], Any]]:
        labels = list(LIKERT_LABELS)

        def _y_int(srs: list[SampleResult]) -> tuple[list[int], list[int]]:
            return (
                [int(s.artifacts.get("_target_int", 0)) for s in srs],
                [int(s.artifacts.get("_pred_int", 0)) for s in srs],
            )

        def _nan_to_zero(x: float) -> float:
            # NaN propagates from constant-input correlations (run path: yp ≡ 0 placeholder)
            # and degenerate kappa cases. NaN is non-portable JSON (`json.dumps(NaN)` →
            # invalid for any non-Python parser); collapse to 0.0 sanity per plan contract.
            return 0.0 if x != x else float(x)

        def _accuracy(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            yt, yp = _y_int(srs)
            return float(accuracy_score(yt, yp))

        def _cohens_kappa(srs: list[SampleResult]) -> float:
            """nominal 解读：1-5 当 5 个无序类（演示 ordinal 当 nominal 看的失明）."""
            if not srs:
                return 0.0
            yt, yp = _y_int(srs)
            return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels))

        def _weighted_kappa_linear(srs: list[SampleResult]) -> float:
            """ordinal-aware: 距离按 |i-j| 线性折扣 disagreement."""
            if not srs:
                return 0.0
            yt, yp = _y_int(srs)
            return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels, weights="linear"))

        def _weighted_kappa_quadratic(srs: list[SampleResult]) -> float:
            """ordinal-aware: 距离按 (i-j)² 二次折扣（off-by-1 远比 off-by-3 轻）.

            **核心叙事 metric**：在 off_by_one 场景下 ≈ 0.71，与 cohens_kappa = -0.25 的对比
            是 ordinal-aware 救场最直观的演示.
            """
            if not srs:
                return 0.0
            yt, yp = _y_int(srs)
            return _nan_to_zero(cohen_kappa_score(yt, yp, labels=labels, weights="quadratic"))

        def _pearson_r(srs: list[SampleResult]) -> float:
            """连续相关：把 likert 当 interval scale；off_by_one 场景 ≈ 0.83."""
            if len(srs) < 2:  # scipy.stats correlations raise on n<2
                return 0.0
            yt, yp = _y_int(srs)
            r, _p = pearsonr(yt, yp)
            return _nan_to_zero(r)

        def _spearman_rho(srs: list[SampleResult]) -> float:
            """rank 相关：对单调变换不变；off_by_one 场景 ≈ 0.82."""
            if len(srs) < 2:
                return 0.0
            yt, yp = _y_int(srs)
            rho, _p = spearmanr(yt, yp)
            return _nan_to_zero(rho)

        def _kendall_tau(srs: list[SampleResult]) -> float:
            """concordance-based rank corr：小样本更稳；off_by_one 场景 ≈ 0.74."""
            if len(srs) < 2:
                return 0.0
            yt, yp = _y_int(srs)
            tau, _p = kendalltau(yt, yp)
            return _nan_to_zero(tau)

        def _lins_ccc(srs: list[SampleResult]) -> float:
            """concordance correlation: 同时罚 shift + scale；off_by_one 场景 ≈ 0.71
            (与 weighted_kappa_quadratic 同步反映 ordinal 救场)."""
            if not srs:
                return 0.0
            yt, yp = _y_int(srs)
            return float(lins_ccc(yt, yp))

        def _fleiss_kappa(srs: list[SampleResult]) -> float:
            """gold + N raters → statsmodels fleiss_kappa（nominal 解读，多 rater 版 Cohen）."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            arr = np.asarray(matrix, dtype=int)
            agg, _cats = aggregate_raters(arr)
            return _nan_to_zero(fleiss_kappa(agg))

        def _krippendorff_alpha_ordinal(srs: list[SampleResult]) -> float:
            """ordinal level：rank 距离权重，但忽略 interval 假设（5−1 与 4−2 同距）."""
            if not srs:
                return 0.0
            matrix = build_rater_matrix(srs, include_gold=True)
            if not matrix or len(matrix[0]) < 2:
                return 0.0
            rd = np.asarray(matrix, dtype=int).T
            # `<2 unique value` 必须在 dtype=int 转换后判（target 是 str("1") raters 是
            # int(1)，转 int 前 unique 假性=2 但 krippendorff 看到 single-domain 后 raise）
            if len(np.unique(rd)) < 2:
                return 0.0
            return _nan_to_zero(
                krippendorff.alpha(reliability_data=rd, level_of_measurement="ordinal")
            )

        def _krippendorff_alpha_interval(srs: list[SampleResult]) -> float:
            """interval level：(i-j)² 距离（与 weighted_kappa_quadratic 同形思路），
            演示 level 选择对 alpha 影响——多 rater 同源."""
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
            """ICC(1,1) one-way random: 假设 raters 从 rater 总体随机抽，单 rater 单评 reliability."""
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
