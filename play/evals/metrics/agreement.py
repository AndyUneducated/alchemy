"""族 1 后半 — IAA (inter-annotator agreement) 手算 metric + 唯一共享 helper.

按 README 指导原则 #3 触发新建：
  - 主信号「无库可用」：4 个统计量在主流 Python 包中无方便现成实现（irrCAC / pingouin /
    audtorch 都各自带特殊依赖；统计量本身公式仅 5-15 行手写即可）
  - 次信号「跨 task 复用」：`build_rater_matrix` 是 `iaa_nominal` / `iaa_ordinal` 共享 helper

scope 收紧 (DECISIONS §8)：本模块仅装手写函数 + 真共享 helper。库直调
（sklearn `cohen_kappa_score` / scipy.stats `pearsonr|spearmanr|kendalltau` /
statsmodels `fleiss_kappa` / krippendorff `alpha`）全部下放 task aggregation 内调用，
与 sentiment_clf 直调 sklearn / mt 直调 sacrebleu 体例完全一致——避免本模块沦为 import 中转站。

为什么不像 metrics/retrieval.py 那样 wrap 库？
  - retrieval.py 的 ranx wrap 是「协议转接 (closure factory) + 输入构造非平凡
    (list[SampleResult] → Qrels/Run dict 的 _build_qrels_run helper) + 5 指标共用」三联立信号；
  - 本模块的 statsmodels.fleiss_kappa / krippendorff.alpha 接口直接吃 list/matrix，
    1-3 行 wrap 没有「协议转接」价值——下放 task 内调用语义更清晰。

行业血统：
  - scott_pi (Scott 1955)：与 Cohen's κ 同公式但 Pe 用合并边际 ∑ p̄_c²
  - gwet_ac1 (Gwet 2008)：Pe 用类间方差 (1/(K-1))·∑ q_c(1-q_c)，破解 κ paradox 边际不均时的失明
  - lins_ccc (Lin 1989)：concordance correlation coefficient，同时罚 shift + scale
  - icc_1_1 (Shrout & Fleiss 1979)：one-way random ANOVA decomposition；
    ICC(2,1) / ICC(3,1) 二阶 decomposition deferred (DECISIONS §8 显式登记)
"""

from __future__ import annotations

from typing import Hashable, Sequence

from ..api import SampleResult


def scott_pi(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Scott's π (Scott 1955)：与 Cohen's κ 同公式但 Pe 用合并边际.

    公式：
      - Po = #(y1[i] == y2[i]) / N
      - p̄_c = (count_c in y1 + count_c in y2) / (2N)  合并边际比例
      - Pe = ∑_c p̄_c²
      - π = (Po − Pe) / (1 − Pe)

    与 Cohen's κ 区别：Cohen κ 的 Pe 用 y1 / y2 各自边际相乘；Scott π 用合并边际平方。
    实操上两者数值接近，但 Scott π 在两 rater 边际显著不同时更保守。

    返回值 ∈ [-1, 1]；空输入 / 长度不一致 → ValueError；Pe=1 (单类完美一致) 退化返 1.0。
    """
    if len(y1) != len(y2):
        raise ValueError(f"scott_pi: length mismatch {len(y1)} vs {len(y2)}")
    n = len(y1)
    if n == 0:
        raise ValueError("scott_pi: empty input")
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    counts: dict[Hashable, int] = {}
    for a in y1:
        counts[a] = counts.get(a, 0) + 1
    for b in y2:
        counts[b] = counts.get(b, 0) + 1
    total = 2 * n
    pe = sum((c / total) ** 2 for c in counts.values())
    if pe >= 1.0:
        # 全部都是同一类 → 完美一致；与 sklearn cohen_kappa_score 退化协议一致
        return 1.0
    return (po - pe) / (1.0 - pe)


def gwet_ac1(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Gwet's AC1 (Gwet 2008)：Pe 用类间方差，破解 κ paradox.

    公式：
      - Po = 同 Cohen / Scott
      - q_c = (count_c in y1 + count_c in y2) / (2N)  合并边际比例
      - K   = #unique categories
      - Pe = (1/(K − 1)) · ∑_c q_c · (1 − q_c)
      - AC1 = (Po − Pe) / (1 − Pe)

    "解 κ paradox 解药 1"：当某类边际占比 > 90% 时，Cohen κ 的 Pe 接近 Po 让 κ → 0；
    Gwet AC1 的 Pe 由类方差决定，边际越极端 q_c(1−q_c) 越小 → Pe 越小 → AC1 仍诚实地反映高一致性。

    返回值 ∈ [-1, 1]；K=1 (单类) 时退化返 Po（与 Gwet 论文约定一致）。
    """
    if len(y1) != len(y2):
        raise ValueError(f"gwet_ac1: length mismatch {len(y1)} vs {len(y2)}")
    n = len(y1)
    if n == 0:
        raise ValueError("gwet_ac1: empty input")
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    counts: dict[Hashable, int] = {}
    for a in y1:
        counts[a] = counts.get(a, 0) + 1
    for b in y2:
        counts[b] = counts.get(b, 0) + 1
    total = 2 * n
    qs = [c / total for c in counts.values()]
    k = len(qs)
    if k <= 1:
        return po
    pe = sum(q * (1 - q) for q in qs) / (k - 1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def lins_ccc(y1: Sequence[float], y2: Sequence[float]) -> float:
    """Lin's CCC (Lin 1989)：concordance correlation coefficient，同时罚 shift + scale.

    公式：CCC = 2·cov(X, Y) / (σ_X² + σ_Y² + (μ_X − μ_Y)²)

    与 Pearson r 区别：Pearson r 只看线性关系（缩放/平移不变），CCC 还罚均值差和方差差——
    完美线性相关但有 shift 时 r=1 而 CCC<1，是 ordinal/continuous rater 一致性的标准选择。

    用群体方差（除以 N 而非 N−1），与 Lin 1989 原版一致。

    返回值 ∈ [-1, 1]；空 / 长度不一致 → ValueError；双方都常数且相等时退化返 1.0。
    """
    if len(y1) != len(y2):
        raise ValueError(f"lins_ccc: length mismatch {len(y1)} vs {len(y2)}")
    n = len(y1)
    if n == 0:
        raise ValueError("lins_ccc: empty input")
    xs = [float(v) for v in y1]
    ys = [float(v) for v in y2]
    mu_x = sum(xs) / n
    mu_y = sum(ys) / n
    var_x = sum((x - mu_x) ** 2 for x in xs) / n
    var_y = sum((y - mu_y) ** 2 for y in ys) / n
    cov = sum((x - mu_x) * (y - mu_y) for x, y in zip(xs, ys)) / n
    denom = var_x + var_y + (mu_x - mu_y) ** 2
    if denom == 0:
        return 1.0
    return 2.0 * cov / denom


def icc_1_1(rating_matrix: Sequence[Sequence[float]]) -> float:
    """ICC(1,1) one-way random ANOVA decomposition (Shrout & Fleiss 1979).

    输入：N×K 矩阵 (N subjects × K raters)；要求 N≥2、K≥2 且各 subject 的 rater 数相等。
    公式：
      - μ_i = mean(row_i)；μ = grand mean
      - BMS (between mean square) = K · ∑(μ_i − μ)² / (N − 1)
      - WMS (within mean square)  = ∑∑(x_ij − μ_i)² / (N · (K − 1))
      - ICC(1,1) = (BMS − WMS) / (BMS + (K − 1) · WMS)

    模型假设：one-way random，所有 rater 视为从 rater 总体随机抽取——measure 的是
    单 rater 单次评分的 reliability。

    ICC(2,1) / ICC(3,1) deferred (DECISIONS §8)：需要额外 between-rater sum of
    squares 的二阶分解；workshop 体量未必能给出稳定数值，phase 8.5+ 单独成 ADR。
    """
    n = len(rating_matrix)
    if n < 2:
        raise ValueError(f"icc_1_1: need >=2 subjects, got {n}")
    k = len(rating_matrix[0])
    if k < 2:
        raise ValueError(f"icc_1_1: need >=2 raters per subject, got {k}")
    for i, row in enumerate(rating_matrix):
        if len(row) != k:
            raise ValueError(
                f"icc_1_1: row {i} length {len(row)} != {k} (uneven matrix not supported)"
            )
    matrix = [[float(v) for v in row] for row in rating_matrix]
    subject_means = [sum(row) / k for row in matrix]
    grand_mean = sum(subject_means) / n
    bms = k * sum((mu - grand_mean) ** 2 for mu in subject_means) / (n - 1)
    wms = sum(
        (matrix[i][j] - subject_means[i]) ** 2
        for i in range(n)
        for j in range(k)
    ) / (n * (k - 1))
    denom = bms + (k - 1) * wms
    if denom == 0:
        return 1.0
    return (bms - wms) / denom


def build_rater_matrix(
    sample_results: Sequence[SampleResult],
    *,
    include_gold: bool = True,
) -> list[list[Hashable]]:
    """N samples × K raters 的统一矩阵入口（fleiss / krippendorff / icc 的共享 helper）.

    数据契约：
      - 每条 SampleResult 在 `artifacts["raters"]` 装 `list[Hashable]`，N 条 sample
        的 raters 长度必须一致 (K)
      - `include_gold=True` (默认)：把 SampleResult.target 作为额外一列拼前面 → 矩阵宽 K+1
      - `include_gold=False`：仅 raters 列 → 矩阵宽 K

    返回 N×(K | K+1) 二维 list；调用方按需进一步转 statsmodels matrix /
    krippendorff reliability_data / numpy array。

    缺失保护：artifacts 缺 raters → ValueError（lm-eval 哲学：契约违背 fail-loud）.
    """
    if not sample_results:
        return []
    matrix: list[list[Hashable]] = []
    expected_k: int | None = None
    for sr in sample_results:
        raters = sr.artifacts.get("raters")
        if raters is None:
            raise ValueError(
                f"build_rater_matrix: sample {sr.doc_id!r} missing artifacts['raters']; "
                "iaa task must populate raters in process_results"
            )
        if expected_k is None:
            expected_k = len(raters)
        elif len(raters) != expected_k:
            raise ValueError(
                f"build_rater_matrix: sample {sr.doc_id!r} raters length {len(raters)} "
                f"!= expected {expected_k}; uneven rater counts not supported"
            )
        row: list[Hashable] = []
        if include_gold:
            row.append(sr.target)
        row.extend(raters)
        matrix.append(row)
    return matrix
