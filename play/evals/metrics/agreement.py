"""Family 1 second half — IAA (inter-annotator agreement) hand-calculated metric + unique shared helper.

Trigger new as per README guideline #3:
  - Main signal "No library available": 4 statistics are not readily implemented in mainstream Python packages (irrCAC / pingouin /
    audtorch each has special dependencies; the statistical formula itself only requires 5-15 lines of handwriting)
  - Secondary signal "cross-task reuse": `build_rater_matrix` is a `iaa_nominal` / `iaa_ordinal` shared helper

Scope tightening (DECISIONS §8): This module only installs handwritten functions + true shared helpers. library direct tune
(sklearn `cohen_kappa_score` / scipy.stats `pearsonr|spearmanr|kendalltau` /
statsmodels `fleiss_kappa` / krippendorff `alpha`) are all decentralized and called within task aggregation,
It is completely consistent with the way sentiment_clf directly tunes sklearn / mt directly tunes sacrebleu - to prevent this module from becoming an import transfer station.

Why not wrap the library like metrics/retrieval.py?
  - ranx wrap of retrieval.py is "protocol switch (closure factory) + input construction is non-trivial
    (list[SampleResult] → _build_qrels_run helper of Qrels/Run dict) + 5 indicators share "three simultaneous signals;
  - The statsmodels.fleiss_kappa / krippendorff.alpha interface of this module directly consumes list/matrix,
    Lines 1-3 wrap has no "protocol transfer" value - the semantics of calls in decentralized tasks are clearer.

Industry pedigree:
  - scott_pi (Scott 1955): Same formula as Cohen's κ but Pe uses pooled margin ∑ p̄_c²
  - gwet_ac1 (Gwet 2008): Pe uses inter-class variance (1/(K-1))·∑ q_c(1-q_c) to solve the blindness of κ paradox when marginal unevenness
  - lins_ccc (Lin 1989): concordance correlation coefficient, while penalizing shift + scale
  - icc_1_1 (Shrout & Fleiss 1979): one-way random ANOVA decomposition;
    ICC(2,1) / ICC(3,1) second-order decomposition deferred (DECISIONS §8 explicit registration)"""

from __future__ import annotations

from typing import Hashable, Sequence

from ..api import SampleResult


def scott_pi(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Scott's π (Scott 1955): Same formula as Cohen's κ but using pooled margins for Pe.

    Formula:
      - Po = #(y1[i] == y2[i]) / N
      - p̄_c = (count_c in y1 + count_c in y2) / (2N) pooled marginal proportion
      - Pe = ∑_c p̄_c²
      - π = (Po − Pe) / (1 − Pe)

    Difference from Cohen's κ: Cohen's Pe uses y1 / y2 to multiply their respective marginals; Scott π uses the square of the pooled marginals.
    In practice, the two values ​​are close, but Scott π is more conservative when the two rater margins are significantly different.

    Return value ∈ [-1, 1]; empty input / inconsistent length → ValueError; Pe=1 (single-class perfect consistency) degenerates to 1.0."""
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
        # All of the same class → perfect agreement; consistent with sklearn cohen_kappa_score degradation protocol
        return 1.0
    return (po - pe) / (1.0 - pe)


def gwet_ac1(y1: Sequence[Hashable], y2: Sequence[Hashable]) -> float:
    """Gwet's AC1 (Gwet 2008): Pe uses inter-class variance to solve the κ paradox.

    Formula:
      - Po = same as Cohen/Scott
      - q_c = (count_c in y1 + count_c in y2) / (2N) combined marginal ratio
      - K = #unique categories
      - Pe = (1/(K − 1)) · ∑_c q_c · (1 − q_c)
      - AC1 = (Po − Pe) / (1 − Pe)

    "Solution to κ paradox antidote 1": When the marginal proportion of a certain type is > 90%, Pe of Cohen κ is close to Po and let κ → 0;
    Pe of Gwet AC1 is determined by the class variance, the more extreme the margin, the smaller q_c(1−q_c) → the smaller Pe → AC1 still honestly reflects high consistency.

    The return value ∈ [-1, 1]; when K=1 (single class), the degenerate return is Po (consistent with the Gwet paper convention)."""
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
    """Lin's CCC (Lin 1989): concordance correlation coefficient, with penalty shift + scale.

    Formula: CCC = 2·cov(X, Y) / (σ_X² + σ_Y² + (μ_X − μ_Y)²)

    The difference with Pearson r: Pearson r only looks at the linear relationship (scaling/translation is unchanged), CCC also penalizes the mean difference and variance difference——
    Perfect linear correlation but with shift r=1 and CCC<1, is the standard choice for ordinal/continuous rater consistency.

    Use the population variance (divided by N instead of N−1), consistent with Lin 1989 original.

    Return value ∈ [-1, 1]; empty / inconsistent length → ValueError; degenerate to 1.0 when both sides are constant and equal."""
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

    Input: N×K matrix (N subjects × K raters); it is required that N≥2, K≥2 and the number of raters of each subject is equal.
    Formula:
      - μ_i = mean(row_i); μ = grand mean
      - BMS (between mean square) = K · ∑(μ_i − μ)² / (N − 1)
      - WMS (within mean square) = ∑∑(x_ij − μ_i)² / (N · (K − 1))
      - ICC(1,1) = (BMS − WMS) / (BMS + (K − 1) · WMS)

    Model assumption: one-way random, all raters are regarded as randomly drawn from the population of raters - the measure is
    The reliability of a single rating by a single rater.

    ICC(2,1) / ICC(3,1) deferred (DECISIONS §8): requires additional between-rater sum of
    The second-order decomposition of squares; the workshop volume may not give a stable value, and phase 8.5+ alone becomes ADR."""
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
    """Unified matrix entry for N samples × K raters (shared helper for fleiss/krippendorff/icc).

    Data contract:
      - Each SampleResult is loaded with `list[Hashable]` in `artifacts["raters"]`, N samples
        The lengths of raters must be consistent (K)
      - `include_gold=True` (default): Add SampleResult.target as an extra column → matrix width K+1
      - `include_gold=False`: only raters columns → matrix width K

    Returns an N×(K | K+1) two-dimensional list; the caller can further convert statsmodels matrix /
    krippendorff reliability_data/numpy array.

    Missing protection: artifacts missing raters → ValueError (lm-eval philosophy: contract violation fail-loud)."""
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
