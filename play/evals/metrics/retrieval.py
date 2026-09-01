"""Family 4 IR metrics: ranx direct wrapper for recall@k / precision@k / mrr / ndcg@k / map@k.

Design points:
  - **ranx direct adjustment**: IR indicator is a mature field with dead mathematical definition (trec_eval has accumulated for decades),
    No need to reinvent the wheel. `ranx` is a numba JIT Python wrapper for trec_eval, with single-ms callback.
  - **Aggregation form**: Return `Callable[[list[SampleResult]], float]`, same as task.aggregation()
    The dict-of-callable protocol isomorphic, and rag_retrieval.aggregation() directly hooks these factories.
  - **Data Contract**: Pull `pred_ids: list[str]` / `gold_ids: list[str]` from `SampleResult.artifacts`
    (non-scalar product bucket introduced in phase 4). Convention: rag_retrieval.process_results requires these two keys.
    Other tasks will not be triggered - the contract coupling points are explicitly marked to avoid implicit dependencies.

Why put IR indicators in metrics/ instead of tasks/rag_retrieval.py:
  - Cross-task reuse: In the future, `rag_qa` will also calculate retrieval-side indicators in process_results
    (recall/precision of contexts), retrieval.py is a natural reuse point
  - Consistent with judge_core / judge_rag style: both "closure factory returns (sample_results) → float"
"""

from __future__ import annotations

from typing import Callable, Sequence

from ranx import Qrels, Run, evaluate as _ranx_evaluate

from ..api import SampleResult


def _build_qrels_run(
    sample_results: Sequence[SampleResult],
) -> tuple[Qrels, Run] | None:
    """Draw (qrels, run) from the SampleResult list and feed ranx; if any required fields are missing → None.

    artifacts contract (rag_retrieval.process_results required):
      - pred_ids: Sequence[str] doc_id list sorted by retrieval rank (after top-k truncation)
      - gold_ids: Sequence[str] The related doc_id collection of this query (the order is irrelevant)

    Returning None allows the aggregation function to gracefully degrade to 0.0 (to prevent empty data sets/test stubs from exploding all metrics)."""
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for sr in sample_results:
        pred_ids = sr.artifacts.get("pred_ids")
        gold_ids = sr.artifacts.get("gold_ids")
        if pred_ids is None or gold_ids is None:
            return None
        if not gold_ids:
            # ranx rejects empty gold - we skip this type of sample (deemed unevaluable)
            continue
        qrels_dict[sr.doc_id] = {gid: 1 for gid in gold_ids}
        # The higher the rank, the higher the score; len(pred_ids) - i gives a monotonically decreasing score (no real score required)
        run_dict[sr.doc_id] = {
            pid: float(len(pred_ids) - i) for i, pid in enumerate(pred_ids)
        }

    if not qrels_dict:
        return None

    return Qrels(qrels_dict), Run(run_dict)


def _make_metric_aggregator(metric_name: str) -> Callable[[list[SampleResult]], float]:
    """Factory: Encapsulate ranx metric names ('recall@5' / 'mrr' / 'ndcg@10' ...) into aggregation closures."""

    def _aggregate(srs: list[SampleResult]) -> float:
        if not srs:
            return 0.0
        built = _build_qrels_run(srs)
        if built is None:
            return 0.0
        qrels, run = built
        return float(_ranx_evaluate(qrels, run, metric_name))

    _aggregate.__name__ = f"aggregate_{metric_name.replace('@', '_at_')}"
    return _aggregate


def recall_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Recall@k: The proportion of gold detected by top-k.

    Classic first-stage retrieval main indicator - tells you "whether the recall is enough", regardless of rank."""
    return _make_metric_aggregator(f"recall@{k}")


def precision_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Precision@k: The proportion of gold in top-k.

    Complementary to recall, it is used for diagnosis of "how much noise is there in top-k"; it should usually be upgraded after rerank."""
    return _make_metric_aggregator(f"precision@{k}")


def mrr() -> Callable[[list[SampleResult]], float]:
    """Mean Reciprocal Rank: average the reciprocal rank of the first gold.

    Suitable for scenarios where "you only care about whether the first article is correct" (grounding on QA usually only takes the top-1 references)."""
    return _make_metric_aggregator("mrr")


def ndcg_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Normalized DCG@k: rank sensitive graded relevance comprehensive score.

    rerank The de facto standard for academic comparison; currently implements binary relevance (0/1), and will expand in the future graded
    You can change gold_ids to dict[str, int] in _build_qrels_run."""
    return _make_metric_aggregator(f"ndcg@{k}")


def map_at_k(k: int = 10) -> Callable[[list[SampleResult]], float]:
    """Mean Average Precision@k: A bird’s-eye view indicator of comprehensive recall + rank.

    Double reward for "the more accurate the previous rank + the more complete the recall"; TREC is the veteran main indicator."""
    return _make_metric_aggregator(f"map@{k}")
