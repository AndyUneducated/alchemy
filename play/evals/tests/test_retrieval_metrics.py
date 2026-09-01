"""metrics/retrieval.py unit layer: 5 IR indicator behavior contracts.

The test goal is not to "rewrite ranx's math tests", but to weld in:
  ① The callable produced by the factory accepts the `list[SampleResult]` protocol (same shape as task.aggregation())
  ② Pull data from SampleResult.artifacts.{pred_ids, gold_ids} (phase 4 contract coupling point)
  ③ Boundary (empty list/artifacts missing fields/gold all empty) goes to 0.0, graceful degradation, no throwing
  ④ Numerical correctness on known toy data (three arrangements of perfect / partial / miss)"""

from __future__ import annotations

from evals.api import SampleResult
from evals.metrics.retrieval import (
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _sr(doc_id: str, pred_ids: list[str], gold_ids: list[str]) -> SampleResult:
    """Constructs a retrieval style SampleResult (pred/target placeholder empty string)."""
    return SampleResult(
        doc_id=doc_id,
        prediction="",
        target="",
        metrics={},
        artifacts={"pred_ids": pred_ids, "gold_ids": gold_ids},
    )


# ---------- Boundaries (4)----------------------------------------------------------------

def test_recall_empty_list_returns_zero():
    """空 sample_results → 0.0（不抛，避免 aggregation 崩溃）."""
    assert recall_at_k(5)([]) == 0.0


def test_recall_missing_artifacts_returns_zero():
    """SampleResult of old task does not have pred_ids/gold_ids → graceful degradation 0.0."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"acc": 1.0})
    assert recall_at_k(5)([sr]) == 0.0


def test_recall_all_gold_empty_returns_zero():
    """gold_ids is all empty (no evaluation samples) → 0.0 (to avoid ranx throwing exceptions)."""
    srs = [_sr("q1", ["d1"], []), _sr("q2", ["d2"], [])]
    assert recall_at_k(5)(srs) == 0.0


def test_metrics_are_aggregation_callable_shape():
    """The callable returned by the factory accepts list[SampleResult] → float (identical to the task.aggregation() protocol)."""
    srs = [_sr("q1", ["d1"], ["d1"])]
    for f in [recall_at_k(5), precision_at_k(5), mrr(), ndcg_at_k(5), map_at_k(5)]:
        v = f(srs)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# ---------- Numerical correctness (5 items) --------------------------------------------------

def test_recall_perfect_recall():
    """gold is all in top-k → recall=1.0."""
    srs = [
        _sr("q1", ["d1", "d2", "d3"], ["d1", "d2"]),
        _sr("q2", ["d4", "d5"], ["d4"]),
    ]
    assert recall_at_k(5)(srs) == 1.0


def test_recall_partial_50pct():
    """q1 recalls 1/2, q2 recalls 1/1 → mean = (0.5 + 1.0) / 2 = 0.75."""
    srs = [
        _sr("q1", ["d1", "dx"], ["d1", "d2"]),  # gold=2, pred hit 1 → recall=0.5
        _sr("q2", ["d4"], ["d4"]),  # gold=1, pred hit 1 → recall=1.0
    ]
    assert abs(recall_at_k(5)(srs) - 0.75) < 1e-9


def test_precision_at_k_top1():
    """precision@1: top1 hit → 1.0; miss → 0.0; mean."""
    srs = [
        _sr("q1", ["d1", "d2"], ["d1"]),  # top1 = d1 = gold → 1.0
        _sr("q2", ["dx", "d4"], ["d4"]),  # top1 = dx ≠ gold → 0.0
    ]
    assert abs(precision_at_k(1)(srs) - 0.5) < 1e-9


def test_mrr_first_relevant_at_rank2():
    """gold 在 rank 2 → reciprocal rank = 1/2 = 0.5."""
    srs = [_sr("q1", ["dx", "d1", "d2"], ["d1"])]
    assert abs(mrr()(srs) - 0.5) < 1e-9


def test_ndcg_at_k_decreases_with_lower_rank():
    """The same gold set, rank 1 vs rank 3 → ndcg@5 strictly decreases.

    Locks the rank-sensitivity of ndcg (unlike recall)."""
    srs_top = [_sr("q1", ["d1", "x", "y"], ["d1"])]
    srs_low = [_sr("q1", ["x", "y", "d1"], ["d1"])]
    assert ndcg_at_k(5)(srs_top) > ndcg_at_k(5)(srs_low)


# ---------- Cross-indicator relationships (2 items) --------------------------------------------------

def test_recall_precision_inverse_at_high_k():
    """Big k → recall goes up / precision goes down (basic IR intuition).

    gold=[d1], pred has 1 gold in the first position + 4 noise:
      recall@5 = 1/1 = 1.0
      precision@5 = 1/5 = 0.2"""
    srs = [_sr("q1", ["d1", "n1", "n2", "n3", "n4"], ["d1"])]
    assert recall_at_k(5)(srs) == 1.0
    assert abs(precision_at_k(5)(srs) - 0.2) < 1e-9


def test_map_perfect_equals_one():
    """perfect rank (all gold on top, no noise) → MAP=1.0."""
    srs = [
        _sr("q1", ["d1", "d2", "d3"], ["d1", "d2", "d3"]),
        _sr("q2", ["d4"], ["d4"]),
    ]
    assert map_at_k(5)(srs) == 1.0
