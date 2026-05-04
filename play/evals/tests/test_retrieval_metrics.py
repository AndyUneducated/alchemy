"""metrics/retrieval.py 单元层：5 个 IR 指标行为契约.

测试目标不是"重写 ranx 的数学测试"，而是焊死：
  ① 工厂生产的 callable 接受 `list[SampleResult]` 协议（与 task.aggregation() 同形）
  ② 从 SampleResult.artifacts.{pred_ids, gold_ids} 拉数据（phase 4 契约耦合点）
  ③ 边界（空列表 / artifacts 缺字段 / gold 全空）走 0.0 优雅降级，不抛
  ④ 已知玩具数据上的数值正确性（perfect / partial / miss 三种排布）
"""

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
    """构造一条 retrieval 风格 SampleResult（pred/target 占位空字符串）."""
    return SampleResult(
        doc_id=doc_id,
        prediction="",
        target="",
        metrics={},
        artifacts={"pred_ids": pred_ids, "gold_ids": gold_ids},
    )


# ---------- 边界（4 条）-----------------------------------------------------

def test_recall_empty_list_returns_zero():
    """空 sample_results → 0.0（不抛，避免 aggregation 崩溃）."""
    assert recall_at_k(5)([]) == 0.0


def test_recall_missing_artifacts_returns_zero():
    """老 task 的 SampleResult 没有 pred_ids/gold_ids → 优雅降级 0.0."""
    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"acc": 1.0})
    assert recall_at_k(5)([sr]) == 0.0


def test_recall_all_gold_empty_returns_zero():
    """gold_ids 全空（无可评样本）→ 0.0（避免 ranx 抛异常）."""
    srs = [_sr("q1", ["d1"], []), _sr("q2", ["d2"], [])]
    assert recall_at_k(5)(srs) == 0.0


def test_metrics_are_aggregation_callable_shape():
    """工厂返回的 callable 接受 list[SampleResult] → float（与 task.aggregation() 协议同形）."""
    srs = [_sr("q1", ["d1"], ["d1"])]
    for f in [recall_at_k(5), precision_at_k(5), mrr(), ndcg_at_k(5), map_at_k(5)]:
        v = f(srs)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# ---------- 数值正确性（5 条）-----------------------------------------------

def test_recall_perfect_recall():
    """gold 全在 top-k → recall=1.0."""
    srs = [
        _sr("q1", ["d1", "d2", "d3"], ["d1", "d2"]),
        _sr("q2", ["d4", "d5"], ["d4"]),
    ]
    assert recall_at_k(5)(srs) == 1.0


def test_recall_partial_50pct():
    """q1 召回 1/2，q2 召回 1/1 → mean = (0.5 + 1.0) / 2 = 0.75."""
    srs = [
        _sr("q1", ["d1", "dx"], ["d1", "d2"]),  # gold=2，pred 命中 1 → recall=0.5
        _sr("q2", ["d4"], ["d4"]),  # gold=1，pred 命中 1 → recall=1.0
    ]
    assert abs(recall_at_k(5)(srs) - 0.75) < 1e-9


def test_precision_at_k_top1():
    """precision@1：top1 命中 → 1.0；未命中 → 0.0；mean."""
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
    """同样的 gold 集合，rank 1 vs rank 3 → ndcg@5 严格下降.

    锁定 ndcg 的 rank-sensitive 性（与 recall 不同）.
    """
    srs_top = [_sr("q1", ["d1", "x", "y"], ["d1"])]
    srs_low = [_sr("q1", ["x", "y", "d1"], ["d1"])]
    assert ndcg_at_k(5)(srs_top) > ndcg_at_k(5)(srs_low)


# ---------- 跨指标关系（2 条）-----------------------------------------------

def test_recall_precision_inverse_at_high_k():
    """大 k → recall 趋升 / precision 趋降（基础 IR 直觉）.

    gold=[d1]，pred 有 1 个 gold 在第一位 + 4 个噪声：
      recall@5 = 1/1 = 1.0
      precision@5 = 1/5 = 0.2
    """
    srs = [_sr("q1", ["d1", "n1", "n2", "n3", "n4"], ["d1"])]
    assert recall_at_k(5)(srs) == 1.0
    assert abs(precision_at_k(5)(srs) - 0.2) < 1e-9


def test_map_perfect_equals_one():
    """perfect rank（所有 gold 都在最前面，无噪声）→ MAP=1.0."""
    srs = [
        _sr("q1", ["d1", "d2", "d3"], ["d1", "d2", "d3"]),
        _sr("q2", ["d4"], ["d4"]),
    ]
    assert map_at_k(5)(srs) == 1.0
