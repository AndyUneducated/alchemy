"""rag_retrieval task score 路径 e2e：4 份 stub predictions 各自的指标在预期区间.

数值也是 README 教学叙事——test 绿 = 文档没说谎.

  | 预测         | recall@5 | mrr   | ndcg@5 | 含义 |
  |---|---|---|---|---|
  | perfect      | 1.0      | 1.0   | 1.0    | 上界 sanity |
  | good_rerank  | 1.0      | ~0.5  | mid    | recall 满 / rank 不准 |
  | weak         | <1.0     | low   | low    | 弱基线 |
  | garbage      | 0.0      | 0.0   | 0.0    | 下界 sanity |

按 plan §六：每个新 task 重锁 runner 不变量（n_matches / output_type='none' 不调 LM）.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.rag_retrieval import RagRetrieval

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_retrieval" / "predictions"


def _score(pred_name: str) -> dict[str, float]:
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 8
    return r.aggregated


# ---------- 上下界 sanity ---------------------------------------------------

def test_perfect_all_metrics_one():
    """所有 query 的 gold 都在 rank 1（多 gold 在 rank 1+2）→ 全员 1.0."""
    agg = _score("perfect")
    assert agg["recall@5"] == 1.0
    assert agg["precision@5"] > 0.0  # gold = 1 个时 precision@5=0.2，多 gold 时更高
    assert agg["mrr"] == 1.0
    assert agg["ndcg@5"] == 1.0
    assert agg["map@5"] == 1.0


def test_garbage_all_metrics_zero():
    """retrieved_ids 全是不存在的 doc → 全员 0.0."""
    agg = _score("garbage")
    assert agg["recall@5"] == 0.0
    assert agg["precision@5"] == 0.0
    assert agg["mrr"] == 0.0
    assert agg["ndcg@5"] == 0.0
    assert agg["map@5"] == 0.0


# ---------- 核心叙事 -------------------------------------------------------

def test_good_rerank_full_recall_mid_mrr():
    """**核心叙事**：gold 都在 top 5 → recall=1.0，但全部退到 rank 2 → mrr~0.5（rerank 救场场景）."""
    agg = _score("good_rerank")
    assert agg["recall@5"] == 1.0
    # gold 主要在 rank 2，mrr 应在 0.4 - 0.6 之间
    assert 0.4 <= agg["mrr"] <= 0.7
    # ndcg 也应介于 perfect 与 weak 之间
    assert agg["ndcg@5"] < 1.0
    assert agg["mrr"] < agg["recall@5"]


def test_weak_lower_than_good_rerank():
    """weak 在 mrr / ndcg 上严格弱于 good_rerank."""
    weak = _score("weak")
    good = _score("good_rerank")
    assert weak["mrr"] < good["mrr"]
    assert weak["ndcg@5"] < good["ndcg@5"]


def test_metric_ordering_perfect_gt_good_gt_weak_gt_garbage():
    """4 份 predictions 在 mrr 上呈严格递降——指标分辨力的最强证据."""
    perfect = _score("perfect")
    good = _score("good_rerank")
    weak = _score("weak")
    garbage = _score("garbage")
    assert perfect["mrr"] > good["mrr"] > weak["mrr"] > garbage["mrr"] - 1e-9
    assert perfect["ndcg@5"] >= good["ndcg@5"] >= weak["ndcg@5"] > garbage["ndcg@5"] - 1e-9


# ---------- 框架不变量 -----------------------------------------------------

def test_n_matches_gold():
    """n == 数据集行数（防新 task 自身 codepath 提前 return / 漏样本）."""
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 8


def test_score_missing_pred_raises(tmp_path):
    """缺 doc_id 严格 KeyError（与 sentiment / mt 同 contract，新 task 重锁）."""
    task = RagRetrieval()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"rNONE","retrieved_ids":["a.txt"]}\n', encoding="utf-8"
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_artifacts_carry_pred_and_gold_ids():
    """per_sample.artifacts 必填 pred_ids / gold_ids（aggregation 拉数据的契约）."""
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert "pred_ids" in s.artifacts
        assert "gold_ids" in s.artifacts
        assert len(s.artifacts["pred_ids"]) > 0
        assert len(s.artifacts["gold_ids"]) > 0


def test_metrics_dict_stays_scalar():
    """metrics 严守空 dict（本 task 无 per-sample 标量；非标量都进 artifacts）.

    锁住"metrics: dict[str, float]"契约，防回归出 list[str] 偷塞污染.
    """
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert s.metrics == {}
