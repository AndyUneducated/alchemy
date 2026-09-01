"""rag_retrieval task score path e2e: 4 copies of stub predictions whose respective indicators are within the expected interval.

The numerical value is also the README teaching narrative - test green = the document does not lie.

  | prediction | recall@5 | mrr | ndcg@5 | meaning |
  |---|---|---|---|---|
  | perfect | 1.0 | 1.0 | 1.0 | upper bound sanity |
  | good_rerank | 1.0 | ~0.5 | mid | recall full / rank inaccurate |
  | weak | <1.0 | low | low | weak baseline |
  | garbage | 0.0 | 0.0 | 0.0 | lower bound sanity |

According to plan §6: Each new task relocks the runner invariant (n_matches / output_type='none' does not adjust LM)."""

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


# ---------- Upper and lower bounds sanity --------------------------------------------------

def test_perfect_all_metrics_one():
    """The gold of all queries is in rank 1 (multiple golds are in rank 1+2) → all members are 1.0."""
    agg = _score("perfect")
    assert agg["recall@5"] == 1.0
    assert agg["precision@5"] > 0.0  # precision@5=0.2 when gold = 1, higher when there are more gold
    assert agg["mrr"] == 1.0
    assert agg["ndcg@5"] == 1.0
    assert agg["map@5"] == 1.0


def test_garbage_all_metrics_zero():
    """retrieved_ids all do not exist doc → all members 0.0."""
    agg = _score("garbage")
    assert agg["recall@5"] == 0.0
    assert agg["precision@5"] == 0.0
    assert agg["mrr"] == 0.0
    assert agg["ndcg@5"] == 0.0
    assert agg["map@5"] == 0.0


# ---------- Core Narrative ---------------------------------------------------------------

def test_good_rerank_full_recall_mid_mrr():
    """**Core narrative**: gold is all in top 5 → recall=1.0, but all retreated to rank 2 → mrr~0.5 (rerank rescue scene)."""
    agg = _score("good_rerank")
    assert agg["recall@5"] == 1.0
    # gold is mainly in rank 2, mrr should be between 0.4 - 0.6
    assert 0.4 <= agg["mrr"] <= 0.7
    # ndcg should also be between perfect and weak
    assert agg["ndcg@5"] < 1.0
    assert agg["mrr"] < agg["recall@5"]


def test_weak_lower_than_good_rerank():
    """weak is strictly weaker than good_rerank on mrr / ndcg."""
    weak = _score("weak")
    good = _score("good_rerank")
    assert weak["mrr"] < good["mrr"]
    assert weak["ndcg@5"] < good["ndcg@5"]


def test_metric_ordering_perfect_gt_good_gt_weak_gt_garbage():
    """The 4 predictions show a strict decrease in mrr - the strongest evidence of metric resolution."""
    perfect = _score("perfect")
    good = _score("good_rerank")
    weak = _score("weak")
    garbage = _score("garbage")
    assert perfect["mrr"] > good["mrr"] > weak["mrr"] > garbage["mrr"] - 1e-9
    assert perfect["ndcg@5"] >= good["ndcg@5"] >= weak["ndcg@5"] > garbage["ndcg@5"] - 1e-9


# ----------Framework invariants--------------------------------------------------------

def test_n_matches_gold():
    """n == the number of rows in the data set (to prevent new task codepath from returning in advance / leaking samples)."""
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 8


def test_score_missing_pred_raises(tmp_path):
    """Missing doc_id strict KeyError (same contract as sentiment / mt, new task re-locked)."""
    task = RagRetrieval()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"rNONE","retrieved_ids":["a.txt"]}\n', encoding="utf-8"
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_artifacts_carry_pred_and_gold_ids():
    """per_sample.artifacts required pred_ids / gold_ids (aggregation contract for pulling data)."""
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert "pred_ids" in s.artifacts
        assert "gold_ids" in s.artifacts
        assert len(s.artifacts["pred_ids"]) > 0
        assert len(s.artifacts["gold_ids"]) > 0


def test_metrics_empty_for_rag_retrieval():
    """rag_retrieval itself does not write per-sample scalars; wave 3 (DECISIONS §7.2) undoes cross-cutting
    After safety AOP, sample.metrics should be an empty dict (rag_retrieval is a retrieval-only task,
    All signals are at artifacts.pred_ids/gold_ids)."""
    task = RagRetrieval()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert s.metrics == {}
