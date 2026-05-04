"""Phase 4 Runner / Task hook 向后兼容 parity：3 hook + dict[str, dict] 改造后老 task 字节级不变.

聚焦点不是"新 hook 自己工作"（有专门的 RAG task 测试覆盖），而是：
  - `Task.load_prediction` default 实现走 score 路径，结果与旧 `_load_predictions[id]`
    + `Response(text=preds[id])` 字节级一致
  - `Task.process_docs` default identity 在 run 路径不改 docs 顺序 / 内容
  - `output_type='none'` 分支在 sentiment / mt 等老 task 上不被触发（保留默认 generate_until）

只在 sentiment + mt 两个 phase 1/2 task 上做 score+run 双路径回归——qa_open 已被
test_qa_open_score / test_qa_open_run 覆盖.
"""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.mt import MT
from evals.tasks.sentiment_clf import SentimentClf

PRED_SENTIMENT = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"
PRED_MT = Path(__file__).resolve().parent.parent / "data" / "mt" / "predictions"


def test_sentiment_score_perfect_unchanged_after_load_prediction_default():
    """sentiment perfect.jsonl 走默认 load_prediction → 仍 100% accuracy（字节相同）."""
    task = SentimentClf()
    r = evaluate_score(task, PRED_SENTIMENT / "perfect.jsonl")
    assert r.aggregated == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_mt_score_perfect_unchanged_after_load_prediction_default():
    """mt perfect.jsonl 走默认 load_prediction → exact_match=1.0."""
    task = MT()
    r = evaluate_score(task, PRED_MT / "perfect.jsonl")
    assert r.aggregated["exact_match"] == 1.0


def test_sentiment_run_gold_unchanged_after_process_docs_default():
    """sentiment run mock:gold 走默认 process_docs identity → 仍 100% accuracy."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert r.aggregated == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_old_task_default_output_type_is_generate_until():
    """老 task 的 output_type 仍是 'generate_until'（不会被新 'none' literal 误命中）."""
    assert SentimentClf.output_type == "generate_until"
    assert MT.output_type == "generate_until"


def test_score_run_parity_after_phase4_hooks():
    """phase 4 改造后 sentiment 上 score / run 仍字节级 parity（架构定海神针）."""
    task = SentimentClf()
    docs = list(task.docs())
    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, PRED_SENTIMENT / "perfect.jsonl")
    assert r_run.aggregated == r_score.aggregated
    assert r_run.n == r_score.n
    a_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_score.per_sample]
    assert a_pairs == o_pairs
