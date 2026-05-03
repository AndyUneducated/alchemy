"""run 路径 e2e + **双模式等价性 test**.

test_run_gold_equals_score_perfect 是整个架构的定海神针：
它把"score 和 run 共享 task.process_results / aggregation 尾段"这句话
变成了可执行的 assert。绿 = 两路径在 task 层真正交汇、没有分支偷偷分叉。
"""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.mt import MT
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"
MT_PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "mt" / "predictions"


def test_run_gold_runs_full_accuracy():
    task = SentimentClf()
    docs = list(task.docs())
    lm = MockLM(mode="gold", docs=docs)
    r = evaluate_run(task, lm)
    assert r.mode == "run"
    assert r.model == "mock:gold"
    assert r.n == 30
    assert r.aggregated == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_run_gold_equals_score_perfect():
    """架构承诺：evaluate_run(task, MockLM(gold)) ≡ evaluate_score(task, perfect.jsonl)."""
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, PRED_DIR / "perfect.jsonl")

    assert r_run.aggregated == r_score.aggregated
    assert r_run.n == r_score.n
    # per_sample 的 (doc_id, prediction, target, metrics) 也应该一样
    a_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_noisy_matches_predictions_file():
    """MockLM(noisy, seed=0) 和 predictions/noisy_0.3.jsonl 是同一份（生成时两侧 seed 对齐）."""
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="noisy", docs=docs, noise=0.3, seed=0))
    r_score = evaluate_score(task, PRED_DIR / "noisy_0.3.jsonl")

    assert r_run.aggregated == r_score.aggregated


def test_run_constant_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="constant", docs=docs, label="neutral"))
    r_score = evaluate_score(task, PRED_DIR / "constant_neutral.jsonl")

    assert r_run.aggregated == r_score.aggregated


def test_run_rule_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="rule", docs=docs))
    r_score = evaluate_score(task, PRED_DIR / "keyword_rule.jsonl")

    assert r_run.aggregated == r_score.aggregated


def test_run_mt_gold_equals_score_perfect():
    """族 2 (mt) 上的 parity：mock:gold ≡ score predictions/perfect.jsonl.

    族 1 已有同名测试，但 mt 引入了 6 个生成指标（含 BERTScore）和不同 task schema，
    在新 task 上重新焊一次双模式等价性，避免回归。"""
    task = MT()
    docs = list(task.docs())

    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, MT_PRED_DIR / "perfect.jsonl")

    assert r_run.aggregated == r_score.aggregated
    assert r_run.n == r_score.n
    a_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_mt_with_fewshot_records_num_fewshot():
    """num_fewshot=2 时 EvalResult.num_fewshot 字段被正确记录."""
    task = MT()
    docs = list(task.docs())

    r = evaluate_run(task, MockLM(mode="gold", docs=docs), num_fewshot=2, fewshot_seed=0)
    assert r.num_fewshot == 2
    # gold mode 下答案和 target 一字不差，K-shot 不影响 perfect score
    assert r.aggregated["exact_match"] == 1.0
