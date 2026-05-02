"""run 路径 e2e + **双模式等价性 test**.

test_active_gold_equals_offline_perfect 是整个架构的定海神针：
它把"score 和 run 共享 task.process_results / aggregation 尾段"这句话
变成了可执行的 assert。绿 = 两路径在 task 层真正交汇、没有分支偷偷分叉。
"""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_active, evaluate_offline
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"


def test_active_gold_runs_full_accuracy():
    task = SentimentClf()
    docs = list(task.docs())
    lm = MockLM(mode="gold", docs=docs)
    r = evaluate_active(task, lm)
    assert r.mode == "run"
    assert r.model == "mock:gold"
    assert r.n == 30
    assert r.aggregated == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_active_gold_equals_offline_perfect():
    """架构承诺：evaluate_active(task, MockLM(gold)) ≡ evaluate_offline(task, perfect.jsonl)."""
    task = SentimentClf()
    docs = list(task.docs())

    r_active = evaluate_active(task, MockLM(mode="gold", docs=docs))
    r_offline = evaluate_offline(task, PRED_DIR / "perfect.jsonl")

    assert r_active.aggregated == r_offline.aggregated
    assert r_active.n == r_offline.n
    # per_sample 的 (doc_id, prediction, target, metrics) 也应该一样
    a_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_active.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, s.metrics) for s in r_offline.per_sample]
    assert a_pairs == o_pairs


def test_active_noisy_matches_predictions_file():
    """MockLM(noisy, seed=0) 和 predictions/noisy_0.3.jsonl 是同一份（生成时两侧 seed 对齐）."""
    task = SentimentClf()
    docs = list(task.docs())

    r_active = evaluate_active(task, MockLM(mode="noisy", docs=docs, noise=0.3, seed=0))
    r_offline = evaluate_offline(task, PRED_DIR / "noisy_0.3.jsonl")

    assert r_active.aggregated == r_offline.aggregated


def test_active_constant_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_active = evaluate_active(task, MockLM(mode="constant", docs=docs, label="neutral"))
    r_offline = evaluate_offline(task, PRED_DIR / "constant_neutral.jsonl")

    assert r_active.aggregated == r_offline.aggregated


def test_active_rule_matches_predictions_file():
    task = SentimentClf()
    docs = list(task.docs())

    r_active = evaluate_active(task, MockLM(mode="rule", docs=docs))
    r_offline = evaluate_offline(task, PRED_DIR / "keyword_rule.jsonl")

    assert r_active.aggregated == r_offline.aggregated
