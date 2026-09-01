"""qa_open task run path + dual-mode parity (architecture Dinghaishenzhen).

Phase 0 established LM ABC + parity invariant continuation lock on phase 3 new task——
Green = "score and run share task.process_results / aggregation tail section" There is no secret fork on qa_open."""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.qa_open import QAOpen
from evals.tests.test_qa_open_score import _jaccard_fake

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def test_run_gold_judge_equals_score_perfect_judge():
    """`evaluate_run(qa_open, MockLM(gold), judge=FakeJudge)` ≡ `evaluate_score(qa_open, perfect.jsonl)`.

    Two independent jaccard fake instances (stateless rule) → same prompt input → same score output,
    parity is byte identical at both aggregated and per_sample levels."""
    docs = list(QAOpen().docs())

    task_run = QAOpen(judge_lm=_jaccard_fake())
    r_run = evaluate_run(task_run, MockLM(mode="gold", docs=docs))

    task_score = QAOpen(judge_lm=_jaccard_fake())
    r_score = evaluate_score(task_score, PRED_DIR / "perfect.jsonl")

    # Phase 6 introduces the cross-cutting efficiency subgroup (the object under test calls class, only run hangs); wave 3
    # (DECISIONS §7.2) Undo safety cross-cutting - qa_open no longer has sample.metrics["safety"] placeholders;
    # wave 3 §7.3 Add efficiency.judge subgroup (evaluation tool call class, both paths are linked) - score path
    # The efficiency top level still appears, but only with the judge subgroup (latency_ms / tokens_in without the task part).
    _crosscut = {"efficiency"}
    task_agg = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    task_metrics = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    assert task_agg(r_run.aggregated) == task_agg(r_score.aggregated)
    # run path: efficiency top level contains task part + judge subgroup
    assert "efficiency" in r_run.aggregated
    assert "latency_ms" in r_run.aggregated["efficiency"]
    assert "judge" in r_run.aggregated["efficiency"]
    # Score path: The top level of efficiency only contains the judge subgroup (task LM is not called, and the efficiency of the object under test is not hung)
    assert "efficiency" in r_score.aggregated
    assert "judge" in r_score.aggregated["efficiency"]
    assert "latency_ms" not in r_score.aggregated["efficiency"]
    assert r_run.n == r_score.n

    a_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_score.per_sample]
    assert a_pairs == o_pairs


def test_run_qa_with_self_consistency_wrapper():
    """Judge can still run through self_consistency(n=3) and the value is bounded (plumbing test).

    stateless Jaccard fake → same prompt, three samples are all divided equally → mode=same → no damage to the integration layer."""
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_jaccard_fake(), judge_n_samples=3)
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))

    assert "judge_pointwise" in r.aggregated
    # gold mode → answer == target → Jaccard=1 → 5 points ×10
    assert 1.0 <= r.aggregated["judge_pointwise"] <= 5.0
    assert r.aggregated["judge_pointwise"] == 5.0


def test_run_qa_open_with_fewshot_records_num_fewshot():
    """num_fewshot=2 field persistence (same contract as mt, new task relocks)."""
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_jaccard_fake())
    r = evaluate_run(
        task,
        MockLM(mode="gold", docs=docs),
        num_fewshot=2,
        fewshot_seed=0,
    )
    assert r.num_fewshot == 2
    # gold mode + few-shot does not affect perfect answer
    assert r.aggregated["exact_match"] == 1.0
