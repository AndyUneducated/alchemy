"""Task ABC default hook behavior lock (3 default hooks + dict[str, dict] row form introduced in Phase 4).

The focus is not on "new hooks working by themselves" (with dedicated RAG task test coverage), but on:
  - `Task.load_prediction` defaults to the score path: the result is the same as "naked row['prediction'] +
    `Response(text=preds[id])`" Byte-level consistency - ensuring tasks that do not require custom row schema
    (sentiment/mt/classification etc.) works with zero override
  - `Task.process_docs` default identity does not change the order/content of docs in the run path
  - `output_type='none'` branch is not triggered on pure generate_until tasks such as sentiment / mt

Only perform score+run dual path regression on the sentiment + mt tasks - qa_open has been
test_qa_open_score / test_qa_open_run override."""

from __future__ import annotations

from pathlib import Path

from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.mt import MT
from evals.tasks.sentiment_clf import SentimentClf

PRED_SENTIMENT = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"
PRED_MT = Path(__file__).resolve().parent.parent / "data" / "mt" / "predictions"


def test_sentiment_score_perfect_unchanged_after_load_prediction_default():
    """sentiment perfect.jsonl uses the default load_prediction → still 100% accurate (the same bytes)."""
    task = SentimentClf()
    r = evaluate_score(task, PRED_SENTIMENT / "perfect.jsonl")
    task_agg = {k: v for k, v in r.aggregated.items() if k not in {"efficiency", "safety"}}
    assert task_agg == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_mt_score_perfect_unchanged_after_load_prediction_default():
    """mt perfect.jsonl uses the default load_prediction → exact_match=1.0."""
    task = MT()
    r = evaluate_score(task, PRED_MT / "perfect.jsonl")
    assert r.aggregated["exact_match"] == 1.0


def test_sentiment_run_gold_unchanged_after_process_docs_default():
    """sentiment run mock:gold takes the default process_docs identity → still 100% accurate."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    task_agg = {k: v for k, v in r.aggregated.items() if k not in {"efficiency", "safety"}}
    assert task_agg == {"accuracy": 1.0, "f1_macro": 1.0, "cohens_kappa": 1.0}


def test_old_task_default_output_type_is_generate_until():
    """The output_type of the old task is still 'generate_until' (and will not be accidentally hit by the new 'none' literal)."""
    assert SentimentClf.output_type == "generate_until"
    assert MT.output_type == "generate_until"


def test_score_run_parity_after_phase4_hooks():
    """After phase 4 transformation, sentiment score / run parity at the task-specific indicator level (architectural anchor).
    Phase 6 introduces cross-cutting efficiency subgroup (call class, only run hangs); wave 3 (DECISIONS §7.2)
    Undo safety cross-cutting, only efficiency remains cross-cutting - should be equivalent when stripped."""
    task = SentimentClf()
    docs = list(task.docs())
    r_run = evaluate_run(task, MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(task, PRED_SENTIMENT / "perfect.jsonl")
    _crosscut = {"efficiency"}
    task_agg = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    task_metrics = lambda d: {k: v for k, v in d.items() if k not in _crosscut}  # noqa: E731
    assert task_agg(r_run.aggregated) == task_agg(r_score.aggregated)
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated
    assert r_run.n == r_score.n
    a_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_run.per_sample]
    o_pairs = [(s.doc_id, s.prediction, s.target, task_metrics(s.metrics)) for s in r_score.per_sample]
    assert a_pairs == o_pairs
