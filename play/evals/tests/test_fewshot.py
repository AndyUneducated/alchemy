"""The few-shot mechanism has three core assertions.

  1. Zero-shot prompt is the same as not passing num_fewshot **bytes** - guaranteed for Phase 1
     `test_active_gold_equals_offline_perfect` etc. parity test is not destroyed (this is Runner
     In `if num_fewshot <= 0: return task.doc_to_text(doc)` (early return implementation)
  2. N-shot prompt contains N paragraphs of example and the query itself is not selected (otherwise it becomes
     "Answer yourself using the answer as an example" cheating)
  3. Take the same example as fewshot_seed - an executable contract with reproducibility

The test uses sentiment_clf as the host, reason: default fewshot_docs/format_fewshot_example
The behavior should be task-independent, sentiment_clf is lighter and faster than mt data."""

from __future__ import annotations

import random

from evals.runner import _build_prompt
from evals.tasks.sentiment_clf import SentimentClf


def _docs():
    return list(SentimentClf().docs())


def test_zero_shot_equals_no_fewshot():
    """num_fewshot=0 → prompt == task.doc_to_text(doc), byte-level equality."""
    task = SentimentClf()
    docs = _docs()
    rng = random.Random(0)
    for doc in docs[:5]:
        prompt = _build_prompt(task, doc, num_fewshot=0, pool=[], rng=rng)
        assert prompt == task.doc_to_text(doc)


def test_n_shot_excludes_self_and_has_n_examples():
    """num_fewshot=2 → prompt contains exactly 2 examples, and query doc itself is not selected."""
    task = SentimentClf()
    docs = _docs()
    pool = docs
    rng = random.Random(42)

    query = docs[0]
    prompt = _build_prompt(task, query, num_fewshot=2, pool=pool, rng=rng)

    # Contains the doc_to_text string of query itself 1 time (at the end)
    query_text = task.doc_to_text(query)
    assert prompt.count(query_text) == 1

    # Separate the example part (before query) and press double newline to switch
    example_blob = prompt.rsplit(query_text, 1)[0].rstrip()
    example_chunks = [c for c in example_blob.split("\n\n") if c.strip()]
    assert len(example_chunks) == 2

    # Each example corresponds to a certain doc in the pool (and not the query itself)
    other_docs = [d for d in pool if d.id != query.id]
    expected_strs = {task.format_fewshot_example(d) for d in other_docs}
    for chunk in example_chunks:
        assert chunk in expected_strs


def test_fewshot_seed_determinism():
    """Same fewshot_seed → extract the same example sequence (reproducibility contract)."""
    task = SentimentClf()
    docs = _docs()
    pool = docs
    query = docs[0]

    p1 = _build_prompt(task, query, num_fewshot=3, pool=pool, rng=random.Random(7))
    p2 = _build_prompt(task, query, num_fewshot=3, pool=pool, rng=random.Random(7))
    p_diff = _build_prompt(task, query, num_fewshot=3, pool=pool, rng=random.Random(8))

    assert p1 == p2
    assert p1 != p_diff


def test_fewshot_pool_smaller_than_n_does_not_raise():
    """No error will be thrown when the pool is not enough, and the number of items that can be extracted will be counted (small dataset boundary protection)."""
    task = SentimentClf()
    docs = _docs()
    query = docs[0]
    pool = docs[:3]  # with query → excluding itself only 2 remains

    prompt = _build_prompt(task, query, num_fewshot=10, pool=pool, rng=random.Random(0))
    query_text = task.doc_to_text(query)
    example_blob = prompt.rsplit(query_text, 1)[0].rstrip()
    example_chunks = [c for c in example_blob.split("\n\n") if c.strip()]
    assert len(example_chunks) == 2  # The maximum value that can actually be drawn
