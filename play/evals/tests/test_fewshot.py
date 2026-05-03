"""few-shot 机制三条核心断言.

  1. zero-shot prompt 与不传 num_fewshot **字节相同**——保证 Phase 1 的
     `test_active_gold_equals_offline_perfect` 等 parity test 不被破坏（这是 Runner
     里 `if num_fewshot <= 0: return task.doc_to_text(doc)` 早 return 的兑现）
  2. N-shot prompt 包含 N 段 example 且 query 自身**不入选**（否则就成
     "用答案当 example 答自己"的作弊）
  3. 同 fewshot_seed 抽相同 example——可复现性的可执行 contract

测试用 sentiment_clf 作宿主，理由：默认 fewshot_docs/format_fewshot_example
的行为该是任务无关的，sentiment_clf 比 mt 数据轻、跑得快。
"""

from __future__ import annotations

import random

from evals.runner import _build_prompt
from evals.tasks.sentiment_clf import SentimentClf


def _docs():
    return list(SentimentClf().docs())


def test_zero_shot_equals_no_fewshot():
    """num_fewshot=0 → prompt == task.doc_to_text(doc)，字节级相等."""
    task = SentimentClf()
    docs = _docs()
    rng = random.Random(0)
    for doc in docs[:5]:
        prompt = _build_prompt(task, doc, num_fewshot=0, pool=[], rng=rng)
        assert prompt == task.doc_to_text(doc)


def test_n_shot_excludes_self_and_has_n_examples():
    """num_fewshot=2 → prompt 含恰好 2 段 example，且 query doc 自身不入选."""
    task = SentimentClf()
    docs = _docs()
    pool = docs
    rng = random.Random(42)

    query = docs[0]
    prompt = _build_prompt(task, query, num_fewshot=2, pool=pool, rng=rng)

    # 包含 query 自身的 doc_to_text 字符串 1 次（在末尾）
    query_text = task.doc_to_text(query)
    assert prompt.count(query_text) == 1

    # 拆出 example 部分（在 query 之前），按双换行切
    example_blob = prompt.rsplit(query_text, 1)[0].rstrip()
    example_chunks = [c for c in example_blob.split("\n\n") if c.strip()]
    assert len(example_chunks) == 2

    # 每一段 example 都对应 pool 里某条 doc（且不是 query 本身）
    other_docs = [d for d in pool if d.id != query.id]
    expected_strs = {task.format_fewshot_example(d) for d in other_docs}
    for chunk in example_chunks:
        assert chunk in expected_strs


def test_fewshot_seed_determinism():
    """相同 fewshot_seed → 抽出相同的 example 序列（可复现性 contract）."""
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
    """pool 不够时不抛错，能抽几条算几条（小 dataset 边界保护）."""
    task = SentimentClf()
    docs = _docs()
    query = docs[0]
    pool = docs[:3]  # 含 query → 排除自身后只剩 2

    prompt = _build_prompt(task, query, num_fewshot=10, pool=pool, rng=random.Random(0))
    query_text = task.doc_to_text(query)
    example_blob = prompt.rsplit(query_text, 1)[0].rstrip()
    example_chunks = [c for c in example_blob.split("\n\n") if c.strip()]
    assert len(example_chunks) == 2  # 实际能抽到的最大值
