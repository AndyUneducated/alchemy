"""Phase 4 path B+C 数据契约：rag_retrieval / rag_qa 的 doc.metadata 注入路径.

零网络 / 零 VDB——用 stub retrieve_fn 直接驱动 process_docs，断言：
  ① rag_retrieval.process_docs 把 retrieved_ids 写进 doc.metadata
  ② rag_retrieval 在 run 路径用 output_type='none' 跳过 LM 调用，process_results 仍能产 SampleResult
  ③ load_prediction 与 process_docs 的注入语义对称（score / run 两路径走到 process_results 时 doc 形状一致）

为什么单写这个测试而不依赖 test_rag_retrieval_score / test_rag_qa_score：
  - 那两个测试只覆盖 score 路径（load_prediction）；run 路径的 process_docs 注入是另一条独立 codepath
  - "doc.metadata 注入"是 path B+C 的核心约定，专测可锁回归
"""

from __future__ import annotations

from typing import Callable

from evals.api import Doc, Request, Response
from evals.models.base import LM
from evals.runner import evaluate_run
from evals.tasks.rag_retrieval import RagRetrieval


class _NoOpLM(LM):
    """nope adapter——rag_retrieval 走 output_type='none' 不会触发 generate_until.

    作 spy：如果 generate_until 被调，断言失败.
    """

    def __init__(self) -> None:
        self.name = "noop"
        self.calls = 0

    def generate_until(self, requests: list[Request]) -> list[Response]:
        self.calls += 1
        raise AssertionError(
            f"output_type='none' 应该跳过 LM 调用，但 generate_until 被触发了 {self.calls} 次"
        )


def _stub_retrieve_fn(mapping: dict[str, list[str]]) -> Callable[[str], tuple[list[str], list[str]]]:
    """规则 retriever：query 字符串 → 预设 doc_ids（全在 mapping 里查）.

    简洁通用：mapping 有 key 用之，无 key 落空——避免每个 query 都要写 stub.
    """

    def _retrieve(query: str) -> tuple[list[str], list[str]]:
        ids = mapping.get(query, [])
        # contents 与 ids 等长占位（rag_retrieval 不消费 contents，rag_qa 才用）
        contents = [f"content for {i}" for i in ids]
        return ids, contents

    return _retrieve


def test_rag_retrieval_run_with_stub_retriever():
    """run 路径 + stub retrieve_fn → process_docs 注入 retrieved_ids → recall@5=1.0."""
    docs = list(RagRetrieval().docs())  # 8 条 query
    # 给前 8 条 query 各自放一个"恰好命中 gold 的"retrieve 结果
    mapping = {}
    for d in docs:
        gold = list(d.metadata["gold_doc_ids"])
        mapping[d.input] = gold + [f"distractor_{i}.txt" for i in range(4)]
    retrieve = _stub_retrieve_fn(mapping)

    task = RagRetrieval(retrieve_fn=retrieve, top_k=10)
    r = evaluate_run(task, _NoOpLM())

    assert r.aggregated["recall@5"] == 1.0
    # process_docs 真的注入了 retrieved_ids（artifacts 拉到的 pred_ids 非空）
    for s in r.per_sample:
        assert len(s.artifacts["pred_ids"]) > 0


def test_rag_retrieval_process_docs_injects_metadata_directly():
    """`task.process_docs(docs)` 的纯函数行为：retrieved_ids 出现在每条 doc.metadata 里."""
    retrieve = _stub_retrieve_fn({"q1": ["a.txt", "b.txt"]})
    task = RagRetrieval(retrieve_fn=retrieve)

    src = [Doc(id="d1", input="q1", target=None, metadata={"gold_doc_ids": ("a.txt",)})]
    out = task.process_docs(src)

    assert out[0].metadata["retrieved_ids"] == ("a.txt", "b.txt")
    assert out[0].metadata["gold_doc_ids"] == ("a.txt",)  # 老 metadata 不被覆盖


def test_rag_retrieval_process_docs_identity_when_no_retrieve_fn():
    """retrieve_fn=None → process_docs 是 identity（默认行为，老 task 不破）."""
    task = RagRetrieval(retrieve_fn=None)
    src = [Doc(id="d1", input="q", target=None, metadata={"gold_doc_ids": ("a.txt",)})]
    out = task.process_docs(src)
    assert out == src


def test_rag_retrieval_load_prediction_injects_retrieved_ids():
    """load_prediction：row['retrieved_ids'] → doc.metadata['retrieved_ids']（score 路径注入）."""
    task = RagRetrieval()
    doc = Doc(id="r1", input="q", target=None, metadata={"gold_doc_ids": ("a.txt",)})
    enriched, response = task.load_prediction(doc, {"id": "r1", "retrieved_ids": ["a.txt", "b.txt"]})

    assert enriched.metadata["retrieved_ids"] == ("a.txt", "b.txt")
    assert enriched.metadata["gold_doc_ids"] == ("a.txt",)
    # Response 占位（path B+C：retrieval task 无 LM-side 数据）
    assert response.text is None


def test_run_score_parity_via_metadata_injection():
    """同样的 retrieved_ids，无论走 process_docs 还是 load_prediction，aggregation 数值相同.

    这是 phase 4 path B+C 的"两条注入路径数据等价"锁——避免 rag task 在 score / run 两路径偷偷分叉.
    """
    docs = list(RagRetrieval().docs())
    # 用同一个 mapping 既驱动 run 也写出 fake predictions
    mapping = {}
    fake_preds = []
    for d in docs:
        gold = list(d.metadata["gold_doc_ids"])
        retrieved = gold + [f"noise_{i}.txt" for i in range(4)]
        mapping[d.input] = retrieved
        fake_preds.append({"id": d.id, "retrieved_ids": retrieved})

    # run 路径
    task_run = RagRetrieval(retrieve_fn=_stub_retrieve_fn(mapping), top_k=10)
    r_run = evaluate_run(task_run, _NoOpLM())

    # score 路径：把 fake_preds 落 tmp 文件
    import json
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in fake_preds:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)

    from evals.runner import evaluate_score
    r_score = evaluate_score(RagRetrieval(), tmp_path)

    # phase 6 起 run 多 efficiency 子组（cross-cutting AOP），score 路径无 LM 调用故不注入；
    # parity 在 task-specific 指标层面成立.
    task_agg = lambda d: {k: v for k, v in d.items() if k != "efficiency"}  # noqa: E731
    assert task_agg(r_run.aggregated) == task_agg(r_score.aggregated)
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated
    assert r_run.n == r_score.n
