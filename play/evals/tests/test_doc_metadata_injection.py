"""Phase 4 path B+C data contract: doc.metadata injection path for rag_retrieval / rag_qa.

Zero network/zero VDB - drive process_docs directly with stub retrieve_fn, assert:
  ① rag_retrieval.process_docs writes retrieved_ids into doc.metadata
  ② rag_retrieval uses output_type='none' in the run path to skip the LM call, and process_results can still produce SampleResult
  ③ The injection semantics of load_prediction and process_docs are symmetrical (the shape of doc is the same when the two paths of score / run go to process_results)

Why write this test alone without relying on test_rag_retrieval_score / test_rag_qa_score:
  - Those two tests only cover the score path (load_prediction); the process_docs injection of the run path is another independent codepath
  - "doc.metadata injection" is the core convention of path B+C, specifically designed to test lockable regressions"""

from __future__ import annotations

from typing import Callable

from evals.api import Doc, Request, Response
from evals.models.base import LM
from evals.runner import evaluate_run
from evals.tasks.rag_retrieval import RagRetrieval


class _NoOpLM(LM):
    """nope adapter——rag_retrieval with output_type='none' will not trigger generate_until.

    As a spy: If generate_until is called, the assertion fails."""

    def __init__(self) -> None:
        self.name = "noop"
        self.calls = 0

    def generate_until(self, requests: list[Request]) -> list[Response]:
        self.calls += 1
        raise AssertionError(
            f"output_type='none' 应该跳过 LM 调用，但 generate_until 被触发了 {self.calls} 次"
        )


def _stub_retrieve_fn(mapping: dict[str, list[str]]) -> Callable[[str], tuple[list[str], list[str]]]:
    """Rule retriever: query string → default doc_ids (all checked in mapping).

    Simple and universal: Mapping can be used if it has a key, but will fail if it does not have a key - to avoid having to write a stub for every query."""

    def _retrieve(query: str) -> tuple[list[str], list[str]]:
        ids = mapping.get(query, [])
        # contents and ids are of equal length (rag_retrieval does not consume contents, only rag_qa uses them)
        contents = [f"content for {i}" for i in ids]
        return ids, contents

    return _retrieve


def test_rag_retrieval_run_with_stub_retriever():
    """run path + stub retrieve_fn → process_docs inject retrieved_ids → recall@5=1.0."""
    docs = list(RagRetrieval().docs())  # 8 queries
    # Give each of the first 8 queries a retrieve result that "just hits gold"
    mapping = {}
    for d in docs:
        gold = list(d.metadata["gold_doc_ids"])
        mapping[d.input] = gold + [f"distractor_{i}.txt" for i in range(4)]
    retrieve = _stub_retrieve_fn(mapping)

    task = RagRetrieval(retrieve_fn=retrieve, top_k=10)
    r = evaluate_run(task, _NoOpLM())

    assert r.aggregated["recall@5"] == 1.0
    # process_docs really injects retrieved_ids (the pred_ids pulled by artifacts are not empty)
    for s in r.per_sample:
        assert len(s.artifacts["pred_ids"]) > 0


def test_rag_retrieval_process_docs_injects_metadata_directly():
    """Pure functional behavior of `task.process_docs(docs)`: retrieved_ids appear in every doc.metadata."""
    retrieve = _stub_retrieve_fn({"q1": ["a.txt", "b.txt"]})
    task = RagRetrieval(retrieve_fn=retrieve)

    src = [Doc(id="d1", input="q1", target=None, metadata={"gold_doc_ids": ("a.txt",)})]
    out = task.process_docs(src)

    assert out[0].metadata["retrieved_ids"] == ("a.txt", "b.txt")
    assert out[0].metadata["gold_doc_ids"] == ("a.txt",)  # Old metadata is not overwritten


def test_rag_retrieval_process_docs_identity_when_no_retrieve_fn():
    """retrieve_fn=None → process_docs is identity (default behavior, old tasks are not broken)."""
    task = RagRetrieval(retrieve_fn=None)
    src = [Doc(id="d1", input="q", target=None, metadata={"gold_doc_ids": ("a.txt",)})]
    out = task.process_docs(src)
    assert out == src


def test_rag_retrieval_load_prediction_injects_retrieved_ids():
    """load_prediction: row['retrieved_ids'] → doc.metadata['retrieved_ids'] (score path injection)."""
    task = RagRetrieval()
    doc = Doc(id="r1", input="q", target=None, metadata={"gold_doc_ids": ("a.txt",)})
    enriched, response = task.load_prediction(doc, {"id": "r1", "retrieved_ids": ["a.txt", "b.txt"]})

    assert enriched.metadata["retrieved_ids"] == ("a.txt", "b.txt")
    assert enriched.metadata["gold_doc_ids"] == ("a.txt",)
    # Response placeholder (path B+C: retrieval task has no LM-side data)
    assert response.text is None


def test_run_score_parity_via_metadata_injection():
    """For the same retrieved_ids, whether using process_docs or load_prediction, the aggregation value is the same.

    This is the "two injection paths data equivalence" lock of phase 4 path B+C - to prevent the rag task from secretly bifurcating between the score / run paths."""
    docs = list(RagRetrieval().docs())
    # Use the same mapping to both drive run and write fake predictions
    mapping = {}
    fake_preds = []
    for d in docs:
        gold = list(d.metadata["gold_doc_ids"])
        retrieved = gold + [f"noise_{i}.txt" for i in range(4)]
        mapping[d.input] = retrieved
        fake_preds.append({"id": d.id, "retrieved_ids": retrieved})

    # run path
    task_run = RagRetrieval(retrieve_fn=_stub_retrieve_fn(mapping), top_k=10)
    r_run = evaluate_run(task_run, _NoOpLM())

    # score path: put fake_preds into tmp file
    import json
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for row in fake_preds:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)

    from evals.runner import evaluate_score
    r_score = evaluate_score(RagRetrieval(), tmp_path)

    # Starting from phase 6/7, run multiple cross-cutting subgroups (efficiency/safety), divided into two by ontology (DECISIONS §7.A):
    # - efficiency is call class, only run hangs;
    # - Safety is content class, score / run are dual-linked (response.text is available in both paths).
    # parity holds at the level of task-specific metrics.
    task_agg = lambda d: {k: v for k, v in d.items() if k not in {"efficiency", "safety"}}  # noqa: E731
    assert task_agg(r_run.aggregated) == task_agg(r_score.aggregated)
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated
    assert r_run.n == r_score.n
