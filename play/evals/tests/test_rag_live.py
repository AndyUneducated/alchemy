"""Phase 4 RAG e2e live: subprocess → real play/rag/query.py → ollama → real VDB.

Double probe gate:
  - ollama_required: ollama service + EVALS_TEST_OLLAMA_MODEL has been pulled
  - panel_vdb_required / test_vdb_required: VDB has been ingested (if missing, skip + prompt ingest command)

CI is clean (default is no ollama / no VDB automatic skip); local dev starts ollama + runs naturally after running ingest.

Test strategy:
  - Use test_vdb (5 lines of facts, ~3s single query) to do the subprocess wrapper smoke test and confirm that the closed loop call is OK
  - Use panel VDB to measure the small limit (limit=2 ~ limit=3) of rag_retrieval / rag_qa,
    Override process_docs injection + recall ordering contract (not complete e2e benchmark, avoid 60s+ timeout)"""

from __future__ import annotations

import os

import pytest

from evals.cli import _build_task_with_optional_deps, parse_model_spec
from evals.models.rag_retrieve import make_retrieve_fn
from evals.runner import evaluate_run
from evals.tests.conftest import (
    ollama_required,
    panel_vdb_required,
    sample_vdb_required,
)


# ---------- subprocess wrapper smoke (test_vdb, smallest corpus) ------------------

@ollama_required
@sample_vdb_required
def test_make_retrieve_fn_returns_real_hits(sample_vdb_path):
    """make_retrieve_fn → subprocess → play/rag/query.py really ran out of ids/contents.

    sample (test_vdb) 5 lines of facts; check 'ZX-7492 project code' should return at least 1 hit, source contains 'project facts.txt'."""
    retrieve = make_retrieve_fn(sample_vdb_path, top_k=3, mode="hybrid")
    ids, contents = retrieve("ZX-7492 项目代号")

    assert len(ids) >= 1
    assert all(isinstance(i, str) for i in ids)
    assert all(isinstance(c, str) for c in contents)
    # The only source file is project facts.txt
    assert any("项目事实" in i or "事实" in i for i in ids)


# ---------- rag_retrieval e2e (panel VDB; limit=2 control time) ------------------

@ollama_required
@panel_vdb_required
def test_rag_retrieval_run_e2e_panel(panel_vdb_path):
    """rag_retrieval + panel VDB → process_docs injects retrieved_ids → ranx calculates recall@5/mrr.

    limit=2 controls the time (each query subprocess ~3s); only checks "process run through + recall>=0",
    No strict threshold is locked (small sample + retriever is greatly affected by configuration, flaky risk)."""
    task = _build_task_with_optional_deps(
        "rag_retrieval",
        vdb=str(panel_vdb_path),
        retrieve_top_k=5,
        retrieve_mode="hybrid",
    )

    # Use retriever tag LM (output_type='none' is not adjusted)
    from evals.cli import _RetrieverOnlyLM
    lm = _RetrieverOnlyLM(name="retriever:panel:hybrid")

    r = evaluate_run(task, lm, limit=2)
    assert r.n == 2
    assert r.mode == "run"
    assert r.model == "retriever:panel:hybrid"
    # All 5 indicators are calculated (all values ​​must be equal even if < 1)
    for m in ("recall@5", "precision@5", "mrr", "ndcg@5", "map@5"):
        assert m in r.aggregated
        assert 0.0 <= r.aggregated[m] <= 1.0
    # The retrieved_ids of at least one sample are not empty (process_docs injection takes effect)
    assert any(len(s.artifacts["pred_ids"]) > 0 for s in r.per_sample)


# ---------- rag_qa e2e (panel VDB + ollama judge; limit=1 further time control)----

_ci_skip_rag_qa_live = pytest.mark.skipif(
    os.environ.get("CI", "").lower() == "true",
    reason="rag_qa live generation is too slow/flaky for GitHub-hosted runners",
)


@_ci_skip_rag_qa_live
@ollama_required
@panel_vdb_required
def test_rag_qa_run_e2e_panel_lexical_only(panel_vdb_path, ollama_model):
    """rag_qa + panel VDB + true ollama answerer + lexical only (no judge_lm).

    limit=1 control time: single query → subprocess retrieval ~3s + ollama generation ~5-10s.
    Lock the two contracts of "process running + lexical indicator calculation", but do not lock the value."""
    task = _build_task_with_optional_deps(
        "rag_qa",
        vdb=str(panel_vdb_path),
        retrieve_top_k=3,
        retrieve_mode="hybrid",
        # Do not pass judge_model_spec → only lexical baseline
    )
    lm = parse_model_spec(f"ollama:{ollama_model}", task)

    r = evaluate_run(task, lm, limit=1)
    assert r.n == 1
    assert r.mode == "run"
    assert "exact_match" in r.aggregated
    assert "rouge_l" in r.aggregated
    # There should be no grounding indicator (judge_lm=None channel)
    assert "faithfulness" not in r.aggregated
    # The retrieved_ids/contexts of a single sample have been injected by process_docs
    [sample] = r.per_sample
    assert len(sample.artifacts["pred_ids"]) > 0
