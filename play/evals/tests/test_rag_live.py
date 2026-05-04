"""Phase 4 RAG e2e live：subprocess → real play/rag/query.py → ollama → 真 VDB.

双 probe gate：
  - ollama_required：ollama 服务 + EVALS_TEST_OLLAMA_MODEL 已 pull
  - panel_vdb_required / test_vdb_required：VDB 已 ingest（缺则 skip + 提示 ingest 命令）

CI 干净（默认无 ollama / 无 VDB 自动 skip）；本地 dev 起 ollama + 跑过 ingest 后自然就跑.

测试 strategy：
  - 用 test_vdb（5 行 facts，~3s 单查询）做 subprocess wrapper 烟雾测试，确认调用闭环 OK
  - 用 panel VDB 做 rag_retrieval / rag_qa 的小 limit (limit=2 ~ limit=3) 实测，
    覆盖 process_docs 注入 + recall 排序契约（非完全 e2e benchmark，避免 60s+ 超时）
"""

from __future__ import annotations

import pytest

from evals.cli import _build_task_with_optional_deps, parse_model_spec
from evals.models.rag_retrieve import make_retrieve_fn
from evals.runner import evaluate_run
from evals.tests.conftest import (
    ollama_required,
    panel_vdb_required,
    sample_vdb_required,
)


# ---------- subprocess wrapper smoke (test_vdb，最小 corpus) -----------------

@ollama_required
@sample_vdb_required
def test_make_retrieve_fn_returns_real_hits(sample_vdb_path):
    """make_retrieve_fn → subprocess → play/rag/query.py 真跑出 ids/contents.

    sample (test_vdb) 5 行 facts；查 'ZX-7492 项目代号' 应至少返回 1 条 hit，source 含 '项目事实.txt'.
    """
    retrieve = make_retrieve_fn(sample_vdb_path, top_k=3, mode="hybrid")
    ids, contents = retrieve("ZX-7492 项目代号")

    assert len(ids) >= 1
    assert all(isinstance(i, str) for i in ids)
    assert all(isinstance(c, str) for c in contents)
    # 唯一 source 文件就是项目事实.txt
    assert any("项目事实" in i or "事实" in i for i in ids)


# ---------- rag_retrieval e2e（panel VDB；limit=2 控时间）-------------------

@ollama_required
@panel_vdb_required
def test_rag_retrieval_run_e2e_panel(panel_vdb_path):
    """rag_retrieval + panel VDB → process_docs 注入 retrieved_ids → ranx 算 recall@5/mrr.

    limit=2 控时间（每条 query subprocess ~3s）；只验"流程跑通 + recall>=0"，
    不锁严格阈值（小样本 + retriever 受配置影响大，flaky 风险）.
    """
    task = _build_task_with_optional_deps(
        "rag_retrieval",
        vdb=str(panel_vdb_path),
        retrieve_top_k=5,
        retrieve_mode="hybrid",
    )

    # 用 retriever 标签 LM（output_type='none' 不调）
    from evals.cli import _RetrieverOnlyLM
    lm = _RetrieverOnlyLM(name="retriever:panel:hybrid")

    r = evaluate_run(task, lm, limit=2)
    assert r.n == 2
    assert r.mode == "run"
    assert r.model == "retriever:panel:hybrid"
    # 5 个指标都被算出（即便值 < 1 也要齐）
    for m in ("recall@5", "precision@5", "mrr", "ndcg@5", "map@5"):
        assert m in r.aggregated
        assert 0.0 <= r.aggregated[m] <= 1.0
    # 至少一条 sample 的 retrieved_ids 非空（process_docs 注入生效）
    assert any(len(s.artifacts["pred_ids"]) > 0 for s in r.per_sample)


# ---------- rag_qa e2e（panel VDB + ollama judge；limit=1 进一步控时间）----

@ollama_required
@panel_vdb_required
def test_rag_qa_run_e2e_panel_lexical_only(panel_vdb_path, ollama_model):
    """rag_qa + panel VDB + 真 ollama answerer + lexical only（无 judge_lm）.

    limit=1 控时间：单 query → subprocess 检索 ~3s + ollama 生成 ~5-10s.
    锁"流程跑通 + lexical 指标算出" 二条契约，不锁数值.
    """
    task = _build_task_with_optional_deps(
        "rag_qa",
        vdb=str(panel_vdb_path),
        retrieve_top_k=3,
        retrieve_mode="hybrid",
        # 不传 judge_model_spec → 仅 lexical baseline
    )
    lm = parse_model_spec(f"ollama:{ollama_model}", task)

    r = evaluate_run(task, lm, limit=1)
    assert r.n == 1
    assert r.mode == "run"
    assert "exact_match" in r.aggregated
    assert "rouge_l" in r.aggregated
    # 不应有 grounding 指标（judge_lm=None 通路）
    assert "faithfulness" not in r.aggregated
    # 单条 sample 的 retrieved_ids / contexts 已经被 process_docs 注入
    [sample] = r.per_sample
    assert len(sample.artifacts["pred_ids"]) > 0
