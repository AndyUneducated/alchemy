"""rag 自己持有的 CLI 契约镜像断言。

`play/agent_engine/tests/test_tools_subprocess.py` 已经从消费者一侧守住了
rag 的 CLI surface（flag 名 / `--mode` choices / envelope shape）。这里在
**rag 自己**的测试集里再钉一份镜像 —— 这样在 rag 仓库内本地修改 `query.py`
时 `pytest play/rag/tests/` 立刻红，不必绕到 agent_engine 才发现。

只做静态文本扫描，无需 chromadb / ollama / VDB —— `query.py` 的导入副作用
（chromadb / sentence-transformers 等）一律避开。
"""
from __future__ import annotations

import re
from pathlib import Path

QUERY_PY = Path(__file__).resolve().parent.parent / "query.py"


def test_query_py_exists():
    assert QUERY_PY.exists(), f"play/rag/query.py missing at {QUERY_PY}"


def test_query_py_exposes_required_cli_flags():
    """retrieve_docs.handler 在 agent_engine 端硬编码了这 6 个 flag；任何
    重命名 / 删除都会让 subprocess 调用失败。"""
    src = QUERY_PY.read_text(encoding="utf-8")
    required = ["--vdb", "--query", "--top-k", "--mode", "--rerank", "--json"]
    missing = [f for f in required if f not in src]
    assert not missing, (
        f"play/rag/query.py CLI no longer accepts {missing} — agent_engine "
        f"retrieve_docs subprocess will break. Either restore the flag here, "
        f"or update tools/retrieve_docs.py and the cross-project contract test."
    )


def test_query_py_mode_choices_are_dense_bm25_hybrid():
    """`--mode` choices 三个字面量与 retrieve_docs 工具 schema 的 enum 严格
    对齐（agent_engine/tools/retrieve_docs.py）；这里护住 rag 一侧。"""
    src = QUERY_PY.read_text(encoding="utf-8")
    pattern = re.compile(
        r'choices\s*=\s*\[\s*"dense"\s*,\s*"bm25"\s*,\s*"hybrid"\s*\]'
    )
    assert pattern.search(src), (
        "--mode choices changed in play/rag/query.py; agent_engine tool schema "
        "still declares enum=[dense, bm25, hybrid]. Keep them in lockstep."
    )


def test_query_py_documents_envelope_shape():
    """rag CLI 的 `--json` 输出契约文本被 `agent_engine.tools.retrieve_docs`
    直接消费 (`payload['data']` / `payload['meta']`)。改 envelope key 或 doc
    措辞会让消费者抛 KeyError，这里用文档字符串守住。"""
    src = QUERY_PY.read_text(encoding="utf-8")
    assert "{query, data, meta}" in src, (
        "rag/query.py CLI help no longer documents the {query, data, meta} "
        "envelope; sync agent_engine retrieve_docs.handler with the new shape."
    )


def test_query_py_envelope_emits_required_meta_keys():
    """envelope 的 `meta` 至少包含 `mode / reranked / top_k`（retrieve_docs
    的 slim 投影硬依赖这三个 key）。这里用源码层面的 key 字面量出现做下界
    检查 —— 不跑 CLI，避免拖 chromadb / ollama 依赖。"""
    src = QUERY_PY.read_text(encoding="utf-8")
    for key in ['"vdb"', '"mode"', '"reranked"', '"top_k"']:
        assert key in src, (
            f"envelope meta no longer emits {key}; agent_engine retrieve_docs "
            f"slim projection expects this key."
        )
