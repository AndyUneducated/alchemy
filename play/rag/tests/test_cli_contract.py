"""Mirror CLI contract assertions owned by rag itself.

`play/agent_engine/tests/test_tools_subprocess.py` already guards rag's CLI surface
from the consumer side (flag names / `--mode` choices / envelope shape). This suite
mirrors that on **rag's own** tests — so local edits to `query.py` fail
`pytest play/rag/tests/` immediately without going through agent_engine.

Static text scan only; no chromadb / ollama / VDB — avoids `query.py` import side
effects (chromadb / sentence-transformers, etc.).
"""
from __future__ import annotations

import re
from pathlib import Path

QUERY_PY = Path(__file__).resolve().parent.parent / "query.py"


def test_query_py_exists():
    assert QUERY_PY.exists(), f"play/rag/query.py missing at {QUERY_PY}"


def test_query_py_exposes_required_cli_flags():
    """retrieve_docs.handler hard-codes these 6 flags in agent_engine; any rename
    or removal breaks subprocess calls."""
    src = QUERY_PY.read_text(encoding="utf-8")
    required = ["--vdb", "--query", "--top-k", "--mode", "--rerank", "--json"]
    missing = [f for f in required if f not in src]
    assert not missing, (
        f"play/rag/query.py CLI no longer accepts {missing} — agent_engine "
        f"retrieve_docs subprocess will break. Either restore the flag here, "
        f"or update tools/retrieve_docs.py and the cross-project contract test."
    )


def test_query_py_mode_choices_are_dense_bm25_hybrid():
    """`--mode` choices must stay aligned with retrieve_docs tool schema enum
    (agent_engine/tools/retrieve_docs.py); this guards the rag side."""
    src = QUERY_PY.read_text(encoding="utf-8")
    pattern = re.compile(
        r'choices\s*=\s*\[\s*"dense"\s*,\s*"bm25"\s*,\s*"hybrid"\s*\]'
    )
    assert pattern.search(src), (
        "--mode choices changed in play/rag/query.py; agent_engine tool schema "
        "still declares enum=[dense, bm25, hybrid]. Keep them in lockstep."
    )


def test_query_py_documents_envelope_shape():
    """rag CLI `--json` output contract is consumed by `agent_engine.tools.retrieve_docs`
    (`payload['data']` / `payload['meta']`). Changing envelope keys or doc wording
    causes KeyError in consumers — docstring guards here."""
    src = QUERY_PY.read_text(encoding="utf-8")
    assert "{query, data, meta}" in src, (
        "rag/query.py CLI help no longer documents the {query, data, meta} "
        "envelope; sync agent_engine retrieve_docs.handler with the new shape."
    )


def test_query_py_envelope_emits_required_meta_keys():
    """envelope `meta` must include at least `mode / reranked / top_k` (retrieve_docs
    slim projection hard-depends on these three keys). Lower-bound check via source
    literals — no CLI run, avoids chromadb / ollama deps."""
    src = QUERY_PY.read_text(encoding="utf-8")
    for key in ['"vdb"', '"mode"', '"reranked"', '"top_k"']:
        assert key in src, (
            f"envelope meta no longer emits {key}; agent_engine retrieve_docs "
            f"slim projection expects this key."
        )


def test_query_py_uses_precomputed_query_embeddings():
    """After ingest writes embeddings explicitly, query must compute query embedding
    explicitly too; injecting embedding_function into Chroma conflicts with persisted
    collection config."""
    src = QUERY_PY.read_text(encoding="utf-8")
    assert "embedding_function=" not in src
    assert "query_embeddings" in src or "_embed_query" in src
