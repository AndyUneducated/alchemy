"""ingest-side contract tests that avoid live Ollama / ChromaDB setup."""
from __future__ import annotations

import ast
from pathlib import Path

INGEST_PY = Path(__file__).resolve().parent.parent / "ingest.py"


def test_ingest_upsert_uses_precomputed_embeddings():
    """CI fixtures should not ask Chroma to embed implicitly during upsert."""
    tree = ast.parse(INGEST_PY.read_text(encoding="utf-8"))
    upsert_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "upsert"
    ]

    assert upsert_calls, "play/rag/ingest.py no longer calls collection.upsert"
    assert any(
        any(keyword.arg == "embeddings" for keyword in call.keywords)
        for call in upsert_calls
    ), (
        "ingest.py must pass precomputed embeddings into Chroma upsert; "
        "otherwise Chroma embeds the same documents internally and CI can "
        "timeout inside upsert."
    )


def test_ingest_embeds_in_batches():
    src = INGEST_PY.read_text(encoding="utf-8")
    assert "RAG_EMBED_BATCH_SIZE" in src
    assert "_embed_documents" in src
