"""Tools dispatch + retrieve_docs subprocess contract + cross-subproject `play/rag/query.py`
CLI contract tests.

agent_engine's only cross-subproject hard dependency is `tools/retrieve_docs.py` → `play/rag/query.py`
subprocess + JSON envelope handshake (DECISIONS §11 / §13 same spirit). This file
locks that contract in one place:

  - `agent_engine.tools.dispatch`: routing / unknown tool / `is_error` / `warn_if_error`
  - `retrieve_docs.handler`:
      * subprocess command args (flags / order / `--rerank` only when rerank=True)
      * slim projection of stdout JSON envelope `{data, meta}` (drop query / extra meta
        fields; keep only `mode / reranked / top_k` the LLM needs)
      * exit code != 0 / non-JSON → return `{"error": ...}`
  - **Cross-project contract**: `play/rag/query.py` CLI must still accept
    `--vdb / --query / --top-k / --mode / --rerank / --json` six flags,
    and `--mode` choices still include `dense / bm25 / hybrid`. Last line of defense
    against silent rag breaks — agent_engine cannot edit rag source but can pin expected CLI surface in tests.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from agent_engine.tools import (
    TOOL_DEFINITIONS,
    dispatch,
    is_error,
    retrieve_docs,
    warn_if_error,
)
from agent_engine.tools._envelope import is_error as is_error_priv

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_QUERY_PATH = REPO_ROOT / "play" / "rag" / "query.py"


# ---------- dispatch / envelope ---------------------------------------

def test_dispatch_unknown_tool_returns_error_envelope():
    out = json.loads(dispatch("nope", {}))
    assert "error" in out
    assert "Unknown tool" in out["error"]


def test_dispatch_routes_retrieve_docs_to_handler(monkeypatch: pytest.MonkeyPatch):
    """`dispatch("retrieve_docs", args)` goes through `retrieve_docs.handler`;
    monkeypatch subprocess so handler does not hit chromadb."""
    captured: dict = {}

    def fake_subprocess(cmd):
        captured["cmd"] = list(cmd)
        return 0, {
            "query": "q",
            "data": [{"content": "x", "score": 0.9, "source": "s", "metadata": {}}],
            "meta": {"mode": "hybrid", "reranked": False, "top_k": 3, "extra": "drop"},
        }
    monkeypatch.setattr(retrieve_docs, "run_json_subprocess", fake_subprocess)
    out = json.loads(dispatch("retrieve_docs", {"query": "q", "vdb_dir": "/v"}))
    assert captured["cmd"], "dispatch failed to invoke the subprocess shim"
    assert out["data"][0]["content"] == "x"
    assert out["meta"] == {"mode": "hybrid", "reranked": False, "top_k": 3}, (
        "tool boundary must slim the envelope to fields the LLM actually needs"
    )


def test_is_error_recognizes_error_envelope_and_ignores_non_json():
    assert is_error('{"error": "boom"}') is True
    assert is_error('{"ok": true}') is False
    assert is_error("not json at all") is False
    assert is_error_priv('{"error": "boom"}') is True


def test_warn_if_error_writes_first_line_to_stderr(capsys):
    warn_if_error("xtool", '{"error": "boom\\nstack"}')
    err = capsys.readouterr().err
    assert "WARNING: tool xtool failed: boom" in err
    # multi-line error: take first line only
    assert "stack" not in err


def test_warn_if_error_silent_on_ok_envelope(capsys):
    warn_if_error("x", '{"ok": true}')
    assert capsys.readouterr().err == ""


def test_tool_definitions_exposes_retrieve_docs():
    names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
    assert "retrieve_docs" in names


# ---------- retrieve_docs handler -------------------------------------

def test_retrieve_docs_handler_passes_required_flags(monkeypatch: pytest.MonkeyPatch):
    """handler must map all LLM-provided fields to query.py CLI flags."""
    captured: dict = {}

    def fake_subprocess(cmd):
        captured["cmd"] = list(cmd)
        return 0, {
            "query": "q", "data": [],
            "meta": {"mode": "dense", "reranked": False, "top_k": 7},
        }
    monkeypatch.setattr(retrieve_docs, "run_json_subprocess", fake_subprocess)

    retrieve_docs.handler(
        query="关键词", vdb_dir="/path/to/vdb",
        top_k=7, mode="dense", rerank=False,
    )
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("query.py"), (
        f"unexpected script path: {cmd[1]} — should resolve to play/rag/query.py"
    )
    # required flags + values
    for pair in [
        ("--vdb", "/path/to/vdb"),
        ("--query", "关键词"),
        ("--top-k", "7"),
        ("--mode", "dense"),
    ]:
        idx = cmd.index(pair[0])
        assert cmd[idx + 1] == pair[1], (
            f"{pair[0]} should be followed by {pair[1]!r}; got {cmd[idx + 1]!r}"
        )
    assert "--json" in cmd
    # rerank=False → do not append --rerank
    assert "--rerank" not in cmd


def test_retrieve_docs_handler_appends_rerank_flag_only_when_true(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    def fake_subprocess(cmd):
        captured["cmd"] = list(cmd)
        return 0, {"data": [], "meta": {"mode": "hybrid", "reranked": True, "top_k": 3}}
    monkeypatch.setattr(retrieve_docs, "run_json_subprocess", fake_subprocess)

    retrieve_docs.handler(query="q", vdb_dir="/v", rerank=True)
    assert "--rerank" in captured["cmd"]


def test_retrieve_docs_handler_returns_error_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        retrieve_docs, "run_json_subprocess", lambda cmd: (2, None),
    )
    out = json.loads(retrieve_docs.handler(query="q", vdb_dir="/v"))
    assert "error" in out
    assert "exited with code 2" in out["error"]


def test_retrieve_docs_handler_returns_error_on_non_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        retrieve_docs, "run_json_subprocess", lambda cmd: (0, None),
    )
    out = json.loads(retrieve_docs.handler(query="q", vdb_dir="/v"))
    assert "error" in out
    assert "non-JSON output" in out["error"]


def test_retrieve_docs_handler_slims_envelope_to_data_and_meta(
    monkeypatch: pytest.MonkeyPatch,
):
    """rag envelope `{query, data, meta: {mode, reranked, top_k, embedding_model, vdb}}`
    → tool boundary projects to `{data, meta: {mode, reranked, top_k}}`. Old consumers
    must not see embedding_model / vdb / query fields the LLM does not need."""
    monkeypatch.setattr(
        retrieve_docs, "run_json_subprocess", lambda cmd: (0, {
            "query": "q",
            "data": [{"content": "c", "score": 1.0, "source": "s", "metadata": {}}],
            "meta": {
                "mode": "bm25", "reranked": True, "top_k": 5,
                "embedding_model": "should-be-dropped",
                "vdb": "/some/path",
            },
        }),
    )
    out = json.loads(retrieve_docs.handler(query="q", vdb_dir="/v"))
    assert set(out.keys()) == {"data", "meta"}
    assert set(out["meta"].keys()) == {"mode", "reranked", "top_k"}
    assert out["meta"]["mode"] == "bm25"


# ---------- cross-project: play/rag/query.py CLI contract -------------

def test_rag_query_script_path_exists_and_resolves_under_play():
    """`retrieve_docs._QUERY_SCRIPT` must point at `play/rag/query.py`. Guards against
    broken internal path resolution (DECISIONS §11 / process boundary between agent_engine and rag).
    """
    resolved = Path(retrieve_docs._QUERY_SCRIPT).resolve()
    assert resolved == RAG_QUERY_PATH.resolve(), (
        f"retrieve_docs points at {resolved}, expected {RAG_QUERY_PATH}"
    )
    assert resolved.exists(), "play/rag/query.py is missing"


def test_rag_query_cli_surface_still_exposes_required_flags():
    """Cross-project contract: `play/rag/query.py` CLI still accepts all flags
    retrieve_docs.handler passes. Static text check (no chromadb / rag deps needed);
    any flag rename/delete fails here immediately."""
    src = RAG_QUERY_PATH.read_text(encoding="utf-8")
    required_flags = ["--vdb", "--query", "--top-k", "--mode", "--rerank", "--json"]
    missing = [f for f in required_flags if f not in src]
    assert not missing, (
        f"play/rag/query.py no longer accepts {missing} — agent_engine "
        f"retrieve_docs subprocess will break. Either restore the flag in rag "
        f"or update tools/retrieve_docs.py + this contract."
    )


def test_rag_query_mode_choices_still_include_hybrid_dense_bm25():
    """`--mode` choices still include three retrieval strategies; retrieve_docs tool schema enum
    strictly aligned (tools/retrieve_docs.py ~line 45)."""
    src = RAG_QUERY_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r'choices\s*=\s*\[\s*"dense"\s*,\s*"bm25"\s*,\s*"hybrid"\s*\]'
    )
    assert pattern.search(src), (
        "play/rag/query.py --mode choices changed; the LLM-facing tool schema "
        "in tools/retrieve_docs.py declares enum=[dense, bm25, hybrid] — keep "
        "them in lockstep or LLM choices will drift from real CLI behavior."
    )


def test_rag_query_json_envelope_shape_documented():
    """rag CLI doc still promises `{query, data, meta}` envelope (DECISIONS §11 same spirit).
    retrieve_docs.handler indexes `payload["data"]` and `payload["meta"][...]` directly;
    rag envelope key changes cause handler KeyError."""
    src = RAG_QUERY_PATH.read_text(encoding="utf-8")
    assert "{query, data, meta}" in src, (
        "rag/query.py CLI help no longer documents the {query, data, meta} "
        "envelope; sync retrieve_docs.handler with whatever the new shape is."
    )
