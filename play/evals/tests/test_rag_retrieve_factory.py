"""models/rag_retrieve.py unit test: subprocess + JSON envelope parsing logic.

Zero Network/Zero VDB: Replace subprocess.run interception calls with monkeypatch + inject fake envelope,
Lock the two contracts of "calling formal parameters" and "envelope → (ids, contents) parsing".

live e2e (real run play/rag/query.py + ollama) is placed in test_rag_live.py and runs vdb-probe gate."""

from __future__ import annotations

import json
import subprocess

import pytest

from evals.models import rag_retrieve
from evals.models.rag_retrieve import RAG_DIR, make_retrieve_fn


def test_subprocess_command_shape(monkeypatch):
    """make_retrieve_fn 调用 subprocess.run 时，参数列表必须含 --vdb / --query / --top-k / --mode / --json."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        envelope = {"query": "q", "data": [{"source": "a.txt", "content": "x"}], "meta": {}}
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb", top_k=7, mode="hybrid")
    ids, contents = fn("how does X work?")

    assert "--vdb" in captured["cmd"]
    assert "--query" in captured["cmd"]
    assert "how does X work?" in captured["cmd"]
    assert "--top-k" in captured["cmd"]
    assert "7" in captured["cmd"]
    assert "--mode" in captured["cmd"]
    assert "hybrid" in captured["cmd"]
    assert "--json" in captured["cmd"]
    # cwd must = RAG_DIR (play/rag/query.py is relative to import config/bm25)
    assert captured["cwd"] == str(RAG_DIR)
    assert ids == ["a.txt"]
    assert contents == ["x"]


def test_rerank_flag_added_when_enabled(monkeypatch):
    """rerank=True → One more command --rerank flag."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, returncode=0,
            stdout=json.dumps({"query": "q", "data": [], "meta": {}}),
            stderr="",
        )

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb", rerank=True)
    fn("q")
    assert "--rerank" in captured["cmd"]


def test_dedup_chunks_to_unique_sources(monkeypatch):
    """Multiple chunks from the same origin → only retain the first rank (remove duplication by source)."""

    def fake_run(cmd, **kwargs):
        envelope = {
            "query": "q",
            "data": [
                {"source": "doc_a.txt", "content": "chunk 1 of A"},
                {"source": "doc_b.txt", "content": "chunk 1 of B"},
                {"source": "doc_a.txt", "content": "chunk 2 of A"},  # Homology should be eliminated
                {"source": "doc_c.txt", "content": "chunk 1 of C"},
            ],
            "meta": {},
        }
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb")
    ids, contents = fn("q")

    assert ids == ["doc_a.txt", "doc_b.txt", "doc_c.txt"]
    assert contents == ["chunk 1 of A", "chunk 1 of B", "chunk 1 of C"]


def test_subprocess_failure_raises_with_stderr(monkeypatch):
    """play/rag/query.py non-zero exit → RuntimeError with stderr (fail-fast instead of silent empty list)."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Ollama not reachable"
        )

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb")
    with pytest.raises(RuntimeError, match="Ollama not reachable"):
        fn("q")


def test_skip_empty_source_chunks(monkeypatch):
    """The source field is missing or empty → the chunk is skipped (does not pollute the ids list)."""

    def fake_run(cmd, **kwargs):
        envelope = {
            "query": "q",
            "data": [
                {"source": "", "content": "no source"},
                {"source": "valid.txt", "content": "ok"},
            ],
            "meta": {},
        }
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb")
    ids, _ = fn("q")
    assert ids == ["valid.txt"]
