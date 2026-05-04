"""models/rag_retrieve.py 单元测试：subprocess + JSON envelope 解析逻辑.

零网络 / 零 VDB：用 monkeypatch 替换 subprocess.run 拦截调用 + 注入伪 envelope，
锁住"调用形参"和"envelope → (ids, contents) 解析"两条契约.

live e2e（真跑 play/rag/query.py + ollama）放在 test_rag_live.py 走 vdb-probe gate.
"""

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
    # cwd 必须 = RAG_DIR（play/rag/query.py 走相对 import config / bm25）
    assert captured["cwd"] == str(RAG_DIR)
    assert ids == ["a.txt"]
    assert contents == ["x"]


def test_rerank_flag_added_when_enabled(monkeypatch):
    """rerank=True → 命令多一个 --rerank flag."""
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
    """同源 chunk 多条 → 仅保留首位 rank（去重 by source）."""

    def fake_run(cmd, **kwargs):
        envelope = {
            "query": "q",
            "data": [
                {"source": "doc_a.txt", "content": "chunk 1 of A"},
                {"source": "doc_b.txt", "content": "chunk 1 of B"},
                {"source": "doc_a.txt", "content": "chunk 2 of A"},  # 同源应被剔
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
    """play/rag/query.py 非零退出 → RuntimeError 携 stderr（fail-fast 而非静默空 list）."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Ollama not reachable"
        )

    monkeypatch.setattr(rag_retrieve.subprocess, "run", fake_run)

    fn = make_retrieve_fn("/tmp/vdb")
    with pytest.raises(RuntimeError, match="Ollama not reachable"):
        fn("q")


def test_skip_empty_source_chunks(monkeypatch):
    """source 字段缺失或空 → 该 chunk 被跳（不污染 ids 列表）."""

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
