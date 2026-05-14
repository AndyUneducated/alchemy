"""Tools dispatch + retrieve_docs 子进程契约 + 跨子项目 `play/rag/query.py`
CLI 契约的单测.

agent_engine 唯一的"跨子项目硬依赖"是 `tools/retrieve_docs.py` → `play/rag/query.py`
的 subprocess + JSON envelope 握手（DECISIONS §11 / §13 同精神）.这一文件
把这一处契约一次性扣死：

  - `agent_engine.tools.dispatch`：路由 / 未知工具 / `is_error` / `warn_if_error`
  - `retrieve_docs.handler`：
      * subprocess 命令参数（旗标 / 顺序 / `--rerank` 仅在 rerank=True 时追加）
      * stdout JSON envelope `{data, meta}` 的 slim 投影（剔除 query / 多余 meta
        字段，只保留 LLM 真正需要的 `mode / reranked / top_k`）
      * exit code != 0 / 非 JSON → 返 `{"error": ...}`
  - **跨项目契约**：`play/rag/query.py` 的 CLI 必须仍然接受
    `--vdb / --query / --top-k / --mode / --rerank / --json` 6 个 flag，
    且 `--mode` choices 仍含 `dense / bm25 / hybrid`. 这是"rag 改了让 agent_engine
    悄悄坏掉"的最后一道防线——agent_engine 改不到 rag 的源码，但能在自己的 test
    里钉死期望的 CLI surface.
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
    """`dispatch("retrieve_docs", args)` 走 `retrieve_docs.handler`；
    monkeypatch subprocess 让 handler 不真发 chromadb 请求."""
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
    # 多行 error 只取首行
    assert "stack" not in err


def test_warn_if_error_silent_on_ok_envelope(capsys):
    warn_if_error("x", '{"ok": true}')
    assert capsys.readouterr().err == ""


def test_tool_definitions_exposes_retrieve_docs():
    names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
    assert "retrieve_docs" in names


# ---------- retrieve_docs handler -------------------------------------

def test_retrieve_docs_handler_passes_required_flags(monkeypatch: pytest.MonkeyPatch):
    """handler 必须把所有 LLM 提供的字段映射到 query.py 的 CLI flag."""
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
    # 必有这些 flag + value
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
    # rerank=False → 不追加 --rerank
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
    → tool 边界投影为 `{data, meta: {mode, reranked, top_k}}`. 老消费者不应看到
    embedding_model / vdb / query 这些 LLM 不关心的字段."""
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
    """`retrieve_docs._QUERY_SCRIPT` 必须真指向 `play/rag/query.py`. 这道断言
    防 retrieve_docs 内部 path 计算被改坏（DECISIONS §11 / agent_engine 与 rag
    解耦的进程边界依赖此 path 解析）.
    """
    resolved = Path(retrieve_docs._QUERY_SCRIPT).resolve()
    assert resolved == RAG_QUERY_PATH.resolve(), (
        f"retrieve_docs points at {resolved}, expected {RAG_QUERY_PATH}"
    )
    assert resolved.exists(), "play/rag/query.py is missing"


def test_rag_query_cli_surface_still_exposes_required_flags():
    """跨子项目契约：`play/rag/query.py` 的 CLI 仍然接受 retrieve_docs.handler
    会传的所有 flag. 静态文本检查（不需要 chromadb / rag 依赖装好），任何
    flag 重命名 / 删除立即在本测试失败."""
    src = RAG_QUERY_PATH.read_text(encoding="utf-8")
    required_flags = ["--vdb", "--query", "--top-k", "--mode", "--rerank", "--json"]
    missing = [f for f in required_flags if f not in src]
    assert not missing, (
        f"play/rag/query.py no longer accepts {missing} — agent_engine "
        f"retrieve_docs subprocess will break. Either restore the flag in rag "
        f"or update tools/retrieve_docs.py + this contract."
    )


def test_rag_query_mode_choices_still_include_hybrid_dense_bm25():
    """`--mode` choices 仍含三种检索策略；retrieve_docs 的 tool schema enum 与之
    严格对齐（tools/retrieve_docs.py 第 ~45 行）."""
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
    """rag CLI doc 仍承诺 `{query, data, meta}` envelope（DECISIONS §11 同精神）.
    retrieve_docs.handler 直接索引 `payload["data"]` 与 `payload["meta"][...]`,
    rag 改 envelope key 会让 handler 抛 KeyError."""
    src = RAG_QUERY_PATH.read_text(encoding="utf-8")
    assert "{query, data, meta}" in src, (
        "rag/query.py CLI help no longer documents the {query, data, meta} "
        "envelope; sync retrieve_docs.handler with whatever the new shape is."
    )
