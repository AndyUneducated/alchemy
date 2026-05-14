"""Smoke：跨 SDK / CLI / Tracer 的最小可用性断言.

针对"别的模块（外部 SDK / OS / 子项目）改动让本模块不可用"的兜底——任一项
失败都意味着 agent_engine 在当前环境跑不起来：

  - 4 个 backend client 模块可独立 import（按 SDK 安装状态 skip 缺失项）.
    每个客户端在 module-level 实例化 SDK 客户端对象 (`anthropic.Anthropic(...)` /
    `OpenAI(...)` / `genai.Client(...)`)，SDK 改 ABI 会让 import 直接挂——
    这是最早期的 ABI 回归报警.
  - `python -m agent_engine --help` 出 help text 且 exit code 0（CLI 入口
    没被改坏）.
  - `ToolTracer.record / drain` 形态稳定：`visible=False`、`ok` 由 `is_error`
    决定、stderr 一行 `🔧` emoji——memory 投影 / observability 都靠这个不变.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_engine.result import ToolCallEntry
from agent_engine.tracer import ToolTracer

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"


# ---------- backend client SDK importability --------------------------

_BACKENDS = [
    # (module_name, sdk_module, key_config_attr)
    # key_config_attr: 若 SDK 在 client 构造期校验 key，需配 config.* 非空才能 import；
    # 留 None 表示无 key 或 SDK 允许空 key（OpenAI / anthropic 当前都允许）
    ("ollama_client", None, None),
    ("anthropic_client", "anthropic", None),
    ("openai_client", "openai", None),
    ("gemini_client", "google.genai", "GEMINI_API_KEY"),
]


@pytest.mark.parametrize("module_name, sdk_module, key_attr", _BACKENDS)
def test_backend_client_module_imports_cleanly(
    module_name: str, sdk_module: str | None, key_attr: str | None,
):
    """每个 backend client 都能在自家 SDK 装好 (+ 必要时 API key 已配) 的前提下
    被 import. 这一步会触发 `_client = SDK_Client(...)` 的模块级实例化——SDK
    ABI 改了任何字段都会让 import 抛 AttributeError / TypeError 在这里立即可见.

    Workshop 默认 BACKEND=ollama，其它三家的 key 默认空字符串；gemini SDK 在
    `genai.Client(api_key="")` 时硬性 raise ValueError，所以 key 缺失即 skip
    （等用户切到该后端再触发这条 ABI smoke）."""
    if sdk_module:
        try:
            importlib.import_module(sdk_module)
        except ImportError:
            pytest.skip(f"{sdk_module} SDK not installed in this env")
    if key_attr:
        from agent_engine import config as ae_config
        if not getattr(ae_config, key_attr, ""):
            pytest.skip(f"{key_attr} not set; SDK rejects empty key at construction")
    module = importlib.import_module(f"agent_engine.{module_name}")
    assert hasattr(module, "chat"), (
        f"agent_engine.{module_name} 必须暴露 chat(...)——`agent.py` 按 BACKEND "
        f"挂接此符号；改名/删除会让 Engine.invoke 启动即崩"
    )


# ---------- CLI entrypoint --------------------------------------------

def test_cli_module_help_exits_zero():
    """`python -m agent_engine --help` 必出 help text 且 exit 0；任何 import 错
    （包括 4 backend client 中默认的 ollama 链路）都会在这里炸开."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_engine", "--help"],
        cwd=PLAY_DIR, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"`python -m agent_engine --help` failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "scenario" in result.stdout
    # README 文档里点名的 4 个 CLI flag 必须出现
    for flag in ("--no-stream", "--save-artifact", "--save-transcript", "--save-result-json"):
        assert flag in result.stdout, f"CLI dropped {flag}"


# ---------- ToolTracer ------------------------------------------------

def test_tool_tracer_record_emits_tool_call_entry_invisible(capsys):
    """`record` 写入的 entry: visible=False (memory 不投影), `ok` 由 is_error 决定,
    stderr 一行 🔧 emoji."""
    tr = ToolTracer()
    tr.record("A", "retrieve_docs", {"q": "x"}, '{"data": []}')
    events = tr.drain()
    assert len(events) == 1
    entry = events[0]
    assert isinstance(entry, ToolCallEntry)
    assert entry.caller == "A"
    assert entry.tool == "retrieve_docs"
    assert entry.arguments == {"q": "x"}
    assert entry.visible is False, (
        "tracer entries must be invisible — memory.py relies on this so "
        "tool_call doesn't leak back into LLM context"
    )
    assert entry.ok is True
    err = capsys.readouterr().err
    assert "🔧" in err and "[A] retrieve_docs" in err


def test_tool_tracer_record_marks_error_envelope_not_ok():
    tr = ToolTracer()
    tr.record("A", "x", {}, json.dumps({"error": "boom"}))
    entry = tr.drain()[0]
    assert entry.ok is False


def test_tool_tracer_drain_clears_buffer():
    tr = ToolTracer()
    tr.record("A", "x", {}, "{}")
    first = tr.drain()
    second = tr.drain()
    assert len(first) == 1
    assert second == []


# ---------- module surface health -------------------------------------

def test_engine_module_exposes_async_stubs():
    """`Engine.ainvoke / stream / astream` 必须 raise NotImplementedError——
    README §快速开始示意，evals/cli 默认不调；任何"悄悄实现一半"会破坏 contract."""
    from agent_engine import Engine
    eng = Engine.__new__(Engine)  # 跳过 __init__ 以免实例化 scenario
    # 异步 / 流式接口未实现是 README 文档化的当前状态
    with pytest.raises(NotImplementedError):
        eng.stream()
    with pytest.raises(NotImplementedError):
        import asyncio
        asyncio.run(eng.ainvoke())
