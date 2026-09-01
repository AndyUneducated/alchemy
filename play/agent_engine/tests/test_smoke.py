"""Smoke: minimal availability assertions across SDK / CLI / Tracer.

Fallback when external SDK / OS / sub-project changes break this module — any
failure means agent_engine cannot run in the current environment:

  - Four backend client modules import independently (skip if SDK missing).
    Each instantiates SDK clients at module level; ABI breaks fail at import —
    earliest ABI regression alarm.
  - `python -m agent_engine --help` prints help with exit code 0 (CLI intact).
  - `ToolTracer.record / drain` stable: `visible=False`, `ok` from `is_error`,
    stderr one-line `🔧` emoji — memory projection / observability depend on this.
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
    # key_config_attr: if SDK validates key at client construction, config.* must be
    # non-empty to import; None = no key or SDK allows empty key (OpenAI SDK 2.x+
    # raises on `OpenAI(api_key="")`, so key skip needed).
    ("ollama_client", None, None),
    ("anthropic_client", "anthropic", None),
    ("openai_client", "openai", "OPENAI_API_KEY"),
    ("gemini_client", "google.genai", "GEMINI_API_KEY"),
]


@pytest.mark.parametrize("module_name, sdk_module, key_attr", _BACKENDS)
def test_backend_client_module_imports_cleanly(
    module_name: str, sdk_module: str | None, key_attr: str | None,
):
    """Each backend client imports when its SDK is installed (+ API key if required).
    Triggers module-level `_client = SDK_Client(...)` — ABI field changes surface
    as AttributeError / TypeError here.

    Default BACKEND=ollama; other backends default empty keys; gemini SDK raises on
    `genai.Client(api_key="")`, so missing key skips until that backend is used."""
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
        f"agent_engine.{module_name} must expose chat(...) — agent.py wires BACKEND "
        f"to this symbol; rename/delete breaks Engine.invoke at startup"
    )


# ---------- CLI entrypoint --------------------------------------------

def test_cli_module_help_exits_zero():
    """`python -m agent_engine --help` must print help and exit 0; import errors
    (including default ollama backend chain) surface here."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_engine", "--help"],
        cwd=PLAY_DIR, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"`python -m agent_engine --help` failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "scenario" in result.stdout
    # Four CLI flags named in README must appear
    for flag in ("--no-stream", "--save-artifact", "--save-transcript", "--save-result-json"):
        assert flag in result.stdout, f"CLI dropped {flag}"


# ---------- ToolTracer ------------------------------------------------

def test_tool_tracer_record_emits_tool_call_entry_invisible(capsys):
    """`record` entry: visible=False (not projected in memory), `ok` from is_error,
    stderr one-line 🔧 emoji."""
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
    """`Engine.ainvoke / stream / astream` must raise NotImplementedError —
    documented in README quick start; half-implemented stubs would break contract."""
    from agent_engine import Engine
    eng = Engine.__new__(Engine)  # skip __init__ to avoid scenario instantiation
    # async / stream APIs not implemented — documented current state
    with pytest.raises(NotImplementedError):
        eng.stream()
    with pytest.raises(NotImplementedError):
        import asyncio
        asyncio.run(eng.ainvoke())
