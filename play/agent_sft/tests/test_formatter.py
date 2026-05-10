"""formatter.py — Triple → MLX-LM messages 格式 + helper 单测.

覆盖：
  - 3-message 结构（system/user/assistant）schema 校验
  - system 含 agent prompt + tool 列表
  - user 含 step.instruction + 渲染过的 recent context
  - assistant content == triple.corrected_response
  - max_recent 截断行为
  - 空 context / 空 instruction 边界
"""

from __future__ import annotations

import textwrap

import pytest

from formatter import (  # type: ignore[import-not-found]
    DEFAULT_MAX_RECENT,
    _agent_prompt,
    _read_scenario_meta,
    _render_recent_context,
    _tool_summary,
    format_triple,
)


SCENARIO_YAML = textwrap.dedent("""\
---
tools:
  - name: retrieve_docs
    vdb_dir: /tmp/none
    top_k: 3
artifact:
  enabled: true
  initial_sections:
    - {name: notes, mode: append}
agents:
  - name: A
    role: member
    prompt: |
      你是 A。先 retrieve_docs 后 append_section。
steps:
  - id: s1
    who: [A]
    require_tool: retrieve_docs
    max_retries: 1
    instruction: |
      调用 retrieve_docs(query="foo") 拿背景。
---
body
""")


def write_scenario(tmp_path, yaml_text=SCENARIO_YAML):
    p = tmp_path / "scen.md"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def make_triple(**overrides):
    base = {
        "run_id": 0,
        "scenario": "scen",
        "turn_idx": 1,
        "step_id": "s1",
        "agent": "A",
        "required_tool": "retrieve_docs",
        "failure_mode": "missed",
        "context": [
            {"type": "topic", "content": "demo"},
            {"type": "turn", "content": "turn 1 of 1"},
        ],
        "instruction": "调用 retrieve_docs(query=\"foo\") 拿背景。",
        "failed_response": "我先想想",
        "nudge": "你刚才没有调用 `retrieve_docs` 工具。",
        "corrected_response": "OK 我调 retrieve_docs(query=\"foo\")",
    }
    base.update(overrides)
    return base


# --- format_triple --------------------------------------------------------

def test_format_returns_three_message_chat(tmp_path):
    scen = write_scenario(tmp_path)
    sample = format_triple(make_triple(), scen)
    assert "messages" in sample
    msgs = sample["messages"]
    assert len(msgs) == 3
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_system_contains_agent_prompt_and_tools(tmp_path):
    scen = write_scenario(tmp_path)
    sample = format_triple(make_triple(), scen)
    sys_msg = sample["messages"][0]["content"]
    assert "你是 A" in sys_msg
    assert "retrieve_docs" in sys_msg
    # artifact 启用 → 至少一个 artifact 工具名应在概要里
    assert "append_section" in sys_msg


def test_user_contains_instruction_and_recent_context(tmp_path):
    scen = write_scenario(tmp_path)
    sample = format_triple(make_triple(), scen)
    user_msg = sample["messages"][1]["content"]
    assert "retrieve_docs" in user_msg
    assert "现在请执行" in user_msg
    assert "最近对话" in user_msg
    assert "demo" in user_msg  # topic 出现在 recent
    assert "turn 1 of 1" in user_msg  # turn marker 出现在 recent


def test_assistant_is_corrected_response_verbatim(tmp_path):
    scen = write_scenario(tmp_path)
    triple = make_triple(corrected_response="EXACTLY THIS TEXT")
    sample = format_triple(triple, scen)
    assert sample["messages"][2]["content"] == "EXACTLY THIS TEXT"


def test_max_recent_truncates_long_context(tmp_path):
    scen = write_scenario(tmp_path)
    long_context = [
        {"type": "topic", "content": "T"},
        {"type": "turn", "content": "turn 1"},
        {"speaker": "X", "content": "old1"},
        {"speaker": "X", "content": "old2"},
        {"speaker": "X", "content": "recent1"},
        {"speaker": "X", "content": "recent2"},
    ]
    triple = make_triple(context=long_context)
    sample = format_triple(triple, scen, max_recent=2)
    user_msg = sample["messages"][1]["content"]
    assert "recent1" in user_msg
    assert "recent2" in user_msg
    assert "old1" not in user_msg
    assert "old2" not in user_msg


def test_empty_context_omits_recent_section(tmp_path):
    scen = write_scenario(tmp_path)
    triple = make_triple(context=[])
    sample = format_triple(triple, scen)
    user_msg = sample["messages"][1]["content"]
    assert "最近对话" not in user_msg
    assert "现在请执行" in user_msg


def test_empty_instruction_falls_back_to_generic(tmp_path):
    scen = write_scenario(tmp_path)
    triple = make_triple(instruction="")
    sample = format_triple(triple, scen)
    user_msg = sample["messages"][1]["content"]
    assert "现在请执行本轮任务" in user_msg


# --- helper units ---------------------------------------------------------

def test_agent_prompt_lookup(tmp_path):
    meta = _read_scenario_meta(write_scenario(tmp_path))
    assert "你是 A" in _agent_prompt(meta, "A")
    assert _agent_prompt(meta, "no_such_agent") == ""


def test_tool_summary_includes_scenario_tools_and_artifact(tmp_path):
    meta = _read_scenario_meta(write_scenario(tmp_path))
    summary = _tool_summary(meta)
    assert "retrieve_docs" in summary
    assert "append_section" in summary
    assert "cast_vote" in summary  # 启用 artifact → 整个 canonical 集都列上


def test_tool_summary_skips_artifact_when_disabled(tmp_path):
    yaml_text = textwrap.dedent("""\
---
tools:
  - name: retrieve_docs
    vdb_dir: /tmp
    top_k: 1
agents:
  - {name: A, role: member, prompt: a}
steps:
  - {id: s, who: [A], instruction: x}
---
""")
    meta = _read_scenario_meta(write_scenario(tmp_path, yaml_text=yaml_text))
    summary = _tool_summary(meta)
    assert summary == "retrieve_docs"


def test_render_recent_context_handles_all_entry_types():
    ctx = [
        {"type": "topic", "content": "T"},
        {"type": "turn", "content": "turn 1 of 2"},
        {"speaker": "A", "content": "hello"},
        {"type": "tool_call", "tool": "foo", "caller": "A"},
        {"type": "artifact_event", "tool": "append_section", "caller": "A"},
        "string entry should be skipped",  # 非 dict 防御
    ]
    out = _render_recent_context(ctx, max_recent=10)
    assert "【主题】T" in out
    assert "【turn 1 of 2】" in out
    assert "[A] hello" in out
    assert "A → foo" in out
    assert "A → append_section" in out


def test_render_recent_context_zero_max_recent_returns_empty():
    ctx = [{"type": "topic", "content": "x"}]
    assert _render_recent_context(ctx, max_recent=0) == ""


def test_default_max_recent_constant_matches_plan():
    """Plan §context 截取策略 says max_recent=6 (与 code_review.md 一致)."""
    assert DEFAULT_MAX_RECENT == 6
