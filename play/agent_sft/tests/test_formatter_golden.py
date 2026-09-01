"""formatter.py — A byte-level golden snapshot of a complete chat sample.

test_formatter.py's 32 tests cover various fields/branches, but none nail the entire dict shape.
Once the downstream (MLX-LM/Ollama chat_template/agent_engine consumer) expects the wire
format drift (rename key/change nesting/adjust role order/arguments form), 32 single tests possible
Still passed but all train data was lost.

This test compares 1 deterministic triple → format_triple → complete `==` to inline golden.
Override the schema decisions of [DECISIONS §4](../DECISIONS.md) (`messages` + `tools` + top-level
`assistant.tool_calls` + `arguments` is a dict, changed from v1.5 to Qwen3.5 chat_template strict items).

Without relying on true scenarios/, inline a minimal YAML control tools[] output determinism."""

from __future__ import annotations

import json
import textwrap

from formatter import format_triple  # type: ignore[import-not-found]


GOLDEN_SCENARIO_YAML = textwrap.dedent("""\
---
agents:
  - name: A
    role: member
    prompt: |
      你是 A，按 instruction 调指定工具。
artifact:
  enabled: true
  initial_sections:
    - {name: 调研笔记, mode: append}
steps:
  - id: s1
    who: [A]
    require_tool: append_section
    max_retries: 0
    instruction: |
      append_section("调研笔记", "- 一句话要点") 把要点记入。
---
body
""")


def _golden_triple() -> dict:
    """Maps to step s1 first-fail + nudge-fire triple (schema aligned with extractor output)."""
    return {
        "run_id": 0,
        "scenario": "golden",
        "turn_idx": 1,
        "step_id": "s1",
        "agent": "A",
        "required_tool": "append_section",
        "failure_mode": "missed",
        "context": [],  # Empty context simplifies user content rendering
        "instruction": 'append_section("调研笔记", "- 一句话要点") 把要点记入。',
        "failed_response": "我先想想",
        "nudge": "你刚才没有调用 `append_section` 工具。请现在补上该调用以完成本轮任务。",
        "corrected_response": 'append_section("调研笔记", "- 一句话要点")',
    }


def _write_scenario(tmp_path) -> "Path":  # noqa: F821 — Path provided by pytest tmp_path
    p = tmp_path / "golden.md"
    p.write_text(GOLDEN_SCENARIO_YAML, encoding="utf-8")
    return p


EXPECTED_MESSAGES = [
    {
        "role": "system",
        "content": "你是 A，按 instruction 调指定工具。",
    },
    {
        "role": "user",
        "content": '现在请执行:\nappend_section("调研笔记", "- 一句话要点") 把要点记入。',
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "append_section",
# v1.5+: arguments is dict (v1 is JSON-string, Qwen3.5 chat_template
# v1.5+: arguments is dict (v1 is JSON-string, Qwen3.5 chat_template
# Use `tool_call.arguments|items` instead (strict mapping);
# Use `tool_call.arguments|items` instead (strict mapping);
# prop name = ArtifactStore.append_section schema:
# prop name = ArtifactStore.append_section schema:
# `name` + `entry` (not `section_name` / `content`, that is write_section)
# `name` + `entry` (not `section_name` / `content`, that is write_section)
                    "arguments": {"name": "调研笔记", "entry": "- 一句话要点"},
                },
            }
        ],
    },
]


EXPECTED_TOOL_APPEND_SECTION = {
    "type": "function",
    "function": {
        "name": "append_section",
        "description": (
            "Append an entry to a section, preserving existing content. "
            "Use this when multiple participants collaborate on the same section. "
            "Blocked if the section was declared as replace-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Section name to append to."},
                "entry": {
                    "type": "string",
                    "description": "Entry text; joined to existing content with a newline.",
                },
            },
            "required": ["name", "entry"],
        },
    },
}


def test_formatter_chat_sample_golden_snapshot(tmp_path):
    """Full messages dict equality — any key rename / nesting / role mismatch fails immediately.

    `tools[]` is not snapshotted wholesale (artifact injects 6 tools — too verbose); instead
    snapshot the required_tool entry dict-equal — still catches schema drift."""
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    assert sample is not None

# Top level 2 key fixed
# Top level 2 key fixed
    assert set(sample.keys()) == {"messages", "tools"}

# messages freeze the entire section
# messages freeze the entire section
    assert sample["messages"] == EXPECTED_MESSAGES

# tools: Use required_tool to make a schema snapshot individually
# tools: Use required_tool to make a schema snapshot individually
    by_name = {t["function"]["name"]: t for t in sample["tools"]}
    assert "append_section" in by_name, f"tools must expose required_tool; got {list(by_name)}"
    assert by_name["append_section"] == EXPECTED_TOOL_APPEND_SECTION, (
        "append_section schema drift — agent_engine.ArtifactStore.build_tool_defs 改了？"
    )


def test_formatter_arguments_is_dict_not_json_string(tmp_path):
    """v1.5+: arguments must be dict (not JSON-string) - Qwen3.5 chat_template
    Use `tool_call.arguments|items` to strictly require mapping; string trigger
    `TypeError: Can only get item pairs from a mapping.`. See [DECISIONS §11](../DECISIONS.md).
    The v1 era is JSON-string (compatible with Qwen2.5), supersede is dict at v1.5."""
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    args = sample["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict), f"arguments must be dict, got {type(args)}"


def test_formatter_assistant_content_is_empty_string_not_none(tmp_path):
    """assistant.content must be '' not None / missing — Qwen2.5 chat template
    renders None as literal `None`, polluting training data."""
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    asst = sample["messages"][-1]
    assert asst["content"] == ""
    assert asst["content"] is not None
