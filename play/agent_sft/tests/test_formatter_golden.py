"""formatter.py — 1 个完整 chat sample 的字节级 golden snapshot.

test_formatter.py 的 32 个测试覆盖各字段 / 分支，但**没有一个钉死整 dict 的形状**。
一旦下游（MLX-LM / Ollama chat_template / agent_engine consumer）期望的 wire
format 漂移（重命名 key / 改嵌套 / 调 role 顺序 / arguments 形态），32 单测可能
仍全过但 train 数据全废.

本测把 1 个 deterministic triple → format_triple → 完整 `==` 比对 inline golden.
覆盖 [DECISIONS §4](../DECISIONS.md) 的 schema 决策 (`messages` + `tools` + 顶层
`assistant.tool_calls` + `arguments` 是 JSON-string).

不依赖真 scenarios/，inline 一份最小 YAML 控制 tools[] 输出确定性.
"""

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
    """对应 step s1 的 first-fail + nudge-fire triple（schema 与 extractor 输出对齐）."""
    return {
        "run_id": 0,
        "scenario": "golden",
        "turn_idx": 1,
        "step_id": "s1",
        "agent": "A",
        "required_tool": "append_section",
        "failure_mode": "missed",
        "context": [],  # 空 context 简化 user content 渲染
        "instruction": 'append_section("调研笔记", "- 一句话要点") 把要点记入。',
        "failed_response": "我先想想",
        "nudge": "你刚才没有调用 `append_section` 工具。请现在补上该调用以完成本轮任务。",
        "corrected_response": 'append_section("调研笔记", "- 一句话要点")',
    }


def _write_scenario(tmp_path) -> "Path":  # noqa: F821 — Path 由 pytest tmp_path 提供
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
                    # arguments JSON-string，prop 名 = ArtifactStore.append_section schema:
                    # `name` + `entry`（不是 `section_name` / `content`，那是 write_section）
                    "arguments": '{"name": "调研笔记", "entry": "- 一句话要点"}',
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
    """完整 messages dict 等价比对 —— 任意 key 改名 / 嵌套调整 / role 错位都立刻挂.

    `tools[]` 不做整数组 snapshot（artifact 自动注入 6 tool 太冗长），改为：
    抽 required_tool 对应的 entry 做 dict-equal snapshot —— 同样能 catch schema drift.
    """
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    assert sample is not None

    # 顶层 2 键固定
    assert set(sample.keys()) == {"messages", "tools"}

    # messages 整段冻结
    assert sample["messages"] == EXPECTED_MESSAGES

    # tools: 抽 required_tool 单条做 schema snapshot
    by_name = {t["function"]["name"]: t for t in sample["tools"]}
    assert "append_section" in by_name, f"tools must expose required_tool; got {list(by_name)}"
    assert by_name["append_section"] == EXPECTED_TOOL_APPEND_SECTION, (
        "append_section schema drift — agent_engine.ArtifactStore.build_tool_defs 改了？"
    )


def test_formatter_arguments_is_json_string_not_dict(tmp_path):
    """钉死 [DECISIONS §4](../DECISIONS.md)：arguments 必须是 JSON-encoded string，
    不是 dict—— OpenAI/Mistral 路径要 str；Qwen2.5 chat template 两者都 ok 但
    train data 形态约定走 str."""
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    args = sample["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str), f"arguments must be JSON-string, got {type(args)}"
    assert isinstance(json.loads(args), dict)


def test_formatter_assistant_content_is_empty_string_not_none(tmp_path):
    """assistant.content 必须是 '' 而非 None / 不存在 —— Qwen2.5 chat template
    碰 None 会渲染成字面量 `None`，污染训练数据."""
    scen = _write_scenario(tmp_path)
    sample = format_triple(_golden_triple(), scen)
    asst = sample["messages"][-1]
    assert asst["content"] == ""
    assert asst["content"] is not None
