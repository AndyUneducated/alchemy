"""formatter.py — Triple → MLX-LM `tools` schema (DECISIONS §4) 单测.

覆盖：
  - 顶层 `messages` + `tools` 字段；3-message (system/user/assistant) 结构
  - assistant.content == "" + tool_calls 含 OpenAI 形态
  - arguments 是 JSON-string，反解为含正确 prop 名的 dict
  - tools 数组复用 agent_engine `_resolve_tool_defs` + `ArtifactStore.build_tool_defs`
    （per-agent role filter）
  - drop 规则：no_template (retrieve_docs fallback wrapper) + unparseable args
  - tolerant fallback 救回 cast_vote 中文 `或` 分隔的 option（覆盖 Phase 2 真数据）
  - max_recent / 空 context / 空 instruction 边界
"""

from __future__ import annotations

import json
import textwrap

import pytest

from formatter import (  # type: ignore[import-not-found]
    DEFAULT_MAX_RECENT,
    _agent_prompt,
    _call_template_to_args_dict,
    _extract_first_literal,
    _find_tool_schema,
    _load_tool_defs,
    _read_scenario_meta,
    _render_recent_context,
    _split_top_level_commas,
    _strict_parse,
    _tolerant_parse,
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
  tool_owners:
    propose_vote: moderator
    finalize_artifact: moderator
agents:
  - name: A
    role: member
    prompt: |
      你是 A。先 retrieve_docs 后 append_section。
  - name: M
    role: moderator
    prompt: |
      你是 M moderator。
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


# --- format_triple top-level shape ----------------------------------------

def test_format_returns_messages_and_tools_top_level(tmp_path):
    scen = write_scenario(tmp_path)
    sample = format_triple(make_triple(), scen)
    assert sample is not None
    assert set(sample.keys()) == {"messages", "tools"}


def test_messages_have_three_roles_in_order(tmp_path):
    scen = write_scenario(tmp_path)
    msgs = format_triple(make_triple(), scen)["messages"]
    assert len(msgs) == 3
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_system_is_agent_prompt_only(tmp_path):
    """Qwen2.5 chat template 在 tools 存在时会自动渲染 # Tools 块到 system；
    我们的 system content 只放 agent.prompt（不预渲染 tools 文本）."""
    scen = write_scenario(tmp_path)
    sys_msg = format_triple(make_triple(), scen)["messages"][0]["content"]
    assert "你是 A" in sys_msg
    # 不应预渲染工具列表
    assert "可用工具" not in sys_msg
    assert "# Tools" not in sys_msg


def test_user_contains_instruction_and_recent_context(tmp_path):
    scen = write_scenario(tmp_path)
    user_msg = format_triple(make_triple(), scen)["messages"][1]["content"]
    assert "现在请执行" in user_msg
    assert "retrieve_docs" in user_msg
    assert "最近对话" in user_msg
    assert "demo" in user_msg
    assert "turn 1 of 1" in user_msg


def test_assistant_has_empty_content_plus_tool_calls(tmp_path):
    scen = write_scenario(tmp_path)
    asst = format_triple(make_triple(), scen)["messages"][2]
    assert asst["role"] == "assistant"
    assert asst["content"] == ""
    assert isinstance(asst["tool_calls"], list)
    assert len(asst["tool_calls"]) == 1


def test_tool_call_uses_openai_function_envelope(tmp_path):
    scen = write_scenario(tmp_path)
    tc = format_triple(make_triple(), scen)["messages"][2]["tool_calls"][0]
    assert tc["type"] == "function"
    assert "id" in tc
    assert tc["function"]["name"] == "retrieve_docs"


def test_tool_call_arguments_is_json_string_with_correct_keys(tmp_path):
    scen = write_scenario(tmp_path)
    tc = format_triple(make_triple(), scen)["messages"][2]["tool_calls"][0]
    args_str = tc["function"]["arguments"]
    assert isinstance(args_str, str)  # OpenAI/Mistral 习惯 — JSON-string
    parsed = json.loads(args_str)
    assert isinstance(parsed, dict)
    assert parsed["query"] == "foo"


# --- tools field ----------------------------------------------------------

def test_tools_field_is_list_of_function_envelopes(tmp_path):
    scen = write_scenario(tmp_path)
    tools = format_triple(make_triple(), scen)["tools"]
    assert isinstance(tools, list)
    assert tools, "tools 不应为空（scenario 有 retrieve_docs + 启用 artifact）"
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]


def test_tools_includes_scenario_tools_and_member_artifact_subset(tmp_path):
    """member agent 只看到 scenario.tools + 非 moderator-only artifact 工具."""
    scen = write_scenario(tmp_path)
    tools = format_triple(make_triple(agent="A"), scen)["tools"]
    names = {t["function"]["name"] for t in tools}
    # scenario 工具
    assert "retrieve_docs" in names
    # 共享 artifact 工具
    assert "append_section" in names
    assert "write_section" in names
    assert "cast_vote" in names
    # moderator-only 工具被过滤
    assert "propose_vote" not in names
    assert "finalize_artifact" not in names


def test_tools_includes_moderator_artifact_tools_for_moderator_agent(tmp_path):
    scen = write_scenario(tmp_path)
    triple = make_triple(
        agent="M", required_tool="propose_vote",
        instruction="propose_vote(question=\"q\", options=[\"a\", \"b\"])",
    )
    sample = format_triple(triple, scen)
    assert sample is not None
    names = {t["function"]["name"] for t in sample["tools"]}
    assert "propose_vote" in names
    assert "finalize_artifact" in names


# --- args extraction ------------------------------------------------------

ARTIFACT_APPEND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "append_section",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "entry": {"type": "string"},
            },
            "required": ["name", "entry"],
        },
    },
}

CAST_VOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cast_vote",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {
                "vote_id": {"type": "string"},
                "option": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["vote_id", "option"],
        },
    },
}


def test_strict_parse_positional_to_named_props():
    args = _call_template_to_args_dict(
        'append_section("review_a", "- 一句话")',
        "append_section",
        ARTIFACT_APPEND_SCHEMA,
    )
    assert args == {"name": "review_a", "entry": "- 一句话"}


def test_strict_parse_keyword_args():
    args = _call_template_to_args_dict(
        'append_section(name="review_b", entry="text")',
        "append_section",
        ARTIFACT_APPEND_SCHEMA,
    )
    assert args == {"name": "review_b", "entry": "text"}


def test_tolerant_parse_recovers_invalid_chinese_or_separator():
    """Phase 2 真数据：cast_vote(vote_id="v1", option="合入" 或 "退回", ...)
    含中文 `或`，ast 解析失败；fallback 应抽到第一个字符串字面量."""
    args = _call_template_to_args_dict(
        'cast_vote(vote_id="v1", option="合入" 或 "退回", rationale="一句话理由")',
        "cast_vote",
        CAST_VOTE_SCHEMA,
    )
    assert args["vote_id"] == "v1"
    assert args["option"] == "合入"
    assert args["rationale"] == "一句话理由"


def test_args_dict_filters_unknown_keys():
    """instruction 偶有 schema 之外的 key — 防御性丢弃."""
    args = _call_template_to_args_dict(
        'append_section(name="x", entry="y", extra="ignored")',
        "append_section",
        ARTIFACT_APPEND_SCHEMA,
    )
    assert args == {"name": "x", "entry": "y"}


def test_args_dict_returns_none_when_no_known_keys():
    args = _call_template_to_args_dict(
        'append_section(extra="ignored")',
        "append_section",
        ARTIFACT_APPEND_SCHEMA,
    )
    assert args is None


def test_args_dict_fills_missing_required_with_empty_string():
    args = _call_template_to_args_dict(
        'append_section(name="x")',
        "append_section",
        ARTIFACT_APPEND_SCHEMA,
    )
    assert args == {"name": "x", "entry": ""}


def test_split_top_level_commas_handles_nested_brackets_and_quotes():
    parts = _split_top_level_commas('a, "b, c", [1, 2], (3, 4)')
    assert parts == ['a', '"b, c"', '[1, 2]', '(3, 4)']


def test_extract_first_literal_string_fallback():
    assert _extract_first_literal('"hello"') == "hello"
    assert _extract_first_literal("'world'") == "world"
    assert _extract_first_literal('"合入" 或 "退回"') == "合入"
    assert _extract_first_literal('["a", "b"]') == ["a", "b"]
    assert _extract_first_literal("garbage_no_literal") == ""


def test_strict_parse_returns_none_on_invalid_python():
    assert _strict_parse('cast_vote(option="x" 或 "y")', "cast_vote",
                          ["vote_id", "option", "rationale"]) is None


def test_tolerant_parse_returns_dict_on_invalid_python():
    out = _tolerant_parse('cast_vote(option="x" 或 "y")', "cast_vote",
                          ["vote_id", "option", "rationale"])
    assert out is not None
    assert out["option"] == "x"


# --- drop rules -----------------------------------------------------------

def test_format_returns_none_when_no_call_template(tmp_path):
    """retrieve_docs 类 instruction 没有字面 retrieve_docs(...) — drop."""
    scen = write_scenario(tmp_path)
    triple = make_triple(
        instruction="调用 retrieve_docs 查询「项目代号」并总结要点。",
    )
    assert format_triple(triple, scen) is None


def test_format_returns_none_for_unknown_required_tool(tmp_path):
    """required_tool 不在 agent 工具清单 — 防御性 drop."""
    scen = write_scenario(tmp_path)
    triple = make_triple(
        required_tool="bogus_tool",
        instruction="bogus_tool(x=\"y\")",
    )
    assert format_triple(triple, scen) is None


def test_cli_summary_counts_drops_correctly(tmp_path, capsys):
    """CLI main() 末尾 print 三类计数：keep / drop_no_template / drop_unparseable."""
    from formatter import main  # type: ignore[import-not-found]

    scen = write_scenario(tmp_path)
    triples_path = tmp_path / "triples.jsonl"
    out_path = tmp_path / "out.jsonl"

    items = [
        make_triple(),  # keep
        make_triple(instruction="调用 retrieve_docs 查询资料"),  # no_template
        make_triple(  # keep (kw)
            instruction='retrieve_docs(query="bar")'
        ),
    ]
    with triples_path.open("w", encoding="utf-8") as f:
        for t in items:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    rc = main([
        "--in", str(triples_path),
        "--out", str(out_path),
        "--scenarios-root", str(tmp_path),  # scen.md 与 triples.scenario="scen" 对齐
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "kept: 2" in captured
    assert "dropped 1 (no call template" in captured

    with out_path.open("r", encoding="utf-8") as f:
        out_samples = [json.loads(l) for l in f]
    assert len(out_samples) == 2


# --- helpers --------------------------------------------------------------

def test_load_tool_defs_member_excludes_moderator_only(tmp_path):
    meta = _read_scenario_meta(write_scenario(tmp_path))
    defs = _load_tool_defs(meta, "A")
    names = {d["function"]["name"] for d in defs}
    assert "retrieve_docs" in names
    assert "append_section" in names
    assert "propose_vote" not in names
    assert "finalize_artifact" not in names


def test_load_tool_defs_handles_disabled_artifact(tmp_path):
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
    defs = _load_tool_defs(meta, "A")
    assert [d["function"]["name"] for d in defs] == ["retrieve_docs"]


def test_find_tool_schema_returns_none_for_missing():
    assert _find_tool_schema([], "x") is None
    schema = {"type": "function", "function": {"name": "x", "parameters": {}}}
    assert _find_tool_schema([schema], "x") is schema
    assert _find_tool_schema([schema], "y") is None


def test_agent_prompt_lookup(tmp_path):
    meta = _read_scenario_meta(write_scenario(tmp_path))
    assert "你是 A" in _agent_prompt(meta, "A")
    assert _agent_prompt(meta, "no_such_agent") == ""


def test_render_recent_context_handles_all_entry_types():
    ctx = [
        {"type": "topic", "content": "T"},
        {"type": "turn", "content": "turn 1 of 2"},
        {"speaker": "A", "content": "hello"},
        {"type": "tool_call", "tool": "foo", "caller": "A"},
        {"type": "artifact_event", "tool": "append_section", "caller": "A"},
        "string entry should be skipped",
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
    user_msg = format_triple(triple, scen, max_recent=2)["messages"][1]["content"]
    assert "recent1" in user_msg
    assert "recent2" in user_msg
    assert "old1" not in user_msg
    assert "old2" not in user_msg


def test_empty_context_omits_recent_section(tmp_path):
    scen = write_scenario(tmp_path)
    triple = make_triple(context=[])
    user_msg = format_triple(triple, scen)["messages"][1]["content"]
    assert "最近对话" not in user_msg
    assert "现在请执行" in user_msg


def test_default_max_recent_constant_matches_plan():
    """Plan §context 截取策略 says max_recent=6 (与 code_review.md 一致)."""
    assert DEFAULT_MAX_RECENT == 6
