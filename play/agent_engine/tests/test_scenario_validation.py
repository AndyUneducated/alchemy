"""Scenario.from_yaml fail-fast 校验路径（DECISIONS §9）.

`scenario._err` 走 `sys.exit("Error: ...")`——任何校验路径变更（删 / 加 / 改字段
要求）都会让本测试失败. 这是"作者写错 scenario 就在启动时挂"的最后一道保险，
也是 evals / agent_sft 等消费者依赖的"上游不可能漏字段"前提.

锁住的校验维度：
  - agents：missing / 非 list / 缺 name / 缺 prompt / 缺 role / role 取值池 / 重名
  - steps：missing / 缺 who / 缺 instruction / require_tool 类型 / max_retries
  - who：未知 scalar / 空 list / 引用未声明 agent / role 不可达
  - memory：未知 type / window/summary 缺 max_recent / max_recent 非正
  - artifact：tool_owners 未知 tool / mode 非法 / sections 非 list

"无校验"的合法情形（artifact.enabled=false / 不写 memory / step 没 require_tool 等）
不在此处再次锁——已由 `test_scenario_static.py` 的现网 scenario 覆盖.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_engine import Scenario


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "scen.md"
    p.write_text(body, encoding="utf-8")
    return p


def _expect_exit(tmp_path: Path, body: str, expected: str) -> None:
    p = _write(tmp_path, body)
    with pytest.raises(SystemExit) as exc:
        Scenario.from_yaml(str(p))
    # SystemExit.code 是 sys.exit 的字符串参数（带 "Error: " 前缀）
    msg = str(exc.value)
    assert expected in msg, f"expected {expected!r} in {msg!r}"


# ---------- frontmatter ------------------------------------------------

def test_no_frontmatter_fails(tmp_path: Path):
    """无 `---` 分隔符 → fail-fast，不是默默把整文件当 body."""
    _expect_exit(
        tmp_path,
        "no frontmatter here\njust body\n",
        "no YAML frontmatter",
    )


def test_frontmatter_not_mapping_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            - not a mapping
            ---
            t
        """),
        "not a valid YAML mapping",
    )


# ---------- agents -----------------------------------------------------

def test_missing_agents_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "agents",
    )


def test_empty_agents_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents: []
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "agents",
    )


def test_agent_missing_name_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {role: member, prompt: p}
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "missing required string 'name'",
    )


def test_agent_missing_prompt_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member}
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "missing required string 'prompt'",
    )


def test_agent_unknown_role_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: bystander, prompt: p}
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "must be one of: moderator, member",
    )


def test_duplicate_agent_name_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: a1}
              - {name: A, role: member, prompt: a2}
            steps:
              - {who: all, instruction: x}
            ---
            t
        """),
        "duplicate name 'A'",
    )


# ---------- steps ------------------------------------------------------

def test_missing_steps_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            ---
            t
        """),
        "steps",
    )


def test_step_missing_who_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {instruction: x}
            ---
            t
        """),
        "missing required field 'who'",
    )


def test_step_missing_instruction_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A]}
            ---
            t
        """),
        "non-empty string 'instruction'",
    )


def test_step_blank_instruction_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: "   "}
            ---
            t
        """),
        "non-empty string 'instruction'",
    )


def test_step_non_string_require_tool_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x, require_tool: 123}
            ---
            t
        """),
        "'require_tool' must be a string",
    )


def test_step_negative_max_retries_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x, max_retries: -1}
            ---
            t
        """),
        "'max_retries' must be a non-negative integer",
    )


# ---------- who --------------------------------------------------------

def test_who_unknown_scalar_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: chair, instruction: x}
            ---
            t
        """),
        "not a valid scalar",
    )


def test_who_empty_list_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [], instruction: x}
            ---
            t
        """),
        "empty list",
    )


def test_who_references_unknown_name_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A, ghost], instruction: x}
            ---
            t
        """),
        "unknown agent name 'ghost'",
    )


def test_who_role_unreachable_fails(tmp_path: Path):
    """`who: moderator` 但全员是 member → fail-fast，间接拦住"声明了 moderator
    寻址却没 moderator"的 bug 类（DECISIONS §9 关键设计讨论）."""
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: moderator, instruction: x}
            ---
            t
        """),
        "matches 0 agents",
    )


# ---------- memory -----------------------------------------------------

def test_memory_unknown_type_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            memory: {type: weird}
            steps:
              - {who: [A], instruction: x}
            ---
            t
        """),
        "Must be one of: full, window, summary",
    )


def test_memory_window_requires_positive_max_recent(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            memory: {type: window, max_recent: 0}
            steps:
              - {who: [A], instruction: x}
            ---
            t
        """),
        "must be a positive integer",
    )


def test_agent_level_memory_validated(tmp_path: Path):
    """agent 自身的 memory 配置错也要 fail-fast，不只是 scenario 级."""
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - name: A
                role: member
                prompt: p
                memory: {type: full, max_recent: -1}  # max_recent 对 full 无影响，但 type=window/summary 类似分支
              - name: B
                role: member
                prompt: p
                memory: {type: summary, max_recent: 0}
            steps:
              - {who: [A], instruction: x}
            ---
            t
        """),
        "must be a positive integer",
    )


# ---------- artifact ---------------------------------------------------

def test_artifact_tool_owners_unknown_tool_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x}
            artifact:
              enabled: true
              tool_owners:
                bogus_tool: [A]
            ---
            t
        """),
        "is not an artifact tool",
    )


def test_artifact_initial_sections_unknown_mode_fails(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x}
            artifact:
              enabled: true
              initial_sections:
                - {name: 数据, mode: prepend}
            ---
            t
        """),
        "Must be one of: replace, append",
    )


def test_artifact_initial_sections_must_be_list(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x}
            artifact:
              enabled: true
              initial_sections:
                数据: replace
            ---
            t
        """),
        "must be a list",
    )


def test_artifact_enabled_must_be_bool(tmp_path: Path):
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - {name: A, role: member, prompt: p}
            steps:
              - {who: [A], instruction: x}
            artifact:
              enabled: "yes please"
            ---
            t
        """),
        "'enabled' must be a boolean",
    )
