"""Scenario.from_yaml fail-fast validation paths (DECISIONS §9).

`scenario._err` uses `sys.exit("Error: ...")` — any validation path change fails this test.
Last guard for "author typo fails at startup"; upstream cannot miss fields for evals/agent_sft.

Locked validation dimensions:
  - agents: missing / not list / missing name / prompt / role / role pool / duplicate names
  - steps: missing / missing who / instruction / require_tool type / max_retries
  - who: unknown scalar / empty list / unknown agent / unreachable role
  - memory: unknown type / window|summary missing max_recent / non-positive max_recent
  - artifact: unknown tool_owners tool / invalid mode / sections not list

Valid "no validation" cases covered by live scenarios in test_scenario_static.py.
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
    # SystemExit.code is sys.exit string arg (with "Error: " prefix)
    msg = str(exc.value)
    assert expected in msg, f"expected {expected!r} in {msg!r}"


# ---------- frontmatter ------------------------------------------------

def test_no_frontmatter_fails(tmp_path: Path):
    """No `---` delimiter → fail-fast, not silently treat whole file as body."""
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
    """`who: moderator` but all members → fail-fast; catches "moderator addressing without moderator" (DECISIONS §9)."""
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
    """Agent-level memory misconfig must fail-fast, not only scenario-level."""
    _expect_exit(
        tmp_path,
        textwrap.dedent("""\
            ---
            agents:
              - name: A
                role: member
                prompt: p
                memory: {type: full, max_recent: -1}  # max_recent irrelevant for full; similar branch for window/summary
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
