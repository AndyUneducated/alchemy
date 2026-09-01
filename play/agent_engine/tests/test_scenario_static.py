"""Scenario.expanded_turns static expansion tests (DECISIONS §13).

Locks three invariants:

  1. **Shape correct**: all four `who` forms (moderator / member / all / [name1, name2])
     expand in correct order with 1-based turn_idx
  2. **Field passthrough**: require_tool / max_retries / step_id / instruction come from step
  3. **Runtime parity**: on all live scenarios, `Scenario(...).expanded_turns()` length
     equals `Discussion._expanded` with identical agent order — static expansion must not drift from runtime
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_engine import ExpandedTurn, Scenario
from agent_engine.discussion import Discussion

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_DIR = REPO_ROOT / "play" / "agent_engine" / "scenarios"


def _write_scenario(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "scen.md"
    p.write_text(yaml_text, encoding="utf-8")
    return p


# ---------- who four forms ---------------------------------------------

def test_expanded_turns_who_list_explicit_names(tmp_path: Path):
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
          - {name: B, role: member, prompt: b}
        steps:
          - id: open
            who: [A, B]
            instruction: hi
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    expanded = s.expanded_turns()
    assert [(e.turn_idx, e.agent, e.step_id) for e in expanded] == [
        (1, "A", "open"),
        (2, "B", "open"),
    ]


def test_expanded_turns_who_member_role_filter(tmp_path: Path):
    """`who: member` filters by role, declaration order, excludes moderator."""
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: M, role: moderator, prompt: m}
          - {name: A, role: member, prompt: a}
          - {name: B, role: member, prompt: b}
        steps:
          - id: vote
            who: member
            instruction: vote
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    assert [e.agent for e in s.expanded_turns()] == ["A", "B"]


def test_expanded_turns_who_moderator_role_filter(tmp_path: Path):
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: M1, role: moderator, prompt: m1}
          - {name: A, role: member, prompt: a}
          - {name: M2, role: moderator, prompt: m2}
        steps:
          - id: open
            who: moderator
            instruction: open
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    assert [e.agent for e in s.expanded_turns()] == ["M1", "M2"]


def test_expanded_turns_who_all_includes_everyone(tmp_path: Path):
    """`who: all` all agents in declaration order (includes moderator)."""
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: M, role: moderator, prompt: m}
          - {name: A, role: member, prompt: a}
        steps:
          - id: any
            who: all
            instruction: speak
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    assert [e.agent for e in s.expanded_turns()] == ["M", "A"]


# ---------- field passthrough ------------------------------------------

def test_expanded_turns_carries_require_tool_and_max_retries(tmp_path: Path):
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - id: nudgeable
            who: [A]
            require_tool: cast_vote
            max_retries: 3
            instruction: vote please
          - id: free
            who: [A]
            instruction: chit chat
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    expanded = s.expanded_turns()
    assert expanded[0].require_tool == "cast_vote"
    assert expanded[0].max_retries == 3
    assert expanded[0].instruction == "vote please"
    # default require_tool + max_retries → require_tool=None, max_retries=0
    assert expanded[1].require_tool is None
    assert expanded[1].max_retries == 0


def test_expanded_turns_default_max_retries_is_one_when_require_tool_present(
    tmp_path: Path,
):
    """Matches Discussion._run_turn default `1 if require_tool else 0` behavior."""
    yaml_text = textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            require_tool: cast_vote
            instruction: vote
        ---
        topic
    """)
    s = Scenario.from_yaml(_write_scenario(tmp_path, yaml_text))
    assert s.expanded_turns()[0].max_retries == 1


def test_expanded_turn_is_frozen_dataclass():
    """ExpandedTurn is immutable (prevents consumer mutation)."""
    e = ExpandedTurn(
        turn_idx=1, agent="A", step_id=None, instruction="x",
        require_tool=None, max_retries=0,
    )
    with pytest.raises(Exception):
        e.turn_idx = 2  # type: ignore[misc]


# ---------- runtime parity (key invariant, DECISIONS §13) ----------------

@pytest.mark.parametrize("scenario_file", [
    "brainstorm.md",
    "debate.md",
    "roundtable.md",
    "panel.md",
    "code_review.md",
    "tool_chain.md",
    "example.md",
])
def test_expanded_turns_matches_discussion_expanded(scenario_file: str):
    """On all live scenarios, `Scenario.expanded_turns()` length + (agent.name, step_id)
    sequence byte-identical to `Discussion._expand_steps()` — locks static == runtime expansion.

    Safety net for reclaiming expansion from evals/agent_sft: future Discussion rule
    changes sync via _resolve_who_names; if shared usage is split, this test fails immediately.
    """
    path = SCENARIOS_DIR / scenario_file
    if not path.exists():
        pytest.skip(f"scenario {path} not present")
    scn = Scenario.from_yaml(str(path))
    asm = scn.assemble()
    discussion = Discussion(
        agents=asm.agents,
        agent_roles=asm.agent_roles,
        topic=asm.topic,
        steps=asm.steps,
        stream=False,
        artifact=asm.artifact,
        tracer=asm.tracer,
    )
    runtime_pairs = [
        (agent.name, step.get("id"))
        for agent, step in discussion._expanded
    ]
    static_expanded = scn.expanded_turns()
    static_pairs = [(e.agent, e.step_id) for e in static_expanded]

    assert len(static_pairs) == len(runtime_pairs), (
        f"{scenario_file}: static={len(static_pairs)} vs runtime={len(runtime_pairs)}"
    )
    assert static_pairs == runtime_pairs, scenario_file
    # turn_idx must also be 1-based monotonic
    for i, e in enumerate(static_expanded, 1):
        assert e.turn_idx == i, f"{scenario_file}: turn_idx[{i-1}] = {e.turn_idx}"
