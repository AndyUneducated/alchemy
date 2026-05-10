"""extractor.py — 6 行为边界 + helper unit 覆盖.

合成 transcript fixture（不依赖真 Ollama）覆盖 plan §Critical edge cases:
  - missed       → 1 triple, failure_mode='missed'
  - wrong_tool   → 1 triple, failure_mode='wrong_tool'
  - 2nd success  → 与 missed 共享路径（first_attempt_success_no_triple 验证负面）
  - 2nd fail drop → 0 triple
  - multi-nudge  → 1 triple，failure_mode 看 first attempt，corrected = 最终成功
  - no require_tool → 0 triple
"""

from __future__ import annotations

import textwrap

import pytest

from extractor import (  # type: ignore[import-not-found]
    NUDGE_TEMPLATE,
    Triple,
    _index_steps_by_turn,
    _parse_envelope_name,
    _split_turns_indexed,
    extract_triples,
)


# --- fixtures -------------------------------------------------------------

SCENARIO_YAML = textwrap.dedent("""\
---
agents:
  - name: A
    role: member
    prompt: 你是 A，按 instruction 调指定工具。
steps:
  - id: s1
    who: [A]
    require_tool: foo_tool
    max_retries: 1
    instruction: |
      调用 foo_tool 完成本步。
---
test body
""")


def write_scenario(tmp_path, yaml_text=SCENARIO_YAML):
    p = tmp_path / "scen.md"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def envelope(transcript):
    return {
        "transcript": transcript,
        "artifact": {},
        "warnings": [],
        "success": True,
    }


def turn_marker(idx, total=1):
    return {"type": "turn", "content": f"turn {idx} of {total}", "ts": 1.0}


def speaker(agent, content):
    return {"speaker": agent, "content": content, "ts": 1.0}


def tool_call(caller, tool, ok=True):
    return {"type": "tool_call", "caller": caller, "tool": tool, "ok": ok}


# --- behavior tests -------------------------------------------------------

def test_missed_first_attempt_then_success(tmp_path):
    scen = write_scenario(tmp_path)
    transcript = [
        {"type": "topic", "content": "test topic", "ts": 0.0},
        turn_marker(1),
        speaker("A", "我先想想"),  # missed: no tool call
        speaker("A", "现在调 foo_tool"),
        tool_call("A", "foo_tool"),
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    assert len(triples) == 1
    t = triples[0]
    assert isinstance(t, Triple)
    assert t.failure_mode == "missed"
    assert t.failed_response == "我先想想"
    assert t.corrected_response == "现在调 foo_tool"
    assert t.required_tool == "foo_tool"
    assert t.nudge == NUDGE_TEMPLATE.format(tool="foo_tool")
    assert t.run_id == 0
    assert t.turn_idx == 1
    assert t.step_id == "s1"
    assert t.scenario == "scen"
    assert t.instruction.startswith("调用 foo_tool")
    # context 包含 topic + turn marker；不包含 first speaker entry
    assert any(e.get("type") == "topic" for e in t.context)
    assert any(e.get("type") == "turn" for e in t.context)
    assert all(e.get("speaker") != "A" for e in t.context)


def test_wrong_tool_first_attempt(tmp_path):
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "调点别的"),
        tool_call("A", "other_tool"),
        speaker("A", "好吧 foo_tool"),
        tool_call("A", "foo_tool"),
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=2, scenario_name="scen")
    assert len(triples) == 1
    assert triples[0].failure_mode == "wrong_tool"
    assert triples[0].failed_response == "调点别的"
    assert triples[0].corrected_response == "好吧 foo_tool"


def test_first_attempt_success_drops_triple(tmp_path):
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "马上调 foo_tool"),
        tool_call("A", "foo_tool"),
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    assert triples == []


def test_all_attempts_fail_drops_triple(tmp_path):
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "想想"),
        speaker("A", "再想想"),  # max_retries=1 → 2 attempts，都没调 foo_tool
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    assert triples == []


def test_multi_nudge_picks_first_failed_and_eventual_success(tmp_path):
    yaml_text = SCENARIO_YAML.replace("max_retries: 1", "max_retries: 2")
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = [
        turn_marker(1),
        speaker("A", "missed"),
        speaker("A", "wrong"),
        tool_call("A", "other_tool"),
        speaker("A", "got it"),
        tool_call("A", "foo_tool"),
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    assert len(triples) == 1
    t = triples[0]
    assert t.failure_mode == "missed"  # 看 first attempt
    assert t.failed_response == "missed"
    assert t.corrected_response == "got it"


def test_no_require_tool_scenario_returns_empty(tmp_path):
    yaml_text = textwrap.dedent("""\
---
agents:
  - name: A
    role: member
    prompt: just chat
steps:
  - id: chat
    who: [A]
    instruction: |
      闲聊一句即可
---
""")
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = [turn_marker(1), speaker("A", "hi")]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    assert triples == []


# --- robustness tests -----------------------------------------------------

def test_segment_count_less_than_expected_skips_turn(tmp_path):
    """subprocess 中途崩 → segments 不够 expected_turns → 该 turn skip 不报错."""
    yaml_text = textwrap.dedent("""\
---
agents:
  - name: A
    role: member
    prompt: a
steps:
  - id: s1
    who: [A]
    require_tool: foo_tool
    max_retries: 1
    instruction: t1
  - id: s2
    who: [A]
    require_tool: foo_tool
    max_retries: 1
    instruction: t2
---
""")
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    # 只跑了 turn 1 就崩了；turn 2 无 segment
    transcript = [
        turn_marker(1),
        speaker("A", "miss"),
        speaker("A", "ok"),
        tool_call("A", "foo_tool"),
    ]
    triples = extract_triples(envelope(transcript), scen, run_id=0, scenario_name="scen")
    # turn 1 产 1 triple，turn 2 因 segment 不存在被跳
    assert len(triples) == 1
    assert triples[0].turn_idx == 1


# --- helper unit tests ----------------------------------------------------

def test_split_turns_indexed_returns_global_offsets():
    transcript = [
        {"type": "topic", "content": "x"},
        turn_marker(1),
        speaker("A", "a"),
        speaker("A", "b"),
        turn_marker(2),
        speaker("B", "c"),
    ]
    segs = _split_turns_indexed(transcript)
    assert len(segs) == 2
    assert segs[0][0] == 2  # 第一段第 1 个 entry 是 transcript[2]
    assert len(segs[0][1]) == 2
    assert segs[1][0] == 5  # 第二段从 transcript[5] 起
    assert len(segs[1][1]) == 1


def test_index_steps_by_turn_expands_who(tmp_path):
    yaml_text = textwrap.dedent("""\
---
agents:
  - {name: A, role: member, prompt: a}
  - {name: B, role: member, prompt: b}
steps:
  - id: open
    who: [A, B]
    instruction: hi everyone
  - id: vote
    who: member
    require_tool: cast_vote
    instruction: vote please
---
""")
    p = tmp_path / "s.md"
    p.write_text(yaml_text, encoding="utf-8")
    out = _index_steps_by_turn(p)
    # Step 1 expands to [A, B] → turns 1, 2; step 2 with who=member expands to [A, B] → 3, 4
    assert set(out.keys()) == {1, 2, 3, 4}
    assert out[1]["id"] == "open"
    assert out[3]["id"] == "vote"
    assert out[3]["require_tool"] == "cast_vote"


def test_parse_envelope_name():
    assert _parse_envelope_name("tool_chain-r3") == ("tool_chain", 3)
    assert _parse_envelope_name("code_review-r12") == ("code_review", 12)
    with pytest.raises(ValueError):
        _parse_envelope_name("no_run_id")
    with pytest.raises(ValueError):
        _parse_envelope_name("scen-rabc")


def test_extract_returns_empty_on_empty_transcript(tmp_path):
    scen = write_scenario(tmp_path)
    triples = extract_triples(envelope([]), scen, run_id=0, scenario_name="scen")
    assert triples == []
