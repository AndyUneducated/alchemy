"""synthesize.py — Approach B: true failure + synthesized correct triple generation test.

Coverage:
  - `_extract_call_template`: template capture (with paren / without paren / cross-line / nested paren / Chinese)
  - `synthesize_corrected_response`: template path + fallback path + empty instruction
  - `envelope_to_synthetic_triples`: per-fire one / first time success skip / wrong_args skip /
    no require_tool / empty transcript"""

from __future__ import annotations

import dataclasses
import textwrap

from synthesize import (  # type: ignore[import-not-found]
    _extract_call_template,
    envelope_to_synthetic_triples,
    synthesize_corrected_response,
)
from agent_engine import (  # type: ignore[import-not-found]
    SpeakerEntry,
    ToolCallEntry,
    TurnEntry,
)


# --- shared fixtures ------------------------------------------------------

SCENARIO_YAML = textwrap.dedent("""\
---
agents:
  - name: A
    role: member
    prompt: 你是 A
steps:
  - id: s1
    who: [A]
    require_tool: foo_tool
    max_retries: 1
    instruction: |
      foo_tool("arg1", "arg2") 完成本步任务。
---
body
""")


def write_scenario(tmp_path, yaml_text=SCENARIO_YAML):
    p = tmp_path / "scen.md"
    p.write_text(yaml_text, encoding="utf-8")
    return p


def envelope(transcript_entries):
    return {
        "transcript": [dataclasses.asdict(e) for e in transcript_entries],
        "artifact": {},
        "warnings": [],
        "success": True,
        "usage": [],
    }


def turn_marker(idx, total=1):
    return TurnEntry(content=f"turn {idx} of {total}", ts=1.0)


def speaker(agent, content):
    return SpeakerEntry(speaker=agent, content=content, ts=1.0)


def tool_call(caller, tool, ok=True):
    return ToolCallEntry(caller=caller, tool=tool, ok=ok, ts=1.0)


# --- _extract_call_template ----------------------------------------------

def test_extract_call_template_simple():
    s = 'foo_tool("a", "b") 干这个'
    assert _extract_call_template(s, "foo_tool") == 'foo_tool("a", "b")'


def test_extract_call_template_no_template():
    s = "调用 foo_tool 完成任务，不写Parameter"
    assert _extract_call_template(s, "foo_tool") is None


def test_extract_call_template_multiline_args():
    s = textwrap.dedent("""
        请按下面的方式调用：
        append_section("review_a",
                       "- 评审结论一句话")
        然后继续。
    """)
    out = _extract_call_template(s, "append_section")
    assert out is not None
    assert out.startswith('append_section("review_a"')
    assert out.endswith(')')


def test_extract_call_template_chinese_quotes_in_args():
    s = '请 cast_vote(vote_id="v1", option="合入" 或 "退回", rationale="一句话")'
    out = _extract_call_template(s, "cast_vote")
    assert out is not None
    assert out.startswith("cast_vote(")
    assert out.endswith(")")


def test_extract_call_template_first_match_only():
    """When two matches appear, take the first (preserve instruction intent)."""
    s = 'foo_tool("a") 然后 foo_tool("b")'
    assert _extract_call_template(s, "foo_tool") == 'foo_tool("a")'


def test_extract_call_template_unbalanced_paren_returns_none():
    s = "foo_tool(unclosed"
    assert _extract_call_template(s, "foo_tool") is None


def test_extract_call_template_word_boundary():
    """foo_tool_x must not match as foo_tool."""
    s = 'foo_tool_x("a")'
    assert _extract_call_template(s, "foo_tool") is None


# --- synthesize_corrected_response ---------------------------------------

def test_synthesize_uses_template_when_present():
    instr = 'foo_tool("a", "b") 干这个'
    out = synthesize_corrected_response(instr, "foo_tool")
    assert "foo_tool" in out
    assert 'foo_tool("a", "b")' in out
    assert "好的" in out  # wrapper phrase


def test_synthesize_falls_back_when_no_template():
    instr = "调用 foo_tool 查询点东西，30 字一句话报告。"
    out = synthesize_corrected_response(instr, "foo_tool")
    assert "foo_tool" in out
    assert "查询点东西" in out  # full instruction in corrected response as fallback
    assert "完成本步" in out


def test_synthesize_handles_empty_instruction():
    out = synthesize_corrected_response("", "foo_tool")
    assert out == "好的，我现在调用 `foo_tool`。"


def test_synthesize_is_deterministic():
    a = synthesize_corrected_response("foo_tool('x')", "foo_tool")
    b = synthesize_corrected_response("foo_tool('x')", "foo_tool")
    assert a == b


# --- envelope_to_synthetic_triples ---------------------------------------

def test_each_fire_produces_one_triple(tmp_path):
    """Key difference vs extractor: first attempt fails → one triple immediately, no later success needed."""
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "我先想想"),  # missed
        speaker("A", "再想想"),    # Still not adjusted (the engine gives up after max_retries=1)
    ]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 1
    t = triples[0]
    assert t.failure_mode == "missed"
    assert t.failed_response == "我先想想"
    assert "foo_tool" in t.corrected_response
    assert 'foo_tool("arg1", "arg2")' in t.corrected_response  #Used instruction template


def test_first_attempt_success_no_triple(tmp_path):
    """Get it right the first time = no fire = no triple."""
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "马上调"),
        tool_call("A", "foo_tool"),
    ]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert triples == []


def test_no_require_tool_returns_empty(tmp_path):
    yaml_text = textwrap.dedent("""\
---
agents:
  - {name: A, role: member, prompt: a}
steps:
  - {id: chat, who: [A], instruction: just chat}
---
""")
    scenario = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = [turn_marker(1), speaker("A", "hi")]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert triples == []


def test_empty_transcript_returns_empty(tmp_path):
    scenario = write_scenario(tmp_path)
    triples = envelope_to_synthetic_triples(envelope([]), scenario, run_id=0)
    assert triples == []


def test_failure_mode_classification_works(tmp_path):
    """failure_mode is still judged according to first attempt, which is consistent with extractor."""
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "调点别的"),
        tool_call("A", "other_tool"),  # wrong_tool first time
    ]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 1
    assert triples[0].failure_mode == "wrong_tool"
    assert triples[0].failed_response == "调点别的"


def test_yield_higher_than_extractor(tmp_path):
    """All 5 fire turns failed: extractor came out with 0, synthesizer came out with 5."""
    yaml_text = textwrap.dedent("""\
---
agents:
  - {name: A, role: member, prompt: a}
steps:
  - {id: t1, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("x")'}
  - {id: t2, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("y")'}
  - {id: t3, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("z")'}
  - {id: t4, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("w")'}
  - {id: t5, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("u")'}
---
""")
    scenario = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = []
    for i in range(1, 6):
        transcript.append(turn_marker(i, total=5))
        transcript.append(speaker("A", f"missed {i}"))
        transcript.append(speaker("A", f"still missed {i}"))
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assertlen(triples) == 5
    assert all(t.required_tool == "foo" for t in triples)
    assert all(t.failure_mode == "missed" for t in triples)


def test_segment_count_less_than_expected_skips_turn(tmp_path):
    """Subprocess crashes → Turns with missing segments are silently skipped."""
    yaml_text = textwrap.dedent("""\
---
agents:
  - {name: A, role: member, prompt: a}
steps:
  - {id: s1, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("a")'}
  - {id: s2, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("b")'}
---
""")
    scenario = write_scenario(tmp_path, yaml_text=yaml_text)
    # run only turn 1
    # run only turn 1
    transcript = [turn_marker(1), speaker("A", "miss")]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assertlen(triples) == 1
    assert triples[0].turn_idx == 1"""
