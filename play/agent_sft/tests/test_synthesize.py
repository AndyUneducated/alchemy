"""synthesize.py — Approach B: 真失败 + 合成正确 三元组生成测试.

覆盖：
  - `_extract_call_template`: 模板抓取（带 paren / 不带 paren / 跨行 / 嵌套 paren / 中文）
  - `synthesize_corrected_response`: 模板路径 + fallback 路径 + 空 instruction
  - `envelope_to_synthetic_triples`: per-fire 一条 / 第一次成功 skip / wrong_args skip /
    no require_tool / 空 transcript
"""

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
    s = "调用 foo_tool 完成任务，不写参数"
    assert _extract_call_template(s, "foo_tool") is None


def test_extract_call_template_multiline_args():
    s = textwrap.dedent("""\
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
    """有两次出现时取第一次（保 instruction 主旨）."""
    s = 'foo_tool("a") 然后 foo_tool("b")'
    assert _extract_call_template(s, "foo_tool") == 'foo_tool("a")'


def test_extract_call_template_unbalanced_paren_returns_none():
    s = "foo_tool(unclosed"
    assert _extract_call_template(s, "foo_tool") is None


def test_extract_call_template_word_boundary():
    """foo_tool_x 不应该被当成 foo_tool 的匹配."""
    s = 'foo_tool_x("a")'
    assert _extract_call_template(s, "foo_tool") is None


# --- synthesize_corrected_response ---------------------------------------

def test_synthesize_uses_template_when_present():
    instr = 'foo_tool("a", "b") 干这个'
    out = synthesize_corrected_response(instr, "foo_tool")
    assert "foo_tool" in out
    assert 'foo_tool("a", "b")' in out
    assert "好的" in out  # 包装语


def test_synthesize_falls_back_when_no_template():
    instr = "调用 foo_tool 查询点东西，30 字一句话报告。"
    out = synthesize_corrected_response(instr, "foo_tool")
    assert "foo_tool" in out
    assert "查询点东西" in out  # 完整 instruction 入正确响应作 fallback
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
    """关键差异 vs extractor: first attempt 失败 → 立刻 1 triple，无需后续 success."""
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "我先想想"),  # missed
        speaker("A", "再想想"),    # 仍未调对（max_retries=1 后引擎放弃）
    ]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 1
    t = triples[0]
    assert t.failure_mode == "missed"
    assert t.failed_response == "我先想想"
    assert "foo_tool" in t.corrected_response
    assert 'foo_tool("arg1", "arg2")' in t.corrected_response  # 用了 instruction 模板


def test_first_attempt_success_no_triple(tmp_path):
    """第一次就调对 = 没 fire = 无 triple."""
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
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = [turn_marker(1), speaker("A", "hi")]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert triples == []


def test_empty_transcript_returns_empty(tmp_path):
    scen = write_scenario(tmp_path)
    triples = envelope_to_synthetic_triples(envelope([]), scen, run_id=0)
    assert triples == []


def test_failure_mode_classification_works(tmp_path):
    """failure_mode 仍按 first attempt 判，与 extractor 一致."""
    scen = write_scenario(tmp_path)
    transcript = [
        turn_marker(1),
        speaker("A", "调点别的"),
        tool_call("A", "other_tool"),  # wrong_tool 第一次
    ]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 1
    assert triples[0].failure_mode == "wrong_tool"
    assert triples[0].failed_response == "调点别的"


def test_yield_higher_than_extractor(tmp_path):
    """5 个 fire turns 全部全程失败：extractor 出 0，synthesize 出 5."""
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
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    transcript = []
    for i in range(1, 6):
        transcript.append(turn_marker(i, total=5))
        transcript.append(speaker("A", f"missed {i}"))
        transcript.append(speaker("A", f"still missed {i}"))
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 5
    assert all(t.required_tool == "foo" for t in triples)
    assert all(t.failure_mode == "missed" for t in triples)


def test_segment_count_less_than_expected_skips_turn(tmp_path):
    """subprocess 中途崩 → 缺 segments 的 turn 被静默跳过."""
    yaml_text = textwrap.dedent("""\
---
agents:
  - {name: A, role: member, prompt: a}
steps:
  - {id: s1, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("a")'}
  - {id: s2, who: [A], require_tool: foo, max_retries: 1, instruction: 'foo("b")'}
---
""")
    scen = write_scenario(tmp_path, yaml_text=yaml_text)
    # 只跑了 turn 1
    transcript = [turn_marker(1), speaker("A", "miss")]
    triples = envelope_to_synthetic_triples(envelope(transcript), scen, run_id=0)
    assert len(triples) == 1
    assert triples[0].turn_idx == 1
