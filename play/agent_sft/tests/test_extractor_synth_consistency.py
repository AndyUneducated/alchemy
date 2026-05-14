"""extractor.py vs synthesize.py — 同 envelope 双跑后元数据应一致.

两个 mining 路径独立演化（[DECISIONS §5](../DECISIONS.md) 给的 1k 数据用 synthesize
per-fire 策略；extractor 走 "first-fail + later-success" 真自纠）。但凡 require_tool
触发 nudge fire，两条路径应该:
  - 锚定相同 (turn_idx, step_id, agent, required_tool, failure_mode)
  - 抓相同的 failed_response（first attempt 的第一句 SpeakerEntry）
  - 抓相同的 instruction（scenario YAML 透传）
  - 抓相同的 context 切点（first speaker entry 之前的 prefix）

唯一允许分歧的字段：**corrected_response**——
  - extractor 取后续 attempt 真实成功的 SpeakerEntry.content
  - synthesize 走 _extract_call_template 程序合成 `tool(args)` 字面量

本测套用 test_extractor.py 同款 typed fixture（max_retries=1，让 extractor 也产
triple），跑双路径后 zip 比对共有字段；catch 两脚本分头改 metadata semantics
导致 train 数据矛盾的事故.
"""

from __future__ import annotations

import dataclasses
import textwrap

from extractor import extract_triples  # type: ignore[import-not-found]
from synthesize import (  # type: ignore[import-not-found]
    envelope_to_synthetic_triples,
)
from agent_engine import (  # type: ignore[import-not-found]
    SpeakerEntry,
    ToolCallEntry,
    TopicEntry,
    TurnEntry,
)


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
      调用 foo_tool(arg="x") 完成本步。
---
body
""")


def _write_scenario(tmp_path):
    p = tmp_path / "scen.md"
    p.write_text(SCENARIO_YAML, encoding="utf-8")
    return p


def _envelope(entries):
    return {
        "transcript": [dataclasses.asdict(e) for e in entries],
        "artifact": {},
        "warnings": [],
        "success": True,
        "usage": [],
    }


def test_extractor_and_synthesize_agree_on_metadata(tmp_path):
    """missed-then-success 场景下，两路径产同一锚点的 triple，metadata 全一致."""
    scen = _write_scenario(tmp_path)
    transcript = [
        TopicEntry(content="demo", ts=0.0),
        TurnEntry(content="turn 1 of 1", ts=1.0),
        SpeakerEntry(speaker="A", content="我先想想", ts=1.0),  # first attempt: missed
        SpeakerEntry(speaker="A", content='好 foo_tool(arg="x")', ts=2.0),  # retry: success
        ToolCallEntry(caller="A", tool="foo_tool", ok=True, ts=2.0),
    ]
    env = _envelope(transcript)

    e_trips = extract_triples(env, scen, run_id=7, scenario_name="scen")
    s_trips = envelope_to_synthetic_triples(env, scen, run_id=7, scenario_name="scen")

    assert len(e_trips) == 1, f"extractor should produce 1 triple, got {len(e_trips)}"
    assert len(s_trips) == 1, f"synthesize should produce 1 triple, got {len(s_trips)}"

    e, s = e_trips[0], s_trips[0]

    # 锚点 5 元组应严格一致
    for field in ("run_id", "scenario", "turn_idx", "step_id", "agent", "required_tool"):
        assert getattr(e, field) == getattr(s, field), (
            f"metadata divergence on `{field}`: extractor={getattr(e, field)!r} "
            f"vs synthesize={getattr(s, field)!r}"
        )

    # failure_mode 必须一致（两路径都走 classify_failure_mode on first attempt）
    assert e.failure_mode == s.failure_mode == "missed"

    # failed_response: 都是 first SpeakerEntry.content
    assert e.failed_response == s.failed_response == "我先想想"

    # instruction: scenario YAML 透传，必须 byte-identical
    assert e.instruction == s.instruction
    assert e.instruction.startswith("调用 foo_tool")

    # context: prefix until first speaker entry，两路径切点应同
    assert e.context == s.context, (
        "context divergence — first-speaker-entry 切点定义两路径不一致"
    )

    # nudge: 同 NUDGE_TEMPLATE.format(tool=required_tool)
    assert e.nudge == s.nudge

    # corrected_response 允许分歧（这是两路径的核心区别），但都必须非空
    assert e.corrected_response, "extractor corrected must be non-empty"
    assert s.corrected_response, "synthesize corrected must be non-empty"
    # extractor 取真 speaker.content；synthesize 取程序合成
    assert e.corrected_response == '好 foo_tool(arg="x")'
    # synthesize 合成必须含 required_tool 名（call template 抽取）
    assert "foo_tool" in s.corrected_response
