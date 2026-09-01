"""extractor.py vs synthesize.py — The same metadata should be consistent after running both envelopes.

The two mining paths evolve independently ([DECISIONS §5](../DECISIONS.md) gives 1k data using synthesize
per-fire strategy; extractor adopts "first-fail + later-success" (true self-correction). Whenever require_tool
To trigger nudge fire, the two paths should be:
  - Anchoring the same (turn_idx, step_id, agent, required_tool, failure_mode)
  - Catch the same failed_response (the first sentence of first attempt SpeakerEntry)
  - Capture the same instruction (scenario YAML transparent transmission)
  - Grab the same context cut point (prefix before first speaker entry)

The only field that allows disagreement: **corrected_response**——
  - extractor retrieves the actual successful SpeakerEntry.content of subsequent attempts
  -synthesize _extract_call_template program to synthesize `tool(args)` literal

This test uses the same typed fixture as test_extractor.py (max_retries=1, so that the extractor can also produce
triple), after running dual paths, zip compares the common fields; the catch two scripts change the metadata semantics separately
Accidents resulting in conflicting train data."""

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
    """In the missed-then-success scenario, the two paths produce triples of the same anchor point, and the metadata is completely consistent."""
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

    # anchor 5-tuple must match exactly
    # anchor 5-tuple must match exactly
    for field in ("run_id", "scenario", "turn_idx", "step_id", "agent", "required_tool"):
        assert getattr(e, field) == getattr(s, field), (
            f"metadata divergence on `{field}`: extractor={getattr(e, field)!r} "
            f"vs synthesize={getattr(s, field)!r}"
        )

# failure_mode must be consistent (both paths use classify_failure_mode on first attempt)
# failure_mode must be consistent (both paths use classify_failure_mode on first attempt)
    assert e.failure_mode == s.failure_mode == "missed"

# failed_response: both are first SpeakerEntry.content
# failed_response: both are first SpeakerEntry.content
    assert e.failed_response == s.failed_response == "我先想想"

# instruction: scenario YAML transparent transmission, must be byte-identical
# instruction: scenario YAML transparent transmission, must be byte-identical
    assert e.instruction == s.instruction
    assert e.instruction.startswith("调用 foo_tool")

# context: prefix until first speaker entry, the two path cut points should be the same
# context: prefix until first speaker entry, the two path cut points should be the same
    assert e.context == s.context, (
        "context divergence — first-speaker-entry 切点定义两路径不一致"
    )

# nudge: Same as NUDGE_TEMPLATE.format(tool=required_tool)
# nudge: Same as NUDGE_TEMPLATE.format(tool=required_tool)
    assert e.nudge == s.nudge

# corrected_response allows differences (this is the core difference between the two paths), but both must be non-empty
# corrected_response allows differences (this is the core difference between the two paths), but both must be non-empty
    assert e.corrected_response, "extractor corrected must be non-empty"
    assert s.corrected_response, "synthesize corrected must be non-empty"
# extractor gets true speaker.content; synthesizesize gets program synthesis
# extractor gets true speaker.content; synthesizesize gets program synthesis
    assert e.corrected_response == '好 foo_tool(arg="x")'
#synthesize must contain required_tool name (call template extraction)
#synthesize must contain required_tool name (call template extraction)
    assert "foo_tool" in s.corrected_response
