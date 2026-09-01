"""Result / ToolCall / TurnView view tests (DECISIONS §13 / §16).

Covers:
  - `Result.from_dict` / `Result.load_json`: envelope ↔ Result round-trip (§16 strict, KeyError on missing fields)
  - `Result.tool_calls()` recognizes `ToolCallEntry` + `ArtifactEventEntry`
  - `Result.turns()` segmentation: empty segments, pre-turn debris dropped, 1-based turn_idx, global start_offset
  - `TurnView.attempts(agent)`: new attempt per SpeakerEntry, silence → 0
  - `Result.find_finalize_decision()` hit / miss / strip / multiple takes last
  - `Result.speakers()` dedup
  - `TranscriptEntry` typed union (§16): frozen dataclass per entry type with explicit type tag
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agent_engine import (
    ArtifactEventEntry,
    Result,
    SpeakerEntry,
    TokenUsage,
    ToolCall,
    ToolCallEntry,
    TopicEntry,
    TurnEntry,
    TurnView,
)


# ---------- helpers ----------------------------------------------------

def _turn(idx: int, total: int = 1) -> TurnEntry:
    return TurnEntry(content=f"turn {idx} of {total}")


def _speaker(name: str, text: str = "") -> SpeakerEntry:
    return SpeakerEntry(speaker=name, content=text)


def _event(tool: str, caller: str, arguments: dict | None = None) -> ArtifactEventEntry:
    return ArtifactEventEntry(
        tool=tool, caller=caller, arguments=arguments or {},
    )


def _tool_call(tool: str, caller: str, arguments: dict | None = None) -> ToolCallEntry:
    return ToolCallEntry(
        tool=tool, caller=caller, arguments=arguments or {},
    )


# ---------- IO --------------------------------------------------------

def test_from_dict_full_envelope_round_trips():
    """asdict(Result) → from_dict byte-identical round-trip."""
    r = Result(
        artifact={"sec": "body"},
        transcript=[_speaker("A", "hi")],
        success=True,
        warnings=["w1"],
        usage=[TokenUsage(model="m", caller="A", input_tokens=10, output_tokens=20)],
    )
    envelope = dataclasses.asdict(r)
    r2 = Result.from_dict(envelope)
    assert r2.artifact == r.artifact
    assert r2.transcript == r.transcript
    assert r2.success == r.success
    assert r2.warnings == r.warnings
    assert r2.usage == r.usage


def test_from_dict_strict_raises_on_missing_field():
    """§16 onward Result.from_dict strict — any missing field → KeyError."""
    with pytest.raises(KeyError):
        Result.from_dict({"transcript": []})  # missing artifact / success / warnings / usage
    with pytest.raises(KeyError):
        Result.from_dict({})


def test_from_dict_strict_raises_on_unknown_entry_type():
    """transcript entry must have valid `type`; unregistered type → KeyError."""
    envelope = {
        "artifact": {}, "transcript": [{"type": "unknown_kind"}],
        "success": True, "warnings": [], "usage": [],
    }
    with pytest.raises(KeyError):
        Result.from_dict(envelope)


def test_load_json_reads_save_result_json_format(tmp_path: Path):
    """JSON from `cli.py --save-result-json` round-trips via load_json."""
    r = Result(
        artifact={"x": "y"},
        transcript=[TopicEntry(content="hello")],
        usage=[],
    )
    envelope = dataclasses.asdict(r)
    p = tmp_path / "envelope.json"
    p.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    r2 = Result.load_json(p)
    assert r2.transcript == r.transcript
    assert r2.artifact == r.artifact
    assert r2.success is True


# ---------- typed entry round-trip -------------------------------------

@pytest.mark.parametrize("entry", [
    TopicEntry(content="t"),
    TurnEntry(content="turn 1 of 1"),
    SpeakerEntry(speaker="A", content="hi"),
    ToolCallEntry(caller="A", tool="retrieve_docs", arguments={"q": "x"}),
    ArtifactEventEntry(tool="cast_vote", caller="A", arguments={"option": "yes"}),
])
def test_typed_entry_round_trips_through_envelope(entry):
    """Each typed entry round-trips asdict → from_dict with same type."""
    r = Result(transcript=[entry])
    envelope = dataclasses.asdict(r)
    r2 = Result.from_dict(envelope)
    assert r2.transcript == [entry]
    assert type(r2.transcript[0]) is type(entry)


def test_speaker_entry_carries_explicit_type_tag():
    """§16: SpeakerEntry must include explicit type='speaker', aligned with other entries."""
    s = SpeakerEntry(speaker="A", content="hi")
    d = dataclasses.asdict(s)
    assert d["type"] == "speaker"


# ---------- tool_calls -------------------------------------------------

def test_tool_calls_recognizes_artifact_event():
    """ArtifactEventEntry → ToolCall(kind='artifact')."""
    r = Result(transcript=[
        ArtifactEventEntry(
            tool="write_section", caller="A",
            arguments={"name": "数据"}, content="...", ts=1.5,
        ),
    ])
    calls = r.tool_calls()
    assert len(calls) == 1
    tc = calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.tool == "write_section"
    assert tc.caller == "A"
    assert tc.arguments == {"name": "数据"}
    assert tc.kind == "artifact"
    assert tc.ts == 1.5


def test_tool_calls_recognizes_tracer_tool_call():
    """ToolCallEntry → ToolCall(kind='tracer'); non-artifact tools like retrieve_docs."""
    r = Result(transcript=[
        ToolCallEntry(tool="retrieve_docs", caller="B", arguments={"q": "..."}, ok=True),
    ])
    calls = r.tool_calls()
    assert len(calls) == 1
    assert calls[0].tool == "retrieve_docs"
    assert calls[0].caller == "B"
    assert calls[0].kind == "tracer"


def test_tool_calls_skips_non_tool_entries():
    """topic / turn / speaker entries are not tool calls; must be filtered."""
    r = Result(transcript=[
        TopicEntry(content="..."),
        TurnEntry(content="turn 1 of 2"),
        _speaker("A", "hi"),
        _event("cast_vote", "A"),
    ])
    calls = r.tool_calls()
    assert len(calls) == 1
    assert calls[0].tool == "cast_vote"


def test_tool_calls_preserves_order():
    """tool_calls order = transcript order; mixed artifact_event and tool_call not reordered."""
    r = Result(transcript=[
        _tool_call("retrieve_docs", "A"),
        _event("append_section", "A"),
        _tool_call("retrieve_docs", "B"),
    ])
    calls = r.tool_calls()
    assert [c.tool for c in calls] == ["retrieve_docs", "append_section", "retrieve_docs"]
    assert [c.caller for c in calls] == ["A", "A", "B"]


# ---------- turns ------------------------------------------------------

def test_turns_partitions_by_marker_and_drops_pre_turn_entries():
    """Segment by turn marker; marker dropped; pre-turn topic debris dropped."""
    r = Result(transcript=[
        TopicEntry(content="话题"),
        _turn(1, 2), _speaker("A", "hi"),
        _turn(2, 2), _speaker("B", "ho"),
    ])
    turns = r.turns()
    assert len(turns) == 2
    assert turns[0].turn_idx == 1
    assert turns[1].turn_idx == 2
    assert turns[0].entries == (_speaker("A", "hi"),)
    assert turns[1].entries == (_speaker("B", "ho"),)


def test_turns_handles_consecutive_markers_yields_empty_segment():
    """Between consecutive turn markers → empty segment (entries=()); turn_idx monotonic."""
    r = Result(transcript=[_turn(1), _turn(2), _speaker("A")])
    turns = r.turns()
    assert len(turns) == 2
    assert turns[0].entries == ()
    assert turns[1].entries == (_speaker("A"),)
    assert turns[0].turn_idx == 1
    assert turns[1].turn_idx == 2


def test_turns_start_offset_maps_back_to_global_index():
    """start_offset = 0-based index of segment first entry in original transcript."""
    transcript = [
        TopicEntry(content="x"),         # idx 0
        _turn(1),                        # idx 1
        _speaker("A", "a"),              # idx 2
        _speaker("A", "b"),              # idx 3
        _turn(2),                        # idx 4
        _speaker("B", "c"),              # idx 5
    ]
    r = Result(transcript=transcript)
    turns = r.turns()
    assert turns[0].start_offset == 2
    assert turns[1].start_offset == 5


def test_turns_returns_empty_when_no_turn_marker():
    """No turn marker → empty list (all pre-turn content dropped)."""
    r = Result(transcript=[_speaker("A"), _speaker("B")])
    assert r.turns() == []


# ---------- TurnView.attempts -----------------------------------------

def test_attempts_each_speaker_starts_new_attempt():
    """Speaker entry starts new attempt; attempt includes events until next speaker."""
    tv = TurnView(turn_idx=1, start_offset=0, entries=(
        _speaker("A", "first"),
        _event("read_artifact", "A"),
        _speaker("A", "second"),
        _event("cast_vote", "A"),
    ))
    attempts = tv.attempts("A")
    assert len(attempts) == 2
    assert attempts[0] == [_event("read_artifact", "A")]
    assert attempts[1] == [_event("cast_vote", "A")]


def test_attempts_silent_segment_returns_zero_attempts():
    """No speaker entries → 0 attempts (caller completely silent)."""
    tv = TurnView(turn_idx=1, start_offset=0, entries=(
        _event("read_artifact", "Other"),
    ))
    assert tv.attempts("A") == []


def test_attempts_drops_entries_before_first_speaker():
    """Events before first speaker dropped (e.g. instruction marker at turn start)."""
    tv = TurnView(turn_idx=1, start_offset=0, entries=(
        TopicEntry(content="..."),
        _speaker("A"),
        _event("cast_vote", "A"),
    ))
    attempts = tv.attempts("A")
    assert len(attempts) == 1
    assert attempts[0] == [_event("cast_vote", "A")]


def test_turn_view_tool_calls_filters_to_segment():
    """TurnView.tool_calls() returns only tool calls within segment."""
    tv = TurnView(turn_idx=1, start_offset=0, entries=(
        _speaker("A"),
        _event("cast_vote", "A"),
        _tool_call("retrieve_docs", "A"),
    ))
    calls = tv.tool_calls()
    assert [c.tool for c in calls] == ["cast_vote", "retrieve_docs"]


# ---------- find_finalize_decision -------------------------------------

def test_find_finalize_decision_extracts_from_arguments():
    r = Result(transcript=[
        _event("cast_vote", "A", {"option": "yes"}),
        _event("finalize_artifact", "M", {"decision": "关停"}),
    ])
    assert r.find_finalize_decision() == "关停"


def test_find_finalize_decision_returns_none_when_absent():
    """No finalize in transcript → None; finalize but missing decision → None."""
    r = Result(transcript=[_event("propose_vote", "M")])
    assert r.find_finalize_decision() is None
    r2 = Result(transcript=[_event("finalize_artifact", "M")])
    assert r2.find_finalize_decision() is None


def test_find_finalize_decision_strips_whitespace():
    r = Result(transcript=[
        _event("finalize_artifact", "M", {"decision": "  采纳  "}),
    ])
    assert r.find_finalize_decision() == "采纳"


def test_find_finalize_decision_returns_last_when_multiple():
    """Multiple successful finalize in edge case → return **last** decision (closest to sealed state)."""
    r = Result(transcript=[
        _event("finalize_artifact", "M", {"decision": "first"}),
        _event("finalize_artifact", "M", {"decision": "second"}),
    ])
    assert r.find_finalize_decision() == "second"


# ---------- speakers ---------------------------------------------------

def test_speakers_returns_distinct_set():
    r = Result(transcript=[
        _speaker("A"), _speaker("A"), _speaker("B"),
        _event("x", "C"),
        TopicEntry(content="..."),
    ])
    assert r.speakers() == {"A", "B"}  # caller not counted as speaker


def test_speakers_empty_when_no_speech():
    r = Result(transcript=[_turn(1)])
    assert r.speakers() == set()
