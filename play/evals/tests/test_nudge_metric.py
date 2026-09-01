"""metrics/nudge.py pure function test - does not rely on task / runner / fixture files.

7 groups of handmade transcripts performing 6 core scenes:
  1. empty expected (vacuous) → rate=None
  2. Perfect: in place the first time (rate=0)
  3. Full nudge: every require_tool turn is missed → mode=missed (rate=1)
  4. wrong_tool: Adjusted another tool for the first time
  5. Multi-tool mixing by_tool breakdown
  6. The number of turns is not enough (subprocess crashes/scenario truncation) → considered missed
  7. Multiple attempts with the same turn but still is not satisfied → still considered fired (conservative)

Starting from §16, fixture uses typed dataclass entry factory; envelope is fed via dataclasses.asdict serialization
`compute_nudge_fire_rate(envelope, expected)` (envelope strictly 5 fields).

Unexpected derive_expected_turns (it comes with evaluate_score in test_nudge_fire_rate_score.py
End-to-end runthrough, indirect coverage; the boundaries of YAML parsing itself are guaranteed by PyYAML)."""

from __future__ import annotations

import dataclasses

from agent_engine import (
    ArtifactEventEntry,
    SpeakerEntry,
    ToolCallEntry,
    TopicEntry,
    TurnEntry,
)
from evals.metrics.nudge import (
    FAILURE_MODES,
    classify_failure_mode,
    compute_nudge_fire_rate,
    nudge_fire_rate_metric,
)


# ---------- helpers ---------------------------------------------------

def _turn(idx: int) -> TurnEntry:
    return TurnEntry(content=f"turn {idx}")


def _speaker(name: str, text: str = "") -> SpeakerEntry:
    return SpeakerEntry(speaker=name, content=text)


def _event(tool: str, caller: str) -> ArtifactEventEntry:
    return ArtifactEventEntry(tool=tool, caller=caller, arguments={})


def _envelope(transcript_entries) -> dict:
    return {
        "transcript": [dataclasses.asdict(e) for e in transcript_entries],
        "artifact": {},
        "warnings": [],
        "success": True,
        "usage": [],
    }


# ---------- compute_nudge_fire_rate main path ----------------------------

def test_vacuous_no_expected_turns_returns_none():
    """expected is empty (such as brainstorm/debate/roundtable) → rate=None, total=0."""
    result = compute_nudge_fire_rate(
        envelope=_envelope([_turn(1), _speaker("A")]),
        expected_turns=[],
    )
    assert result["nudge_fire_rate"] is None
    assert result["nudge_fire_count"] == 0
    assert result["require_tool_total"] == 0
    assert result["by_tool"] == {}
    assert result["by_failure_mode"] == {m: 0 for m in FAILURE_MODES}


def test_perfect_first_attempt_satisfies():
    """Each require_tool turn is correct the first time → rate=0, no fires."""
    transcript = [
        _turn(1), _speaker("A", "ack"),
        _turn(2), _speaker("B", "vote"), _event("cast_vote", "B"),
        _turn(3), _speaker("C", "vote"), _event("cast_vote", "C"),
    ]
    expected = [
        {"turn_idx": 2, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        {"turn_idx": 3, "agent": "C", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 0.0
    assert result["nudge_fire_count"] == 0
    assert result["require_tool_total"] == 2
    assert result["by_tool"] == {"cast_vote": {"fired": 0, "total": 2}}
    assert result["by_failure_mode"] == {m: 0 for m in FAILURE_MODES}


def test_all_nudged_missed_mode():
    """Each require_tool turn 1st attempt silent → fired, mode=missed."""
    transcript = [
        _turn(1),
        _speaker("B", "我先打个招呼"),       # Missing cast_vote
        _speaker("B", "补上"), _event("cast_vote", "B"),  # add after nudge
        _turn(2),
        _speaker("C", "也是"),
        _speaker("C", "补"), _event("cast_vote", "C"),
    ]
    expected = [
        {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        {"turn_idx": 2, "agent": "C", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 1.0
    assert result["nudge_fire_count"] == 2
    assert result["by_failure_mode"]["missed"] == 2
    assert result["by_failure_mode"]["wrong_tool"] == 0
    for pt in result["per_turn"]:
        assert pt["fired"] is True
        assert pt["mode"] == "missed"
        assert pt["n_attempts"] == 2


def test_wrong_tool_failure_mode():
    """The first attempt adjusted another tool → fired, mode=wrong_tool."""
    transcript = [
        _turn(1),
        _speaker("B", "先看一下"), _event("read_artifact", "B"),
        _speaker("B", "补对"), _event("cast_vote", "B"),
    ]
    expected = [
        {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 1.0
    assert result["by_failure_mode"]["missed"] == 0
    assert result["by_failure_mode"]["wrong_tool"] == 1
    assert result["per_turn"][0]["mode"] == "wrong_tool"


def test_by_tool_breakdown_separates_per_tool():
    """Mixing of two tools: append_section is fully adjusted, cast_vote is fully leaked → by_tool is correct respectively."""
    transcript = [
        _turn(1),
        _speaker("A", "append"), _event("append_section", "A"),  # satisfy
        _turn(2),
        _speaker("B", "漏 vote"),
        _speaker("B", "补"), _event("cast_vote", "B"),  # nudge
        _turn(3),
        _speaker("C", "再漏"),
        _speaker("C", "补"), _event("cast_vote", "C"),  # nudge
    ]
    expected = [
        {"turn_idx": 1, "agent": "A", "step_id": "vdb", "tool": "append_section"},
        {"turn_idx": 2, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        {"turn_idx": 3, "agent": "C", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert abs(result["nudge_fire_rate"] - 2 / 3) < 1e-9
    assert result["by_tool"] == {
        "append_section": {"fired": 0, "total": 1},
        "cast_vote": {"fired": 2, "total": 2},
    }
    assert result["by_failure_mode"]["missed"] == 2


def test_truncated_run_counts_missing_turn_as_fired():
    """expected.turn_idx exceeds the number of segments (subprocess crashes midway) → counted as missed + fired."""
    transcript = [_turn(1), _speaker("B", "only one turn")]
    expected = [
        {"turn_idx": 5, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 1.0
    assert result["nudge_fire_count"] == 1
    assert result["by_failure_mode"]["missed"] == 1
    assert result["per_turn"][0]["n_attempts"] == 0


def test_multiple_attempts_still_no_required_tool_counts_as_fired():
    """Same as turn 3 attempt still not correct → fired (conservatively counted as failure; agent_engine warnings will
    Mark 'skipped required tool' alone, but nudge_fire_rate cares about "whether it is in place the first time")."""
    transcript = [
        _turn(1),
        _speaker("B", "attempt 1"),
        _speaker("B", "attempt 2"),
        _speaker("B", "attempt 3"),  # Cast_vote is never adjusted
    ]
    expected = [
        {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 1.0
    assert result["per_turn"][0]["n_attempts"] == 3
    assert result["per_turn"][0]["mode"] == "missed"


# ---------- classify_failure_mode direct test -------------------------------

def test_classify_missed_when_no_tools_at_all():
    """The caller in attempt does not adjust any tools at all → missed."""
    events = [_speaker("B", "hi")]
    assert classify_failure_mode(events, "B", "cast_vote") == "missed"


def test_classify_wrong_tool_when_caller_called_other_tool():
    """attempt has internally adjusted other tools → wrong_tool."""
    events = [_event("read_artifact", "B")]
    assert classify_failure_mode(events, "B", "cast_vote") == "wrong_tool"


def test_classify_wrong_tool_recognizes_tool_call_events_too():
    """ToolCallEntry (non-artifact) written by tracer is also counted as wrong_tool signal."""
    events = [ToolCallEntry(caller="B", tool="retrieve_docs", arguments={}, ok=True)]
    assert classify_failure_mode(events, "B", "cast_vote") == "wrong_tool"


def test_classify_ignores_other_caller_events():
    """Events of other callers are not counted as the agent has adjusted the tool - it is still missed."""
    events = [_event("cast_vote", "Other")]  # caller is not B
    assert classify_failure_mode(events, "B", "cast_vote") == "missed"


def test_wrong_args_bucket_is_api_placeholder_phase_1():
    """wrong_args (the adjustment tool but the schema is rejected) is an API placeholder - the artifact handler is in the error path
    No event is sent, and transcript alone cannot distinguish between "correctly adjusted and rejected" vs "adjusted to another tool". Currently, it belongs to wrong_tool.

    deferred to Phase 5 (agent_engine adds `{ok: false}` event to the dispatch error path
    enabled); FAILURE_MODES lists this key to stabilize the downstream by_failure_mode header."""
    assert "wrong_args" in FAILURE_MODES
    events_called_other = [_event("read_artifact", "B")]
    assert classify_failure_mode(events_called_other, "B", "cast_vote") == "wrong_tool"


def test_by_failure_mode_always_lists_three_buckets():
    """The by_failure_mode output by compute is always listed in three buckets (missed / wrong_tool / wrong_args),
    Even if the count is 0 - stable schema, downstream aggregation/reporting headers do not drift."""
    result = compute_nudge_fire_rate(
        envelope=_envelope([_turn(1), _speaker("B"), _event("cast_vote", "B")]),
        expected_turns=[
            {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        ],
    )
    assert set(result["by_failure_mode"].keys()) == {"missed", "wrong_tool", "wrong_args"}
    assert all(v == 0 for v in result["by_failure_mode"].values())


# ---------- closure factory (same shape as trajectory.py protocol) ------------------

def test_nudge_fire_rate_metric_closure_factory_protocol():
    """nudge_fire_rate_metric() returns (Doc, Response) → rate; same as trajectory.py
    The task_success / tool_call_set_f1 and other closure factory protocols are consistent."""
    from evals.api import Doc, Response

    metric = nudge_fire_rate_metric()

    transcript = [
        _turn(1), _speaker("B"), _event("cast_vote", "B"),  # satisfy
        _turn(2), _speaker("C"), _speaker("C"), _event("cast_vote", "C"),  # nudge
    ]
    expected = [
        {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        {"turn_idx": 2, "agent": "C", "step_id": "ballot", "tool": "cast_vote"},
    ]
    doc = Doc(
        id="x",
        input="",
        target=None,
        metadata={
            "trajectory": _envelope(transcript),
            "expected_require_tool_turns": expected,
        },
    )
    rate = metric(doc, Response(doc_id="x"))
    assert rate == 0.5  # 1 fire / 2 total

    # missing metadata (vacuous) → None
    empty_doc = Doc(id="empty", input="", target=None, metadata={})
    assert metric(empty_doc, Response(doc_id="empty")) is None
