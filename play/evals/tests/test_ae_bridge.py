"""evals/_ae_bridge.py re-export sentinel: the first sentinel across subproject import boundaries.

DECISIONS §13 / §16 The established bridge module: put the typed view of `agent_engine` (Result/
Scenario / 6 transcript entries / TokenUsage, etc.) centralized re-export to avoid each metric / task
Modules individually `sys.path.insert(...)` + try/finally cleanup.

If `play/agent_engine` changes the import name / deletes the dataclass / changes the fields:
  - The location where the error is reported directly should be **this bridge** (a visible place);
  - Instead of waiting for downstream `metrics/nudge.py` / `tasks/agent_traj.py` / `tasks/nudge_fire_rate.py`
    Three e2e tests exploded at the same time (error signals spread to irrelevant modules).

This file does not import any evals type other than agent_engine, it is pure import-time sentinel.
If agent_engine is not present, the entire test fails-loud (no skip) - with conftest::agent_engine_required
The runtime probe is different: bridge is an import-level contract, and its absence is truly broken."""

from __future__ import annotations

import dataclasses


# ---------- ① __all__ Complete whitelist -----------------------------------------------

def test_bridge_all_lists_expected_symbols():
    """`_ae_bridge.__all__` explicitly lists all re-export symbols - the sentinel is visible when adding or subtracting."""
    from evals import _ae_bridge as br

    expected = {
        "ArtifactEventEntry",
        "ExpandedTurn",
        "Result",
        "Scenario",
        "SpeakerEntry",
        "SummaryEntry",
        "TokenUsage",
        "ToolCall",
        "ToolCallEntry",
        "TopicEntry",
        "TranscriptEntry",
        "TurnEntry",
        "TurnView",
        "_resolve_who_names",
    }
    assert set(br.__all__) == expected, (
        f"_ae_bridge.__all__ 漂移：\n"
        f"  expected: {sorted(expected)}\n"
        f"  actual:   {sorted(br.__all__)}\n"
        f"  missing:  {sorted(expected - set(br.__all__))}\n"
        f"  extra:    {sorted(set(br.__all__) - expected)}"
    )


def test_bridge_imports_each_symbol():
    """Each symbol must be importable (in case __all__ is listed but not actually imported from agent_engine)."""
    from evals._ae_bridge import (  # noqa: F401
        ArtifactEventEntry,
        ExpandedTurn,
        Result,
        Scenario,
        SpeakerEntry,
        SummaryEntry,
        TokenUsage,
        ToolCall,
        ToolCallEntry,
        TopicEntry,
        TranscriptEntry,
        TurnEntry,
        TurnView,
        _resolve_who_names,
    )


# ---------- ② Key dataclass field schema ----------------------------------

def test_result_dataclass_fields():
    """`Result` 5 fields: envelope schema has the same origin (test_agent_traj_envelope is locked; lock again here
    Prevent agent_engine from changing field names/orders when nudge / agent_traj tasks are exposed at the bridge layer before exploding at the same time)."""
    from evals._ae_bridge import Result

    assert dataclasses.is_dataclass(Result)
    fields = [f.name for f in dataclasses.fields(Result)]
    assert fields == ["artifact", "transcript", "success", "warnings", "usage"], (
        f"Result 字段漂移：{fields}"
    )


def test_speaker_entry_has_type_tag():
    """`SpeakerEntry.type` field exists - §16 forces transcript entry to have an explicit `type` tag
    (metrics/trajectory._score_speakers dispatched with `entry.get('type') == 'speaker'`)."""
    from evals._ae_bridge import SpeakerEntry

    assert dataclasses.is_dataclass(SpeakerEntry)
    fields = {f.name for f in dataclasses.fields(SpeakerEntry)}
    assert "type" in fields and "speaker" in fields and "content" in fields, (
        f"SpeakerEntry 缺关键字段：{fields}"
    )


def test_artifact_event_entry_has_tool_caller_arguments():
    """`ArtifactEventEntry` must contain (type, tool, caller, arguments)——
    `metrics/nudge.classify_failure_mode` + `tasks/agent_traj._pin_trajectory` both isinstance
    + Take these 3 fields to judge wrong_tool / draw decision."""
    from evals._ae_bridge import ArtifactEventEntry

    assert dataclasses.is_dataclass(ArtifactEventEntry)
    fields = {f.name for f in dataclasses.fields(ArtifactEventEntry)}
    for required in ("type", "tool", "caller", "arguments"):
        assert required in fields, f"ArtifactEventEntry 缺 {required!r}：{fields}"


def test_tool_call_entry_has_caller_tool_arguments():
    """`ToolCallEntry` (non-artifact tool call written by tracer) must contain (type, caller, tool, arguments)——
    `metrics/nudge.classify_failure_mode` Use it in parallel with ArtifactEventEntry for dispatch."""
    from evals._ae_bridge import ToolCallEntry

    assert dataclasses.is_dataclass(ToolCallEntry)
    fields = {f.name for f in dataclasses.fields(ToolCallEntry)}
    for required in ("type", "caller", "tool", "arguments"):
        assert required in fields, f"ToolCallEntry 缺 {required!r}：{fields}"


def test_token_usage_has_input_output_tokens():
    """`TokenUsage` (element type of envelope.usage) must contain input_tokens / output_tokens——
    metrics/efficiency.py subsequently obtains the contract entry of usage and cost from envelope."""
    from evals._ae_bridge import TokenUsage

    assert dataclasses.is_dataclass(TokenUsage)
    fields = {f.name for f in dataclasses.fields(TokenUsage)}
    for required in ("input_tokens", "output_tokens"):
        assert required in fields, f"TokenUsage 缺 {required!r}：{fields}"


def test_expanded_turn_has_turn_idx_agent_tool_fields():
    """`ExpandedTurn` (element of Scenario.expanded_turns()) must contain turn_idx / agent /
    step_id / require_tool——derive_expected_turns directly reads these 4 fields to produce expected_require_tool_turns."""
    from evals._ae_bridge import ExpandedTurn

    assert dataclasses.is_dataclass(ExpandedTurn)
    fields = {f.name for f in dataclasses.fields(ExpandedTurn)}
    for required in ("turn_idx", "agent", "step_id", "require_tool"):
        assert required in fields, f"ExpandedTurn 缺 {required!r}：{fields}"


def test_tool_call_has_tool_caller_arguments():
    """`ToolCall` (element of Result.tool_calls(), unified specification ArtifactEvent + ToolCallEntry)
    Required tool / caller / arguments——`tasks/agent_traj._pin_trajectory` direct consumption."""
    from evals._ae_bridge import ToolCall

    assert dataclasses.is_dataclass(ToolCall)
    fields = {f.name for f in dataclasses.fields(ToolCall)}
    for required in ("tool", "caller", "arguments"):
        assert required in fields, f"ToolCall 缺 {required!r}：{fields}"


# ---------- ③ Scenario / TurnView class entrance ----------------------------------

def test_scenario_is_class_with_from_yaml():
    """`Scenario` is a class and exposes the `from_yaml` class method - `test_new_scenarios_smoke`
    + `metrics/nudge.derive_expected_turns` all call this entry."""
    from evals._ae_bridge import Scenario

    assert isinstance(Scenario, type), f"Scenario 应是 class，got {type(Scenario)}"
    assert hasattr(Scenario, "from_yaml"), "Scenario 缺 from_yaml 类方法"
    assert hasattr(Scenario, "expanded_turns"), "Scenario 缺 expanded_turns 方法"


def test_turn_view_is_class_with_attempts():
    """`TurnView` (element of Result.turns()) must have .attempts() / .start_offset entry——
    `metrics/nudge.compute_nudge_fire_rate` takes these two typed views."""
    from evals._ae_bridge import TurnView

    assert isinstance(TurnView, type), f"TurnView 应是 class，got {type(TurnView)}"
    assert hasattr(TurnView, "attempts"), "TurnView 缺 attempts 方法"


# ---------- ④ TranscriptEntry typed union shape ----------------------------

def test_transcript_entry_is_typed_union():
    """`TranscriptEntry` is a typing.Union of 6 entry classes - downstream isinstance dispatch
    Depend on this constraint (`metrics/trajectory.predicate_speakers_covered` etc.)."""
    import typing

    from evals._ae_bridge import (
        ArtifactEventEntry,
        SpeakerEntry,
        SummaryEntry,
        ToolCallEntry,
        TopicEntry,
        TranscriptEntry,
        TurnEntry,
    )

    # Members of typing.Union are exposed through __args__
    assert hasattr(TranscriptEntry, "__args__"), (
        f"TranscriptEntry 应是 typing.Union，got {TranscriptEntry!r}"
    )
    members = set(TranscriptEntry.__args__)
    expected = {
        TopicEntry, TurnEntry, SpeakerEntry,
        ToolCallEntry, ArtifactEventEntry, SummaryEntry,
    }
    assert members == expected, (
        f"TranscriptEntry union 漂移：\n"
        f"  expected: {sorted(c.__name__ for c in expected)}\n"
        f"  actual:   {sorted(c.__name__ for c in members)}"
    )


# ---------- ⑤ sys.path injection side effects ------------------------------------------

def test_bridge_injects_play_dir_into_sys_path():
    """bridge must add `play/` to sys.path (other modules can only be reached through `from agent_engine import`)."""
    import sys

    import evals._ae_bridge  # noqa: F401 — trigger sys.path.insert

    assert any(p.endswith("/play") for p in sys.path), (
        f"bridge 未注入 play/ 到 sys.path；当前 sys.path 后缀片段："
        f"{[p for p in sys.path if 'play' in p]}"
    )


def test_resolve_who_names_callable():
    """`_resolve_who_names` (private in agent_engine.scenario but explicitly exposed by bridge to evals)
    Must be adjustable."""
    from evals._ae_bridge import _resolve_who_names

    assert callable(_resolve_who_names)
