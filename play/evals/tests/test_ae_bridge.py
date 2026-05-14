"""evals/_ae_bridge.py re-export sentinel：跨子项目 import 边界的第一道哨兵.

DECISIONS §13 / §16 立的 bridge 模块：把 `agent_engine` 的 typed view（Result /
Scenario / 6 个 transcript entry / TokenUsage 等）集中 re-export，避免各 metric / task
模块各自 `sys.path.insert(...)` + try/finally 清理.

如果 `play/agent_engine` 改了 import 名 / 删了 dataclass / 改了字段：
  - 直接报错的位置应该是**这个 bridge**（一处显形）；
  - 而不是等下游 `metrics/nudge.py` / `tasks/agent_traj.py` / `tasks/nudge_fire_rate.py`
    三处 e2e test 同时炸（错误信号弥散到无关模块）.

本文件不引 agent_engine 之外的任何 evals 类型，纯 import-time sentinel.
agent_engine 不在则整体 test fail-loud（无 skip）—— 与 conftest::agent_engine_required
的运行时 probe 不同：bridge 是 import-level 契约，缺失就是真破.
"""

from __future__ import annotations

import dataclasses


# ---------- ① __all__ 白名单完整 -------------------------------------------

def test_bridge_all_lists_expected_symbols():
    """`_ae_bridge.__all__` 显式列出所有 re-export 符号——增减时 sentinel 显形."""
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
    """每个符号必须可 import（防 __all__ 列了但实际没 from agent_engine import）."""
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


# ---------- ② 关键 dataclass 字段 schema ----------------------------------

def test_result_dataclass_fields():
    """`Result` 5 字段：envelope schema 同源（test_agent_traj_envelope 已锁；这里再锁
    防止 agent_engine 改字段名/顺序时 nudge / agent_traj task 同时炸前先在 bridge 层显形）.
    """
    from evals._ae_bridge import Result

    assert dataclasses.is_dataclass(Result)
    fields = [f.name for f in dataclasses.fields(Result)]
    assert fields == ["artifact", "transcript", "success", "warnings", "usage"], (
        f"Result 字段漂移：{fields}"
    )


def test_speaker_entry_has_type_tag():
    """`SpeakerEntry.type` 字段存在——§16 强制 transcript entry 显式 `type` 标签
    （metrics/trajectory._score_speakers 用 `entry.get('type') == 'speaker'` 派发）."""
    from evals._ae_bridge import SpeakerEntry

    assert dataclasses.is_dataclass(SpeakerEntry)
    fields = {f.name for f in dataclasses.fields(SpeakerEntry)}
    assert "type" in fields and "speaker" in fields and "content" in fields, (
        f"SpeakerEntry 缺关键字段：{fields}"
    )


def test_artifact_event_entry_has_tool_caller_arguments():
    """`ArtifactEventEntry` 必含 (type, tool, caller, arguments)——
    `metrics/nudge.classify_failure_mode` + `tasks/agent_traj._pin_trajectory` 都 isinstance
    + 取这 3 字段判 wrong_tool / 抽 decision."""
    from evals._ae_bridge import ArtifactEventEntry

    assert dataclasses.is_dataclass(ArtifactEventEntry)
    fields = {f.name for f in dataclasses.fields(ArtifactEventEntry)}
    for required in ("type", "tool", "caller", "arguments"):
        assert required in fields, f"ArtifactEventEntry 缺 {required!r}：{fields}"


def test_tool_call_entry_has_caller_tool_arguments():
    """`ToolCallEntry`（tracer 写的非 artifact 工具调用）必含 (type, caller, tool, arguments)——
    `metrics/nudge.classify_failure_mode` 用其与 ArtifactEventEntry 并列做派发."""
    from evals._ae_bridge import ToolCallEntry

    assert dataclasses.is_dataclass(ToolCallEntry)
    fields = {f.name for f in dataclasses.fields(ToolCallEntry)}
    for required in ("type", "caller", "tool", "arguments"):
        assert required in fields, f"ToolCallEntry 缺 {required!r}：{fields}"


def test_token_usage_has_input_output_tokens():
    """`TokenUsage`（envelope.usage 的元素类型）必含 input_tokens / output_tokens——
    metrics/efficiency.py 后续从 envelope 取 usage 计 cost 的契约入口."""
    from evals._ae_bridge import TokenUsage

    assert dataclasses.is_dataclass(TokenUsage)
    fields = {f.name for f in dataclasses.fields(TokenUsage)}
    for required in ("input_tokens", "output_tokens"):
        assert required in fields, f"TokenUsage 缺 {required!r}：{fields}"


def test_expanded_turn_has_turn_idx_agent_tool_fields():
    """`ExpandedTurn`（Scenario.expanded_turns() 的元素）必含 turn_idx / agent /
    step_id / require_tool——derive_expected_turns 直接读这 4 字段产 expected_require_tool_turns."""
    from evals._ae_bridge import ExpandedTurn

    assert dataclasses.is_dataclass(ExpandedTurn)
    fields = {f.name for f in dataclasses.fields(ExpandedTurn)}
    for required in ("turn_idx", "agent", "step_id", "require_tool"):
        assert required in fields, f"ExpandedTurn 缺 {required!r}：{fields}"


def test_tool_call_has_tool_caller_arguments():
    """`ToolCall`（Result.tool_calls() 的元素，统一规约 ArtifactEvent + ToolCallEntry）
    必含 tool / caller / arguments——`tasks/agent_traj._pin_trajectory` 直接消费."""
    from evals._ae_bridge import ToolCall

    assert dataclasses.is_dataclass(ToolCall)
    fields = {f.name for f in dataclasses.fields(ToolCall)}
    for required in ("tool", "caller", "arguments"):
        assert required in fields, f"ToolCall 缺 {required!r}：{fields}"


# ---------- ③ Scenario / TurnView 类入口 ----------------------------------

def test_scenario_is_class_with_from_yaml():
    """`Scenario` 是 class 且暴露 `from_yaml` 类方法——`test_new_scenarios_smoke`
    + `metrics/nudge.derive_expected_turns` 都 call 这个入口."""
    from evals._ae_bridge import Scenario

    assert isinstance(Scenario, type), f"Scenario 应是 class，got {type(Scenario)}"
    assert hasattr(Scenario, "from_yaml"), "Scenario 缺 from_yaml 类方法"
    assert hasattr(Scenario, "expanded_turns"), "Scenario 缺 expanded_turns 方法"


def test_turn_view_is_class_with_attempts():
    """`TurnView`（Result.turns() 的元素）必有 .attempts() / .start_offset 入口——
    `metrics/nudge.compute_nudge_fire_rate` 走这两个 typed view."""
    from evals._ae_bridge import TurnView

    assert isinstance(TurnView, type), f"TurnView 应是 class，got {type(TurnView)}"
    assert hasattr(TurnView, "attempts"), "TurnView 缺 attempts 方法"


# ---------- ④ TranscriptEntry typed union 形状 ----------------------------

def test_transcript_entry_is_typed_union():
    """`TranscriptEntry` 是 6 个 entry class 的 typing.Union——下游 isinstance dispatch
    依赖此约束（`metrics/trajectory.predicate_speakers_covered` 等）."""
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

    # typing.Union 的成员通过 __args__ 暴露
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


# ---------- ⑤ sys.path 注入副作用 -----------------------------------------

def test_bridge_injects_play_dir_into_sys_path():
    """bridge 必须把 `play/` 加进 sys.path（其它模块 `from agent_engine import` 才能可达）."""
    import sys

    import evals._ae_bridge  # noqa: F401  — 触发 sys.path.insert

    assert any(p.endswith("/play") for p in sys.path), (
        f"bridge 未注入 play/ 到 sys.path；当前 sys.path 后缀片段："
        f"{[p for p in sys.path if 'play' in p]}"
    )


def test_resolve_who_names_callable():
    """`_resolve_who_names`（agent_engine.scenario 内私有但 bridge 显式暴露给 evals 用）
    必须可调."""
    from evals._ae_bridge import _resolve_who_names

    assert callable(_resolve_who_names)
