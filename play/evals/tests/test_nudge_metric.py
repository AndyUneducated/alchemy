"""metrics/nudge.py 纯函数测试 — 不依赖 task / runner / fixture 文件.

7 组手工小 transcript 演 6 个核心场景：
  1. 空 expected（vacuous） → rate=None
  2. 完美：第一次到位（rate=0）
  3. 全 nudge：每个 require_tool turn 都漏 → mode=missed（rate=1）
  4. wrong_tool：第一次调了别的工具
  5. 多 tool 混合 by_tool breakdown
  6. turn 数不够（subprocess 中途崩 / scenario 截断）→ 算 missed
  7. 同 turn 多次 attempt 但 still 没满足 → 仍算 fired（保守）

§16 起 fixture 用 typed dataclass entry 工厂；envelope 经 dataclasses.asdict 序列化喂给
`compute_nudge_fire_rate(envelope, expected)`（envelope 严格 5 字段）.

不测 derive_expected_turns（它在 test_nudge_fire_rate_score.py 里随 evaluate_score
端到端跑通，间接覆盖；YAML 解析本身的边界由 PyYAML 担保）.
"""

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


# ---------- compute_nudge_fire_rate 主路径 ----------------------------

def test_vacuous_no_expected_turns_returns_none():
    """expected 为空（如 brainstorm/debate/roundtable）→ rate=None, total=0."""
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
    """每个 require_tool turn 第一次就调对 → rate=0, no fires."""
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
    """每个 require_tool turn 第 1 attempt 沉默 → fired, mode=missed."""
    transcript = [
        _turn(1),
        _speaker("B", "我先打个招呼"),       # 漏了 cast_vote
        _speaker("B", "补上"), _event("cast_vote", "B"),  # nudge 后补上
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
    """第 1 attempt 调了别的工具 → fired, mode=wrong_tool."""
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
    """两个工具混合：append_section 全调对、cast_vote 全漏 → by_tool 分别正确."""
    transcript = [
        _turn(1),
        _speaker("A", "append"), _event("append_section", "A"),  # 满足
        _turn(2),
        _speaker("B", "漏 vote"),
        _speaker("B", "补"), _event("cast_vote", "B"),  # nudge 后补
        _turn(3),
        _speaker("C", "再漏"),
        _speaker("C", "补"), _event("cast_vote", "C"),  # nudge 后补
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
    """expected.turn_idx 超过 segments 数（subprocess 中途崩） → 算 missed + fired."""
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
    """同 turn 3 attempt 仍没调对 → fired（保守计为失败；agent_engine warnings 会
    单独标 'skipped required tool'，但 nudge_fire_rate 关心的是"第一次是否到位"）."""
    transcript = [
        _turn(1),
        _speaker("B", "attempt 1"),
        _speaker("B", "attempt 2"),
        _speaker("B", "attempt 3"),  # 始终没调 cast_vote
    ]
    expected = [
        {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
    ]
    result = compute_nudge_fire_rate(_envelope(transcript), expected)
    assert result["nudge_fire_rate"] == 1.0
    assert result["per_turn"][0]["n_attempts"] == 3
    assert result["per_turn"][0]["mode"] == "missed"


# ---------- classify_failure_mode 直测 -------------------------------

def test_classify_missed_when_no_tools_at_all():
    """attempt 内 caller 完全没调任何工具 → missed."""
    events = [_speaker("B", "hi")]
    assert classify_failure_mode(events, "B", "cast_vote") == "missed"


def test_classify_wrong_tool_when_caller_called_other_tool():
    """attempt 内调了别的工具 → wrong_tool."""
    events = [_event("read_artifact", "B")]
    assert classify_failure_mode(events, "B", "cast_vote") == "wrong_tool"


def test_classify_wrong_tool_recognizes_tool_call_events_too():
    """tracer 写的 ToolCallEntry（非 artifact）也算 wrong_tool 信号."""
    events = [ToolCallEntry(caller="B", tool="retrieve_docs", arguments={}, ok=True)]
    assert classify_failure_mode(events, "B", "cast_vote") == "wrong_tool"


def test_classify_ignores_other_caller_events():
    """别的 caller 的事件不算该 agent 调了工具——他还是 missed."""
    events = [_event("cast_vote", "Other")]  # caller 不是 B
    assert classify_failure_mode(events, "B", "cast_vote") == "missed"


def test_wrong_args_bucket_is_api_placeholder_phase_1():
    """wrong_args（调对工具但 schema 拒）是 API 占位——artifact handler 在 error 路径
    不发 event，仅靠 transcript 区分不出"调对了被拒"vs"调了别的工具". 当前归 wrong_tool.

    deferred 到 Phase 5（agent_engine 在 dispatch error 路径补 `{ok: false}` event 后
    启用）；FAILURE_MODES 列出该 key 让下游 by_failure_mode 表头稳定.
    """
    assert "wrong_args" in FAILURE_MODES
    events_called_other = [_event("read_artifact", "B")]
    assert classify_failure_mode(events_called_other, "B", "cast_vote") == "wrong_tool"


def test_by_failure_mode_always_lists_three_buckets():
    """compute 输出的 by_failure_mode 永远列三桶（missed / wrong_tool / wrong_args），
    哪怕计数为 0——稳定 schema，下游聚合 / 报告表头不漂移."""
    result = compute_nudge_fire_rate(
        envelope=_envelope([_turn(1), _speaker("B"), _event("cast_vote", "B")]),
        expected_turns=[
            {"turn_idx": 1, "agent": "B", "step_id": "ballot", "tool": "cast_vote"},
        ],
    )
    assert set(result["by_failure_mode"].keys()) == {"missed", "wrong_tool", "wrong_args"}
    assert all(v == 0 for v in result["by_failure_mode"].values())


# ---------- closure factory（与 trajectory.py 协议同形）-----------------

def test_nudge_fire_rate_metric_closure_factory_protocol():
    """nudge_fire_rate_metric() 返回 (Doc, Response) → rate；与 trajectory.py
    的 task_success / tool_call_set_f1 等闭包工厂协议一致.
    """
    from evals.api import Doc, Response

    metric = nudge_fire_rate_metric()

    transcript = [
        _turn(1), _speaker("B"), _event("cast_vote", "B"),  # 满足
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

    # 缺 metadata（vacuous）→ None
    empty_doc = Doc(id="empty", input="", target=None, metadata={})
    assert metric(empty_doc, Response(doc_id="empty")) is None
