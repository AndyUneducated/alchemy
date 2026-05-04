"""agent_traj envelope 契约：跨项目 JSON 形状 + 派生字段抽取的 contract test.

不跑 agent_engine subprocess（live e2e 在 test_agent_traj_run_live.py），仅在
evals 这一侧锁:
  ① envelope schema：`{transcript, artifact, warnings, success}` → AgentTraj
     可以正确派生 tool_calls / tool_seq / decision 写回 doc.metadata['trajectory']
  ② AgentTraj.load_prediction：score 路径用 row 直接当 envelope 的同型映射
  ③ tool_call (tracer) 与 artifact_event (artifact) 两类事件都能被 _extract_tool_calls
     识别成统一的 tool_calls 形状

为什么独立成文：phase 5 与 phase 4 同源——data shape 是 cross-project 接口契约，
比纯 metric 单测更接近"线上事故来源"，单独留一个 file 让 grep 'envelope' 能直接撞到.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

# play/evals/tests/ → ai_workshops/
REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_DIR = REPO_ROOT / "play"

# agent_engine.result 的 dataclass 字段：契约源头
sys.path.insert(0, str(PLAY_DIR))
try:
    from agent_engine.result import Result  # type: ignore
finally:
    if str(PLAY_DIR) in sys.path:
        sys.path.remove(str(PLAY_DIR))

from evals.api import Doc
from evals.tasks.agent_traj import (
    AgentTraj,
    _extract_decision,
    _extract_tool_calls,
    _pin_trajectory,
)


# ---------- envelope schema 同源（2 条）------------------------------------

def test_envelope_field_names_match_result_dataclass():
    """envelope 必须 1:1 对应 agent_engine.Result 的字段名（cli.py 用 dataclasses.asdict）.

    锁这点是为了让"agent_engine 改字段 → evals 失败"在 CI 第一时间显形——而不是等
    metrics 阶段才发现 'success 字段不见了'.
    """
    result_fields = {f.name for f in dataclasses.fields(Result)}
    expected = {"artifact", "transcript", "success", "warnings"}
    assert result_fields == expected


def test_dataclasses_asdict_matches_envelope_shape():
    """`dataclasses.asdict(Result(...))` 写出的 dict 就是 evals 期望的 envelope."""
    r = Result(
        artifact={"x": "y"},
        transcript=[{"speaker": "A", "content": "hi"}],
        warnings=["w1"],
        success=True,
    )
    envelope = dataclasses.asdict(r)
    assert set(envelope.keys()) == {"artifact", "transcript", "warnings", "success"}
    # 各字段类型匹配 phase 5 数据契约
    assert isinstance(envelope["transcript"], list)
    assert isinstance(envelope["artifact"], dict)
    assert isinstance(envelope["warnings"], list)
    assert isinstance(envelope["success"], bool)


# ---------- _extract_tool_calls：两类事件统一规约（4 条）------------------

def test_extract_tool_calls_recognizes_artifact_event():
    """artifact_event（artifact.py 写）必须被识别成 tool_call."""
    transcript = [
        {"type": "artifact_event", "tool": "write_section", "caller": "A",
         "arguments": {"name": "数据"}, "content": "..."},
    ]
    calls = _extract_tool_calls(transcript)
    assert len(calls) == 1
    assert calls[0] == {
        "tool": "write_section", "caller": "A",
        "arguments": {"name": "数据"},
    }


def test_extract_tool_calls_recognizes_tool_call_type():
    """tracer 写的 tool_call 类型也要识别（含 retrieve_docs 等非 artifact 工具）."""
    transcript = [
        {"type": "tool_call", "tool": "retrieve_docs", "caller": "B",
         "arguments": {"q": "..."}, "result": "...", "ok": True},
    ]
    calls = _extract_tool_calls(transcript)
    assert len(calls) == 1
    assert calls[0]["tool"] == "retrieve_docs"
    assert calls[0]["caller"] == "B"


def test_extract_tool_calls_skips_speaker_and_topic_entries():
    """speaker / topic / turn 标记不是工具调用——必须被过滤."""
    transcript = [
        {"type": "topic", "content": "..."},
        {"type": "turn", "content": "..."},
        {"speaker": "A", "content": "..."},  # 无 type 字段
        {"type": "artifact_event", "tool": "cast_vote", "caller": "A", "arguments": {}},
    ]
    calls = _extract_tool_calls(transcript)
    assert len(calls) == 1
    assert calls[0]["tool"] == "cast_vote"


def test_extract_tool_calls_handles_missing_arguments():
    """老格式 artifact_event 没有 arguments 字段（pre-phase 5）→ 默认空 dict."""
    transcript = [
        {"type": "artifact_event", "tool": "finalize_artifact", "caller": "A",
         "content": "..."},  # 无 arguments
    ]
    calls = _extract_tool_calls(transcript)
    assert calls[0]["arguments"] == {}


# ---------- _extract_decision（3 条）---------------------------------------

def test_extract_decision_from_finalize_artifact_args():
    calls = [
        {"tool": "cast_vote", "caller": "A", "arguments": {"option": "保留"}},
        {"tool": "finalize_artifact", "caller": "M", "arguments": {"decision": "关停"}},
    ]
    assert _extract_decision({"x": "y"}, calls) == "关停"


def test_extract_decision_returns_none_when_no_finalize():
    calls = [{"tool": "propose_vote", "caller": "M", "arguments": {}}]
    assert _extract_decision({}, calls) is None


def test_extract_decision_strips_whitespace():
    calls = [{"tool": "finalize_artifact", "caller": "M", "arguments": {"decision": "  采纳  "}}]
    assert _extract_decision({}, calls) == "采纳"


# ---------- _pin_trajectory：注入完整契约（2 条）---------------------------

def test_pin_trajectory_writes_all_required_keys():
    """pin 后 doc.metadata['trajectory'] 必须有 phase 5 metric 全部依赖的 7 个 key."""
    doc = Doc(id="x", input="...", target=None, metadata={"existing": "v"})
    envelope = {
        "transcript": [
            {"type": "artifact_event", "tool": "finalize_artifact", "caller": "M",
             "arguments": {"decision": "关停"}},
        ],
        "artifact": {"sec": "body"},
        "warnings": [],
        "success": True,
    }
    pinned = _pin_trajectory(doc, envelope)
    traj = pinned.metadata["trajectory"]
    for key in ("transcript", "artifact", "warnings", "success",
                "tool_calls", "tool_seq", "decision"):
        assert key in traj, f"trajectory missing {key!r}"
    assert traj["decision"] == "关停"
    assert traj["tool_seq"] == ["finalize_artifact"]
    # 既有 metadata 字段保留
    assert pinned.metadata["existing"] == "v"


def test_pin_trajectory_does_not_mutate_input_doc():
    """immutability：pin 返回新 Doc（dataclass replace），输入 doc.metadata 不变."""
    doc = Doc(id="x", input="...", target=None, metadata={})
    envelope = {"transcript": [], "artifact": {}, "warnings": [], "success": False}
    _pin_trajectory(doc, envelope)
    assert "trajectory" not in doc.metadata


# ---------- AgentTraj.load_prediction（2 条）------------------------------

def test_load_prediction_translates_row_to_trajectory():
    """row 内 envelope 字段 → doc.metadata['trajectory']；Response 占位（output_type='none'）."""
    task = AgentTraj()
    doc = Doc(id="panel", input="...", target=None, metadata={})
    row = {
        "id": "panel",
        "transcript": [
            {"type": "artifact_event", "tool": "cast_vote", "caller": "A",
             "arguments": {"vote_id": "v1", "option": "保留"}}
        ],
        "artifact": {"sec": "body"},
        "warnings": [],
        "success": True,
    }
    enriched, response = task.load_prediction(doc, row)
    assert response.doc_id == "panel"
    assert response.text is None  # output_type='none'，Response 仅占位 doc_id
    assert enriched.metadata["trajectory"]["tool_seq"] == ["cast_vote"]
    assert enriched.metadata["trajectory"]["success"] is True


def test_load_prediction_handles_minimal_row():
    """row 缺字段（transcript/artifact/warnings/success）→ 全部按空值降级，不抛."""
    task = AgentTraj()
    doc = Doc(id="x", input="...", target=None, metadata={})
    enriched, _ = task.load_prediction(doc, {"id": "x"})
    traj = enriched.metadata["trajectory"]
    assert traj["transcript"] == []
    assert traj["tool_calls"] == []
    assert traj["success"] is False


# ---------- run_fn 缺失时的 fail-fast（1 条）-------------------------------

def test_process_docs_requires_scenario_path():
    """run 路径 process_docs 走到没有 scenario_path 的 doc 必须 fail-fast."""
    def fake_run(_p): return {"transcript": [], "artifact": {}, "warnings": [], "success": True}
    task = AgentTraj(run_fn=fake_run)
    doc_no_scenario = Doc(id="x", input="...", target=None, metadata={})
    with pytest.raises(ValueError, match="scenario_path"):
        task.process_docs([doc_no_scenario])
