"""agent_traj envelope 契约：跨项目 JSON 形状 + 派生字段抽取的 contract test.

不跑 agent_engine subprocess（live e2e 在 test_agent_traj_run_live.py），仅在
evals 这一侧锁:
  ① envelope schema：`{transcript, artifact, warnings, success, usage}` → AgentTraj
     可以正确派生 tool_calls / tool_seq / decision 写回 doc.metadata['trajectory']
  ② AgentTraj.load_prediction：score 路径用 row 直接当 envelope 的同型映射

DECISIONS §16 起 envelope 5 字段（usage 字段加入）；transcript / usage 两侧都是
typed dataclass，asdict 序列化为 dict 形态保存到 predictions JSONL.

为什么独立成文：phase 5 与 phase 4 同源——data shape 是 cross-project 接口契约，
比纯 metric 单测更接近"线上事故来源"，单独留一个 file 让 grep 'envelope' 能直接撞到.

DECISIONS §13 后 transcript 内"工具调用规约"+"decision 抽取"的等价覆盖已迁到
[`agent_engine/tests/test_result_views.py`]；本文件只保留 evals 自身的契约：envelope
字段 ↔ Result 同源 + `_pin_trajectory` 注入形状 + `AgentTraj.load_prediction` 行为.
"""

from __future__ import annotations

import dataclasses

import pytest

from evals._ae_bridge import ArtifactEventEntry, Result, SpeakerEntry
from evals.api import Doc
from evals.tasks.agent_traj import AgentTraj, _pin_trajectory


# ---------- envelope schema 同源 -------------------------------------

def test_envelope_field_names_match_result_dataclass():
    """envelope 必须 1:1 对应 agent_engine.Result 的字段名（cli.py 用 dataclasses.asdict）."""
    result_fields = {f.name for f in dataclasses.fields(Result)}
    expected = {"artifact", "transcript", "success", "warnings", "usage"}
    assert result_fields == expected


def test_dataclasses_asdict_matches_envelope_shape():
    """`dataclasses.asdict(Result(...))` 写出的 dict 就是 evals 期望的 envelope."""
    r = Result(
        artifact={"x": "y"},
        transcript=[SpeakerEntry(speaker="A", content="hi")],
        warnings=["w1"],
        success=True,
        usage=[],
    )
    envelope = dataclasses.asdict(r)
    assert set(envelope.keys()) == {"artifact", "transcript", "warnings", "success", "usage"}
    assert isinstance(envelope["transcript"], list)
    assert isinstance(envelope["transcript"][0], dict)  # asdict 递归展平
    assert envelope["transcript"][0]["type"] == "speaker"
    assert isinstance(envelope["artifact"], dict)
    assert isinstance(envelope["warnings"], list)
    assert isinstance(envelope["success"], bool)
    assert isinstance(envelope["usage"], list)


# ---------- _pin_trajectory：注入完整契约 ---------------------------

def test_pin_trajectory_writes_all_required_keys():
    """pin 后 doc.metadata['trajectory'] 必须有 phase 5 metric 全部依赖的 8 个 key."""
    doc = Doc(id="x", input="...", target=None, metadata={"existing": "v"})
    envelope = dataclasses.asdict(Result(
        transcript=[ArtifactEventEntry(
            tool="finalize_artifact", caller="M",
            arguments={"decision": "关停"},
        )],
        artifact={"sec": "body"},
        warnings=[],
        success=True,
        usage=[],
    ))
    pinned = _pin_trajectory(doc, envelope)
    traj = pinned.metadata["trajectory"]
    for key in ("transcript", "artifact", "warnings", "success", "usage",
                "tool_calls", "tool_seq", "decision"):
        assert key in traj, f"trajectory missing {key!r}"
    assert traj["decision"] == "关停"
    assert traj["tool_seq"] == ["finalize_artifact"]
    # 既有 metadata 字段保留
    assert pinned.metadata["existing"] == "v"


def test_pin_trajectory_does_not_mutate_input_doc():
    """immutability：pin 返回新 Doc（dataclass replace），输入 doc.metadata 不变."""
    doc = Doc(id="x", input="...", target=None, metadata={})
    envelope = {
        "transcript": [], "artifact": {},
        "warnings": [], "success": False, "usage": [],
    }
    _pin_trajectory(doc, envelope)
    assert "trajectory" not in doc.metadata


# ---------- AgentTraj.load_prediction --------------------------------

def test_load_prediction_translates_row_to_trajectory():
    """row 内 envelope 字段 → doc.metadata['trajectory']；Response 占位（output_type='none'）."""
    task = AgentTraj()
    doc = Doc(id="panel", input="...", target=None, metadata={})
    row = dataclasses.asdict(Result(
        transcript=[ArtifactEventEntry(
            tool="cast_vote", caller="A",
            arguments={"vote_id": "v1", "option": "保留"},
        )],
        artifact={"sec": "body"},
        warnings=[],
        success=True,
        usage=[],
    ))
    row["id"] = "panel"
    enriched, response = task.load_prediction(doc, row)
    assert response.doc_id == "panel"
    assert response.text is None  # output_type='none'，Response 仅占位 doc_id
    assert enriched.metadata["trajectory"]["tool_seq"] == ["cast_vote"]
    assert enriched.metadata["trajectory"]["success"] is True


def test_load_prediction_strict_on_missing_envelope_field():
    """§16 起 envelope 严格 5 字段——load_prediction 缺字段直接 KeyError."""
    task = AgentTraj()
    doc = Doc(id="x", input="...", target=None, metadata={})
    with pytest.raises(KeyError):
        task.load_prediction(doc, {"id": "x"})


# ---------- run_fn 缺失时的 fail-fast --------------------------------

def test_process_docs_requires_scenario_path():
    """run 路径 process_docs 走到没有 scenario_path 的 doc 必须 fail-fast."""
    def fake_run(_p):
        return dataclasses.asdict(Result(usage=[]))
    task = AgentTraj(run_fn=fake_run)
    doc_no_scenario = Doc(id="x", input="...", target=None, metadata={})
    with pytest.raises(ValueError, match="scenario_path"):
        task.process_docs([doc_no_scenario])
