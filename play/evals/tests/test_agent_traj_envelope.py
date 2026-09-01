"""agent_traj envelope contract: contract test for cross-project JSON shape + derived field extraction.

Do not run agent_engine subprocess (live e2e in test_agent_traj_run_live.py), only
evals lock on this side:
  ① envelope schema：`{transcript, artifact, warnings, success, usage}` → AgentTraj
     Can correctly derive tool_calls / tool_seq / decision and write back doc.metadata['trajectory']
  ② AgentTraj.load_prediction: The score path uses row directly as the same type mapping of envelope.

DECISIONS §16 envelope 5 field (usage field added); transcript / usage on both sides
typed dataclass, asdict is serialized into dict form and saved to predictions JSONL.

Why it is written independently: phase 5 and phase 4 have the same origin - data shape is a cross-project interface contract.
It is closer to the "source of online accidents" than the pure metric single test. Leave a separate file so that grep 'envelope' can hit it directly.

After DECISIONS §13, the equivalent coverage of "tool calling convention" + "decision extraction" in transcript has been moved to
[`agent_engine/tests/test_result_views.py`]; This file only retains the contract of evals itself: envelope
Fields ↔ Result origin + `_pin_trajectory` injection shape + `AgentTraj.load_prediction` behavior."""

from __future__ import annotations

import dataclasses

import pytest

from evals._ae_bridge import ArtifactEventEntry, Result, SpeakerEntry
from evals.api import Doc
from evals.tasks.agent_traj import AgentTraj, _pin_trajectory


# ---------- envelope schema homology ---------------------------------------------

def test_envelope_field_names_match_result_dataclass():
    """envelope must correspond 1:1 to the field name of agent_engine.Result (cli.py uses dataclasses.asdict)."""
    result_fields = {f.name for f in dataclasses.fields(Result)}
    expected = {"artifact", "transcript", "success", "warnings", "usage"}
    assert result_fields == expected


def test_dataclasses_asdict_matches_envelope_shape():
    """The dict written by `dataclasses.asdict(Result(...))` is the envelope expected by evals."""
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
    assert isinstance(envelope["transcript"][0], dict)  # asdict recursive flattening
    assert envelope["transcript"][0]["type"] == "speaker"
    assert isinstance(envelope["artifact"], dict)
    assert isinstance(envelope["warnings"], list)
    assert isinstance(envelope["success"], bool)
    assert isinstance(envelope["usage"], list)


# ---------- _pin_trajectory: Inject the complete contract ----------------------------

def test_pin_trajectory_writes_all_required_keys():
    """After pin, doc.metadata['trajectory'] must have 8 keys that phase 5 metric all depends on."""
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
    # Existing metadata fields are retained
    assert pinned.metadata["existing"] == "v"


def test_pin_trajectory_does_not_mutate_input_doc():
    """immutability: pin returns new Doc (dataclass replace), input doc.metadata unchanged."""
    doc = Doc(id="x", input="...", target=None, metadata={})
    envelope = {
        "transcript": [], "artifact": {},
        "warnings": [], "success": False, "usage": [],
    }
    _pin_trajectory(doc, envelope)
    assert "trajectory" not in doc.metadata


# ---------- AgentTraj.load_prediction --------------------------------

def test_load_prediction_translates_row_to_trajectory():
    """Row inner envelope field → doc.metadata['trajectory']; Response placeholder (output_type='none')."""
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
    assert response.text is None  # output_type='none', Response only takes place doc_id
    assert enriched.metadata["trajectory"]["tool_seq"] == ["cast_vote"]
    assert enriched.metadata["trajectory"]["success"] is True


def test_load_prediction_strict_on_missing_envelope_field():
    """Starting from §16, envelope strictly has 5 fields - load_prediction, missing fields will directly cause KeyError."""
    task = AgentTraj()
    doc = Doc(id="x", input="...", target=None, metadata={})
    with pytest.raises(KeyError):
        task.load_prediction(doc, {"id": "x"})


# ---------- fail-fast when run_fn is missing --------------------------------

def test_process_docs_requires_scenario_path():
    """Run path process_docs to doc without scenario_path must fail-fast."""
    def fake_run(_p):
        return dataclasses.asdict(Result(usage=[]))
    task = AgentTraj(run_fn=fake_run)
    doc_no_scenario = Doc(id="x", input="...", target=None, metadata={})
    with pytest.raises(ValueError, match="scenario_path"):
        task.process_docs([doc_no_scenario])
