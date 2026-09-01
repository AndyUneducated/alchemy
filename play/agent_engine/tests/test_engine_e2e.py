"""Engine.invoke end-to-end integration tests (no LLM / no VDB / no network).

Locks cross-module assembly chain: `Engine` → `Scenario.assemble` → `Discussion.run` →
`Agent.respond` → `Memory.build_messages` → `_client.chat` (fake) →
`tool_handler` → `ArtifactStore.dispatch` / `ToolTracer.record` → `Result`.

Strongest layer for "other module changes break this module" — any contract change
(signature, entry field names, event schema, warning signals) fails this test.

Design:
- No LLM; inject `FakeBackendClient.chat` into `agent_engine.agent._client`
  (`scenario._backend_client` same module object; SummaryMemory summarizer same patch)
- tmp_path scenarios cover happy path / require_tool retry /
  --save-result-json envelope round-trip / artifact tool paths
"""
from __future__ import annotations

import dataclasses
import json
import textwrap
from pathlib import Path

import pytest

from agent_engine import (
    ArtifactEventEntry,
    Engine,
    Result,
    Scenario,
    SpeakerEntry,
    ToolCallEntry,
    TopicEntry,
    TurnEntry,
)
from agent_engine import agent as _agent_mod

from ._fake_client import FakeBackendClient, Script


# ---------- fixtures ---------------------------------------------------

@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeBackendClient:
    """Fresh FakeBackendClient per test, patched onto `agent._client.chat`.

    Note: `scenario.py` top-level `from .agent import _client as _backend_client`
    and `agent._client` are the same module; patching `_client.chat` shares fake
    between Agent.respond and SummaryMemory.
    """
    fc = FakeBackendClient()
    monkeypatch.setattr(_agent_mod._client, "chat", fc.chat)
    return fc


def _write_scenario(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "scen.md"
    p.write_text(body, encoding="utf-8")
    return p


# ---------- basic happy path -------------------------------------------

def test_invoke_minimal_scenario_assembles_history_in_order(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """Two agents / one step minimal: history = topic → (turn + speaker) × N,
    Result fields = Engine.invoke defaults + no warning + usage from fake."""
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a-sys}
          - {name: B, role: member, prompt: b-sys}
        steps:
          - id: open
            who: [A, B]
            instruction: say hi
        ---
        topic body
    """))
    fake_client.script("A", Script(text="hello-A"))
    fake_client.script("B", Script(text="hello-B"))

    result = Engine(Scenario.from_yaml(str(scn))).invoke()

    assert isinstance(result, Result)
    assert result.success is True
    assert result.warnings == []
    assert result.artifact == {}
    types = [type(e).__name__ for e in result.transcript]
    assert types == [
        "TopicEntry",
        "TurnEntry", "SpeakerEntry",
        "TurnEntry", "SpeakerEntry",
    ]
    assert result.transcript[0] == TopicEntry(
        content="topic body", ts=result.transcript[0].ts,
    )
    assert isinstance(result.transcript[1], TurnEntry)
    assert result.transcript[1].content == "turn 1 of 2"
    assert isinstance(result.transcript[2], SpeakerEntry)
    assert result.transcript[2].speaker == "A"
    assert result.transcript[2].content == "hello-A"
    assert result.transcript[4].speaker == "B"  # type: ignore[union-attr]
    # usage once per agent (no summarizer), 2 total
    assert len(result.usage) == 2
    assert [u.caller for u in result.usage] == ["A", "B"]


def test_invoke_passes_per_agent_system_prompt_and_caller(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """FakeBackendClient.chat receives caller / system_prompt matching scenario declaration."""
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: prompt-for-A}
        steps:
          - who: [A]
            instruction: go
        ---
        t
    """))
    Engine(Scenario.from_yaml(str(scn))).invoke()
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["caller"] == "A"
    assert call["system_prompt"] == "prompt-for-A"


# ---------- require_tool retry + warning -------------------------------

def test_invoke_require_tool_miss_then_hit_succeeds(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """attempt 0 silent (require_tool miss) → nudge triggers attempt 1;
    fake calls propose_vote on attempt 1 → no warning."""
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: M, role: moderator, prompt: m}
        steps:
          - id: vote
            who: [M]
            require_tool: propose_vote
            max_retries: 2
            instruction: propose
        artifact:
          enabled: true
          initial_sections:
            - 决策
        ---
        t
    """))
    fake_client.script(
        "M",
        Script(text="silent"),
        Script(text="now-voting", tools=[
            {"name": "propose_vote", "args": {
                "question": "Q?", "options": ["yes", "no"],
            }},
        ]),
    )
    result = Engine(Scenario.from_yaml(str(scn))).invoke()
    assert result.warnings == []
    assert result.success is True
    # two attempts → two SpeakerEntry
    speakers = [e for e in result.transcript if isinstance(e, SpeakerEntry)]
    assert [s.content for s in speakers] == ["silent", "now-voting"]
    # propose_vote → artifact_event
    artifact_events = [
        e for e in result.transcript if isinstance(e, ArtifactEventEntry)
    ]
    assert [e.tool for e in artifact_events] == ["propose_vote"]


def test_invoke_require_tool_exhaust_retries_emits_warning(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """attempt 0 + attempt 1 both silent → warning in Result.warnings + success=False."""
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            require_tool: cast_vote
            max_retries: 1
            instruction: vote
        artifact:
          enabled: true
        ---
        t
    """))
    fake_client.script("A", Script(text="quiet-0"), Script(text="quiet-1"))
    result = Engine(Scenario.from_yaml(str(scn))).invoke()
    assert result.success is False
    assert len(result.warnings) == 1
    assert "skipped required tool 'cast_vote'" in result.warnings[0]


def test_invoke_require_tool_covers_tracer_event_for_retrieve_docs(
    tmp_path: Path, fake_client: FakeBackendClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DECISIONS §12: require_tool must observe tracer (non-artifact) events —
    `retrieve_docs` call recognized, no nudge."""
    scn_body = textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            require_tool: retrieve_docs
            instruction: search
        tools:
          - {name: retrieve_docs, vdb_dir: /tmp/vdb-fake}
        ---
        t
    """)
    scn = _write_scenario(tmp_path, scn_body)
    monkeypatch.setattr(
        "agent_engine.tools.retrieve_docs.handler",
        lambda **kwargs: json.dumps({
            "data": [], "meta": {
                "mode": kwargs.get("mode", "hybrid"),
                "reranked": False, "top_k": kwargs.get("top_k", 3),
            },
        }),
    )
    fake_client.script("A", Script(text="searched", tools=[
        {"name": "retrieve_docs", "args": {
            "query": "q", "vdb_dir": "/tmp/vdb-fake",
        }},
    ]))
    result = Engine(Scenario.from_yaml(str(scn))).invoke()
    assert result.warnings == [], "require_tool should accept tracer events (§12)"
    tool_calls = [
        e for e in result.transcript if isinstance(e, ToolCallEntry)
    ]
    assert [t.tool for t in tool_calls] == ["retrieve_docs"]
    # ToolTracer writes visible=False so memory does not feed back to LLM
    assert all(t.visible is False for t in tool_calls)


# ---------- artifact integration -------------------------------------------

def test_invoke_artifact_tool_call_persists_section_and_event(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            instruction: write
        artifact:
          enabled: true
          initial_sections:
            - 数据
        ---
        t
    """))
    fake_client.script("A", Script(text="done", tools=[
        {"name": "write_section", "args": {"name": "数据", "content": "hello"}},
    ]))
    result = Engine(Scenario.from_yaml(str(scn))).invoke()
    assert result.artifact == {"数据": "hello"}
    events = [e for e in result.transcript if isinstance(e, ArtifactEventEntry)]
    assert [e.tool for e in events] == ["write_section"]
    assert events[0].caller == "A"
    assert events[0].arguments == {"name": "数据", "content": "hello"}


def test_invoke_initial_artifact_seeds_sections(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """Engine.invoke(initial_artifact=...) prefills section outside ACL."""
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            instruction: read
        artifact:
          enabled: true
          initial_sections:
            - PRD
        ---
        t
    """))
    fake_client.script("A", Script(text="ok"))
    result = Engine(Scenario.from_yaml(str(scn))).invoke(
        initial_artifact={"PRD": "preloaded"},
    )
    assert result.artifact["PRD"] == "preloaded"


# ---------- IO：transcript / artifact / save_result_json --------------

def test_invoke_writes_transcript_and_artifact_files(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
        steps:
          - who: [A]
            instruction: write
        artifact:
          enabled: true
          initial_sections:
            - 结论
        ---
        t
    """))
    fake_client.script("A", Script(text="ok", tools=[
        {"name": "write_section", "args": {"name": "结论", "content": "X"}},
    ]))
    transcript_path = tmp_path / "out" / "trans.json"
    artifact_path = tmp_path / "out" / "art.md"
    Engine(Scenario.from_yaml(str(scn))).invoke(
        transcript_path=transcript_path,
        artifact_path=artifact_path,
    )
    assert transcript_path.exists()
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert isinstance(transcript, list)
    assert transcript[0]["type"] == "topic"
    assert any(e.get("type") == "speaker" for e in transcript)
    assert artifact_path.exists()
    md = artifact_path.read_text(encoding="utf-8")
    assert "## 结论" in md
    assert "X" in md


def test_invoke_result_envelope_roundtrips_via_save_result_json(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """In-memory Result == round-tripped Result via asdict → JSON → Result.load_json.

    Locks §11 (envelope SoT) + §13 (typed view) + §16 (strict from_dict):
    Engine.invoke Result must round-trip fully via envelope.
    """
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - {name: A, role: member, prompt: a}
          - {name: B, role: member, prompt: b}
        steps:
          - who: [A, B]
            instruction: talk
        artifact:
          enabled: true
        ---
        topic
    """))
    fake_client.script("A", Script(text="aa"))
    fake_client.script("B", Script(text="bb"))
    result = Engine(Scenario.from_yaml(str(scn))).invoke()

    envelope_path = tmp_path / "envelope.json"
    envelope = dataclasses.asdict(result)
    envelope_path.write_text(
        json.dumps(envelope, ensure_ascii=False), encoding="utf-8",
    )
    restored = Result.load_json(envelope_path)
    assert restored.artifact == result.artifact
    assert restored.success == result.success
    assert restored.warnings == result.warnings
    assert restored.usage == result.usage
    assert restored.transcript == result.transcript


# ---------- token usage / SummaryMemory -------------------------------

def test_summary_memory_triggers_summarizer_usage(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """SummaryMemory triggers extra summarizer LLM call when `stale_new >= max_recent`,
    and that call's TokenUsage lands in `Result.usage`.

    `memory.py::_run_summarizer` calls client.chat without `caller=`, so summarizer
    usage has `caller==""` (distinct from agent calls); if changed to `caller="_summarizer"`,
    this test exposes it for evals/agent_sft caller filter sync.
    """
    scn = _write_scenario(tmp_path, textwrap.dedent("""\
        ---
        agents:
          - name: A
            role: member
            prompt: a
            memory: {type: summary, max_recent: 1}
          - {name: B, role: member, prompt: b}
        steps:
          - who: [A, B, A, B, A]
            instruction: chat
        ---
        topic
    """))
    for spk in ("A", "B", "A", "B", "A"):
        fake_client.script(spk, Script(text=f"{spk}-reply"))

    result = Engine(Scenario.from_yaml(str(scn))).invoke()
    agent_usage = [u for u in result.usage if u.caller in {"A", "B"}]
    summarizer_usage = [u for u in result.usage if u.caller == ""]
    assert len(agent_usage) == 5
    assert len(summarizer_usage) >= 1, (
        "SummaryMemory should record at least one summarizer call once stale "
        "history exceeds max_recent"
    )


# ---------- public API smoke ------------------------------------------

def test_public_api_symbols_importable() -> None:
    """Public symbols from README Quick Start + DECISIONS §13/§14 import from `agent_engine`
    top level; delete/rename fails immediately."""
    import agent_engine as ae

    expected = {
        "Engine", "Scenario", "Result", "Callback",
        "ExpandedTurn", "ToolCall", "TurnView",
        "TopicEntry", "TurnEntry", "SpeakerEntry",
        "ToolCallEntry", "ArtifactEventEntry", "SummaryEntry",
        "TranscriptEntry", "TokenUsage",
    }
    missing = expected - set(dir(ae))
    assert not missing, f"public API symbols missing: {missing}"
    assert set(ae.__all__) == expected
