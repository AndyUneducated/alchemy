"""Engine.invoke 端到端集成测试（无 LLM / 无 VDB / 无网络）.

锁住跨模块装配链：`Engine` → `Scenario.assemble` → `Discussion.run` →
`Agent.respond` → `Memory.build_messages` → `_client.chat`（fake）→
`tool_handler` → `ArtifactStore.dispatch` / `ToolTracer.record` → `Result`.

这是覆盖"其它模块改动让本模块不可用"最有力的一层——任一模块改了对外契约（函数
签名、entry 字段名、event schema、warning 信号）都会让本测试失败。

设计原则：
- 不跑 LLM，用 `FakeBackendClient.chat` 注入到 `agent_engine.agent._client` 上
  （`scenario._backend_client` 引用同一模块对象，SummaryMemory 内部 summarizer
   也走同一 patch）
- scenario 用 `tmp_path` 写小文件，覆盖 happy path / require_tool 重试 /
  --save-result-json envelope round-trip / artifact 工具触达 4 个独立形态
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
    """每个测试一份新 FakeBackendClient，patch 到 `agent._client.chat` 上.

    注意：`scenario.py` 顶层 `from .agent import _client as _backend_client`
    与 `agent._client` 指向同一 module；patch `_client.chat` 后 Agent.respond +
    SummaryMemory 共用此 fake。
    """
    fc = FakeBackendClient()
    monkeypatch.setattr(_agent_mod._client, "chat", fc.chat)
    return fc


def _write_scenario(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "scen.md"
    p.write_text(body, encoding="utf-8")
    return p


# ---------- 基础 happy path -------------------------------------------

def test_invoke_minimal_scenario_assembles_history_in_order(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """两 agent / 一 step 最简场景：history 形态 = topic → (turn + speaker) × N，
    Result 字段 = Engine.invoke 默认值 + 无 warning + usage 来自 fake."""
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
    # usage 每 agent 1 次（无 summarizer），共 2 条
    assert len(result.usage) == 2
    assert [u.caller for u in result.usage] == ["A", "B"]


def test_invoke_passes_per_agent_system_prompt_and_caller(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """FakeBackendClient.chat 收到的 caller / system_prompt 与 scenario 声明一致."""
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


# ---------- require_tool 重试 + warning -------------------------------

def test_invoke_require_tool_miss_then_hit_succeeds(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """attempt 0 沉默（require_tool 未命中）→ nudge 触发 attempt 1，
    fake 在 attempt 1 调用 propose_vote → 不产生 warning."""
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
    # 两次 attempt → 两条 SpeakerEntry
    speakers = [e for e in result.transcript if isinstance(e, SpeakerEntry)]
    assert [s.content for s in speakers] == ["silent", "now-voting"]
    # propose_vote 落 artifact_event
    artifact_events = [
        e for e in result.transcript if isinstance(e, ArtifactEventEntry)
    ]
    assert [e.tool for e in artifact_events] == ["propose_vote"]


def test_invoke_require_tool_exhaust_retries_emits_warning(
    tmp_path: Path, fake_client: FakeBackendClient,
) -> None:
    """attempt 0 + attempt 1 都沉默 → warning 落 Result.warnings + success=False."""
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
    """DECISIONS §12 锁：require_tool 必须同时观测 tracer (非 artifact) 工具的事件——
    `retrieve_docs` 调用应被识别，不再触发 nudge."""
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
    # ToolTracer 写入 visible=False，确保 memory 不会回喂给 LLM
    assert all(t.visible is False for t in tool_calls)


# ---------- artifact 集成 -------------------------------------------

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
    """Engine.invoke(initial_artifact=...) 在 ACL 之外预填 section."""
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

    这把 §11 (envelope SoT) + §13 (typed view) + §16 (strict from_dict) 三层
    契约扣在一起：Engine.invoke 输出的 Result 必须能通过 envelope 完整还原.
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
    """SummaryMemory 在 `stale_new >= max_recent` 时会触发一次额外的 summarizer
    LLM 调用，且该调用的 TokenUsage 也落进 `Result.usage`.

    实测口径：`memory.py::_run_summarizer` 调 client.chat 时不传 `caller=`，
    所以 summarizer usage 的 `caller==""`（与 agent 调用区分开）；若有人改成
    显式传 `caller="_summarizer"`，本测试自动暴露，提醒同步更新 evals/agent_sft
    的 caller 过滤逻辑.
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
    """README §快速开始 + DECISIONS §13/§14 列出的公开符号都能从 `agent_engine`
    顶层 import；如有人删 / 改名要立刻发现."""
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
