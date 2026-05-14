"""Memory 投影规则单测（DECISIONS §2 / §4 / §5）.

锁 `memory.py` 的对外契约——任一 backend client / agent.respond / discussion
循环里对 history 形态做的假设都依赖这些投影规则：

  - `FullHistory`：全量；speaker == owner 投 assistant，其它投 `<message from>`
  - `WindowMemory`：保留所有 pinned (topic/turn/artifact_event) + 最近 N 条 speech
  - `SummaryMemory`：stale_new >= max_recent 时调用一次 client.chat
    summarizer；产物以 `<summary>` block 注入消息流
  - `ToolCallEntry.visible=False` 一律被投影跳过
  - 各 entry 类型对应的 wrapping tag（topic / turn / artifact_event / summary
    / tool_call）字节稳定，evals / agent_sft 离线回放靠这些 tag 切段

任何对 entry 类型 / 投影逻辑 / pinned 集合的破坏性改动都会让本测试失败.
"""
from __future__ import annotations

from typing import Any

from agent_engine.memory import (
    DEFAULT_SUMMARIZE_INSTRUCTION,
    DEFAULT_SUMMARIZER_PROMPT,
    FullHistory,
    SummaryMemory,
    WindowMemory,
)
from agent_engine.result import (
    ArtifactEventEntry,
    SpeakerEntry,
    SummaryEntry,
    TokenUsage,
    ToolCallEntry,
    TopicEntry,
    TurnEntry,
)


# ---------- helpers ----------------------------------------------------

def _spk(name: str, text: str) -> SpeakerEntry:
    return SpeakerEntry(speaker=name, content=text)


def _topic(text: str = "T") -> TopicEntry:
    return TopicEntry(content=text)


def _turn(text: str = "turn 1 of 1") -> TurnEntry:
    return TurnEntry(content=text)


def _art(tool: str, caller: str) -> ArtifactEventEntry:
    return ArtifactEventEntry(tool=tool, caller=caller, content=f"{caller} did {tool}")


class _RecordingClient:
    """`SummaryMemory` 注入用 stub. 收集每次 chat() 调用 + 返回固定文本."""

    def __init__(self, reply: str = "fake-summary") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def chat(
        self, model: str, *, system_prompt: str, messages: list[dict],
        temperature: float, max_tokens: int, stream: bool,
        tools: list[dict] | None = None, **kw: Any,
    ) -> tuple[str, TokenUsage]:
        self.calls.append({
            "model": model,
            "system_prompt": system_prompt,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "tools": tools,
        })
        return self.reply, TokenUsage(model=model, caller="", input_tokens=1, output_tokens=1)


# ---------- FullHistory -----------------------------------------------

def test_full_history_owner_speech_becomes_assistant_others_user():
    """owner 自己的 SpeakerEntry → role=assistant；他人 → role=user 包 <message from>."""
    history = [_spk("A", "hi"), _spk("B", "ho")]
    msgs = FullHistory().build_messages(history, owner="A")
    assert msgs == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": '<message from="B">\nho\n</message>'},
    ]


def test_full_history_topic_turn_artifact_event_wrap_with_tag():
    history = [
        _topic("讨论话题"),
        _turn("turn 1 of 2"),
        _spk("A", "hi"),
        _art("write_section", "A"),
    ]
    msgs = FullHistory().build_messages(history, owner="A")
    assert msgs[0] == {"role": "user", "content": "<topic>\n讨论话题\n</topic>"}
    assert msgs[1] == {"role": "user", "content": "<turn>\nturn 1 of 2\n</turn>"}
    assert msgs[2] == {"role": "assistant", "content": "hi"}
    assert msgs[3] == {
        "role": "user", "content": "<artifact_event>\nA did write_section\n</artifact_event>",
    }


def test_full_history_skips_tool_call_with_visible_false():
    """`ToolTracer` 写入的 ToolCallEntry 默认 visible=False —— memory 必须跳过，
    否则会把内部工具调用回喂给 LLM 污染 context."""
    invisible = ToolCallEntry(
        caller="A", tool="retrieve_docs", arguments={"q": "x"},
        result="...", visible=False,
    )
    visible = ToolCallEntry(
        caller="A", tool="x", arguments={}, result="r", visible=True,
    )
    history = [_spk("A", "hi"), invisible, visible]
    msgs = FullHistory().build_messages(history, owner="A")
    assert len(msgs) == 2  # invisible 被跳过
    assert msgs[1] == {"role": "user", "content": "<tool_call>\nr\n</tool_call>"}


def test_full_history_summary_entry_wraps_with_tag():
    msgs = FullHistory().build_messages([SummaryEntry(content="...")], owner="A")
    assert msgs == [{"role": "user", "content": "<summary>\n...\n</summary>"}]


# ---------- WindowMemory ----------------------------------------------

def test_window_memory_keeps_pinned_and_last_n_speech():
    """topic / turn / artifact_event 全保留；speech 只留最近 max_recent 条."""
    history = [
        _topic("话题"),
        _turn("turn 1 of 5"),
        _spk("A", "1"),
        _turn("turn 2 of 5"),
        _spk("B", "2"),
        _turn("turn 3 of 5"),
        _spk("A", "3"),
        _art("write_section", "A"),
        _turn("turn 4 of 5"),
        _spk("B", "4"),
        _turn("turn 5 of 5"),
        _spk("A", "5"),
    ]
    msgs = WindowMemory(max_recent=2).build_messages(history, owner="A")
    contents = [m["content"] for m in msgs]
    # pinned: topic + 5 turn marker + artifact_event = 7
    # speech: 最近 2 条 (B's "4", A's "5") = 2
    assert sum("<topic>" in c for c in contents) == 1
    assert sum("<turn>" in c for c in contents) == 5
    assert sum("<artifact_event>" in c for c in contents) == 1
    # speech: 只剩 "4" / "5"，老的 "1" "2" "3" 不见
    speech_msgs = [
        c for c in contents
        if c in {"5", '<message from="B">\n4\n</message>'}
    ]
    assert len(speech_msgs) == 2
    assert all(c not in contents for c in {"1", "3"})
    assert not any('<message from="B">\n2\n</message>' == c for c in contents)


def test_window_memory_speech_fewer_than_window_keeps_all():
    history = [_topic(), _spk("A", "x"), _spk("B", "y")]
    msgs = WindowMemory(max_recent=10).build_messages(history, owner="A")
    assert len(msgs) == 3  # 全保留


# ---------- SummaryMemory ---------------------------------------------

def test_summary_memory_no_trigger_when_speech_under_max_recent():
    """speech 数 <= max_recent → 走全量分支，不调 summarizer."""
    client = _RecordingClient()
    mem = SummaryMemory(
        max_recent=3, client=client, summary_model="m",
        summary_max_tokens=100, summary_temperature=0.0,
    )
    history = [_topic(), _spk("A", "1"), _spk("B", "2")]
    msgs = mem.build_messages(history, owner="A")
    assert client.calls == []
    assert mem.drain_usage() == []
    assert len(msgs) == 3


def test_summary_memory_triggers_summarizer_when_stale_reaches_threshold():
    """stale_new >= max_recent → 调用一次 summarizer，<summary> block 注入消息流."""
    client = _RecordingClient(reply="MERGED")
    mem = SummaryMemory(
        max_recent=2, client=client, summary_model="model-x",
        summary_max_tokens=99, summary_temperature=0.5,
    )
    history = [
        _topic("topic"),
        _spk("A", "1"), _spk("B", "2"), _spk("A", "3"),
        _spk("B", "4"), _spk("A", "5"),
    ]
    msgs = mem.build_messages(history, owner="A")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "model-x"
    assert call["system_prompt"] == DEFAULT_SUMMARIZER_PROMPT
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 99
    # 最后一条 messages 是 summarize instruction
    assert call["messages"][-1]["content"] == (
        f"<instruction>\n{DEFAULT_SUMMARIZE_INSTRUCTION}\n</instruction>"
    )
    # build_messages 输出含 <summary>MERGED</summary>
    contents = [m["content"] for m in msgs]
    assert any("<summary>\nMERGED\n</summary>" == c for c in contents)


def test_summary_memory_drain_usage_returns_summarizer_token_usage_once():
    """summarizer 产生的 TokenUsage 通过 drain_usage 暴露给 Agent.respond；
    drain 后清空，与 ToolTracer.drain 同语义."""
    client = _RecordingClient()
    mem = SummaryMemory(
        max_recent=1, client=client, summary_model="m",
        summary_max_tokens=10, summary_temperature=0.0,
    )
    history = [_spk("A", "1"), _spk("B", "2"), _spk("A", "3")]
    mem.build_messages(history, owner="A")
    first = mem.drain_usage()
    second = mem.drain_usage()
    assert len(first) == 1
    assert isinstance(first[0], TokenUsage)
    assert second == []


def test_summary_memory_reuses_previous_summary_with_previous_summary_block():
    """二次 build_messages 仍 stale → summarizer 收到 <previous_summary> block，
    增量折叠语义（详见 memory.py::_run_summarizer）."""
    client = _RecordingClient(reply="MERGED2")
    mem = SummaryMemory(
        max_recent=1, client=client, summary_model="m",
        summary_max_tokens=10, summary_temperature=0.0,
    )
    h1 = [_spk("A", "1"), _spk("B", "2"), _spk("A", "3")]
    mem.build_messages(h1, owner="A")
    # 第二次扩张，让 stale_new 再次触发
    h2 = h1 + [_spk("B", "4"), _spk("A", "5")]
    mem.build_messages(h2, owner="A")
    second = client.calls[-1]
    assert second["messages"][0]["content"].startswith("<previous_summary>")


# ---------- 跨投影规则的小不变量 ---------------------------------------

def test_owner_attribution_consistent_across_memory_strategies():
    """同一 history 三种 memory 投影下，"owner 视角识别"都成立：
    A 在自己的 messages 里看到 'mine-msg' 是 assistant，看不到 from='A' wrap."""
    history = [_spk("A", "mine-msg"), _spk("B", "other")]
    full = FullHistory().build_messages(history, owner="A")
    window = WindowMemory(max_recent=5).build_messages(history, owner="A")
    summary = SummaryMemory(
        max_recent=10, client=_RecordingClient(), summary_model="m",
        summary_max_tokens=10, summary_temperature=0.0,
    ).build_messages(history, owner="A")
    for tag, msgs in [("full", full), ("window", window), ("summary", summary)]:
        roles = {m["role"] for m in msgs}
        assert "assistant" in roles, tag
        assert {"mine-msg"} <= {m["content"] for m in msgs}, tag
        assert not any('<message from="A"' in m["content"] for m in msgs), tag
