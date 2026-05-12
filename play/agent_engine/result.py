"""Run result + transcript / scenario 解读视图（schema 解读权 SoT，DECISIONS §13 / §16）.

`Result` 既是 `Engine.invoke()` 的返回类型，又是 `cli.py --save-result-json` 写出的
envelope schema 来源（`dataclasses.asdict`）。

§13 立"transcript / scenario 解读 typed view 公开化"；§16 接着把 transcript entry 自身
从 `list[dict]` 升级到 `list[TranscriptEntry]` typed dataclass union，并给 envelope 加
逐次 LLM 调用的 `usage: list[TokenUsage]`. 老 envelope（pre-§16，speaker entry 无 type
字段、无 usage 字段）一律不可读 —— 用户已 rerun mining 落新 schema.

公开面：
- `Result` envelope dataclass（5 字段：artifact / transcript / success / warnings / usage）
- `TranscriptEntry` union：`TopicEntry | TurnEntry | SpeakerEntry | ToolCallEntry |
  ArtifactEventEntry | SummaryEntry`
- `ToolCall` / `TurnView` typed view（§13）；`TokenUsage` per-LLM-call 明细（§16）
- `Result.from_dict / load_json` envelope ↔ Result IO（缺字段 KeyError，无降级）
- `Result.tool_calls() / turns() / speakers() / find_finalize_decision()` 解读视图

设计参照：OpenAI Agents SDK `RunResult.new_items`（typed `RunItem` union）/
Anthropic `Message.content[ContentBlock]`（typed block union + `usage`）/
inspect_ai `ChatMessage`（typed dispatch）.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

ToolCallKind = Literal["artifact", "tracer"]


# =========================================================================
# Transcript entry typed dataclass union（§16）
# =========================================================================
#
# 每条 transcript entry 都是 frozen dataclass 含显式 `type` Literal tag.
# `dataclasses.asdict` 序列化为 dict 时 `type` 字段保留，envelope JSON 通过 `type`
# 反向 dispatch 回正确的 typed class（`_entry_from_dict`）.

@dataclass(frozen=True)
class TopicEntry:
    """讨论起首注入的 topic（scenario body 文本），每个 transcript 第一条."""
    type: Literal["topic"] = "topic"
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class TurnEntry:
    """`<turn X of N>` marker，切 turn 用. 每个 (agent, step) 展开一条."""
    type: Literal["turn"] = "turn"
    content: str = ""        # "turn N of M"
    ts: float = 0.0


@dataclass(frozen=True)
class SpeakerEntry:
    """agent 单次 LLM 回复. §16 起显式带 `type="speaker"` tag（与其它 entry 体例对齐）."""
    type: Literal["speaker"] = "speaker"
    speaker: str = ""
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class ToolCallEntry:
    """非 artifact 工具调用记录（`ToolTracer` 写入），如 `retrieve_docs`."""
    type: Literal["tool_call"] = "tool_call"
    caller: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    ok: bool = True
    visible: bool = True
    ts: float = 0.0


@dataclass(frozen=True)
class ArtifactEventEntry:
    """artifact 工具调用记录（`ArtifactStore` 写入），如 `write_section / cast_vote / finalize_artifact`."""
    type: Literal["artifact_event"] = "artifact_event"
    tool: str = ""
    caller: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class SummaryEntry:
    """SummaryMemory 投影 LLM messages 时合并历史的占位，不进真 transcript（仅 build_messages 内部）."""
    type: Literal["summary"] = "summary"
    content: str = ""


TranscriptEntry = Union[
    TopicEntry, TurnEntry, SpeakerEntry,
    ToolCallEntry, ArtifactEventEntry, SummaryEntry,
]


_ENTRY_BY_TYPE: dict[str, type] = {
    "topic": TopicEntry,
    "turn": TurnEntry,
    "speaker": SpeakerEntry,
    "tool_call": ToolCallEntry,
    "artifact_event": ArtifactEventEntry,
    "summary": SummaryEntry,
}


def _entry_from_dict(d: dict) -> TranscriptEntry:
    """envelope dict 的单条 entry → typed `TranscriptEntry`.

    严格 dispatch：缺 `type` 字段或 `type` 未注册都直接 raise；不再 fallback 到
    "speaker 字段存在则视作 speaker" 这类隐式规则（§16 立 strict schema）.
    """
    entry_type = d["type"]
    cls = _ENTRY_BY_TYPE[entry_type]
    return cls(**d)


# =========================================================================
# Token usage（§16）
# =========================================================================

@dataclass(frozen=True)
class TokenUsage:
    """单次 LLM 调用的 token / 时延明细.

    逐次明细而非 aggregate：消费者用 `sum(u.input_tokens for u in result.usage)`
    自算 total 的同时，可按 `model` / `caller` 切分（per-agent / per-model 成本分析）.

    流式调用未提供 usage 时（某些 backend stream 尾 chunk 缺字段）填 0；
    `play/evals/metrics/efficiency.py` 在 cost 计算时降级到 0.0，与历史口径一致.
    """
    model: str
    caller: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    duration_ms: int = 0
    ts: float = 0.0


# =========================================================================
# §13 typed view（小调整以接 typed entry）
# =========================================================================

@dataclass(frozen=True)
class ToolCall:
    """transcript 中一次工具调用的 typed 视图.

    `kind="artifact"` 来自 `ArtifactStore` 写入的 `artifact_event`（六个 artifact
    工具）；`kind="tracer"` 来自 `ToolTracer` 写入的 `tool_call`（非 artifact 工具
    如 `retrieve_docs`）。两类事件的字段不同，但这一层规约让消费者只看 typed
    `(tool, caller, arguments)`。
    """

    tool: str
    caller: str
    arguments: dict[str, Any]
    kind: ToolCallKind
    ts: float | None = None


@dataclass(frozen=True)
class TurnView:
    """transcript 中一段 turn（`<turn X of N>` marker 之间的所有 entry）的 typed 视图.

    `turn_idx` 1-based，与 `Discussion.run` 写入的 `turn N of M` marker 对齐。
    `start_offset` 是该段第一个 entry 在原 `transcript` 列表里的 0-based 全局索引——
    `play/agent_sft/data/extractor.py` 需要它把段内 local idx 映射回 transcript
    全局位置以切 context；其它消费者可忽略。
    """

    turn_idx: int
    start_offset: int
    entries: tuple[TranscriptEntry, ...]

    def attempts(self, agent: str) -> list[list[TranscriptEntry]]:
        """段内按 `agent` 的 SpeakerEntry 入栈切 attempt——每次 speaker 标志一次新 attempt 的开始.

        与 `Discussion._run_turn` 的 retry 循环对齐：第一个 SpeakerEntry 是 attempt 0，
        require_tool 未满足触发 nudge 后第二个 SpeakerEntry 是 attempt 1，依此类推。

        规约：
          - 一个 turn 通常只属于一个 agent，所以"其他 speaker"很少出现；万一出现
            会被作为前一 attempt 的 trailing 事件吞掉，不影响计数
          - 没有任何 speaker entry 的段 → 0 attempts（caller 完全沉默）
          - speaker 之前的事件会被丢弃
        """
        out: list[list[TranscriptEntry]] = []
        current: list[TranscriptEntry] | None = None
        for entry in self.entries:
            if isinstance(entry, SpeakerEntry) and entry.speaker == agent:
                if current is not None:
                    out.append(current)
                current = []
            elif current is not None:
                current.append(entry)
        if current is not None:
            out.append(current)
        return out

    def tool_calls(self) -> list[ToolCall]:
        """段内的工具调用（`ToolCallEntry` + `ArtifactEventEntry`），同 `Result.tool_calls` 的规约."""
        return _entries_to_tool_calls(self.entries)


def _entry_to_tool_call(entry: TranscriptEntry) -> ToolCall | None:
    """把 typed `TranscriptEntry` 规约成 `ToolCall`；非工具事件返 None."""
    if isinstance(entry, ArtifactEventEntry):
        return ToolCall(
            tool=entry.tool, caller=entry.caller,
            arguments=dict(entry.arguments), kind="artifact", ts=entry.ts,
        )
    if isinstance(entry, ToolCallEntry):
        return ToolCall(
            tool=entry.tool, caller=entry.caller,
            arguments=dict(entry.arguments), kind="tracer", ts=entry.ts,
        )
    return None


def _entries_to_tool_calls(
    entries: tuple[TranscriptEntry, ...] | list[TranscriptEntry],
) -> list[ToolCall]:
    out: list[ToolCall] = []
    for entry in entries:
        tc = _entry_to_tool_call(entry)
        if tc is not None:
            out.append(tc)
    return out


# =========================================================================
# Result envelope
# =========================================================================

@dataclass
class Result:
    artifact: dict[str, str] = field(default_factory=dict)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    success: bool = True
    warnings: list[str] = field(default_factory=list)
    usage: list[TokenUsage] = field(default_factory=list)

    # ---- IO ------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        """envelope dict → Result.

        §16 起严格：缺任何字段直接 `KeyError`. 老 envelope（pre-§16）不可读；如需消费
        历史数据先重跑 mining 重建.
        """
        return cls(
            artifact=dict(data["artifact"]),
            transcript=[_entry_from_dict(e) for e in data["transcript"]],
            success=bool(data["success"]),
            warnings=list(data["warnings"]),
            usage=[TokenUsage(**u) for u in data["usage"]],
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "Result":
        """从 `cli.py --save-result-json` 写出的文件加载 Result."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- transcript views ----------------------------------------------

    def tool_calls(self) -> list[ToolCall]:
        """transcript 内所有工具调用按时间顺序合并（`ToolCallEntry` ∪ `ArtifactEventEntry`）."""
        return _entries_to_tool_calls(self.transcript)

    def turns(self) -> list[TurnView]:
        """按 `TurnEntry` marker 切段；`turn_idx` 从 1 起，marker 自身丢，turn 前杂物丢.

        引擎在每个 (agent, step) 展开 turn 前 append 一个 turn marker
        (`Discussion.run`)；段数 = 总 turn 数；segment 元素 tuple 化保持
        immutability。
        """
        out: list[TurnView] = []
        current: list[TranscriptEntry] = []
        start_offset = -1
        turn_idx = 0
        started = False
        for i, entry in enumerate(self.transcript):
            if isinstance(entry, TurnEntry):
                if started:
                    out.append(TurnView(
                        turn_idx=turn_idx,
                        start_offset=start_offset,
                        entries=tuple(current),
                    ))
                turn_idx += 1
                start_offset = i + 1
                current = []
                started = True
                continue
            if started:
                current.append(entry)
        if started:
            out.append(TurnView(
                turn_idx=turn_idx,
                start_offset=start_offset,
                entries=tuple(current),
            ))
        return out

    def speakers(self) -> set[str]:
        """transcript 里实际说过话的 speaker 名集合（去重）."""
        return {e.speaker for e in self.transcript if isinstance(e, SpeakerEntry)}

    def find_finalize_decision(self) -> str | None:
        """扫工具调用找最后一次 `finalize_artifact`，从 arguments['decision'] 取并 strip.

        `finalize_artifact` 设计幂等（重入返 error），理论上 transcript 内最多一次成功调用；
        若仍出现多次（如边界事故），返**最后**一次的 decision 更贴近"封板状态"语义.
        decision 缺失 / 空 / 非 str 时该次返 None，继续往前找。
        """
        decision: str | None = None
        for tc in self.tool_calls():
            if tc.tool != "finalize_artifact":
                continue
            d = tc.arguments.get("decision")
            if isinstance(d, str) and d.strip():
                decision = d.strip()
        return decision
