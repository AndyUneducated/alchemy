"""Run result + transcript / scenario interpretation views (schema interpretation SoT, DECISIONS §13 / §16).

`Result` is both the return type of `Engine.invoke()` and the envelope schema source
written by `cli.py --save-result-json` (`dataclasses.asdict`).

§13 establishes public typed views for transcript / scenario interpretation; §16 upgrades
transcript entries themselves from `list[dict]` to a `list[TranscriptEntry]` typed dataclass
union and adds per-LLM-call `usage: list[TokenUsage]` to the envelope. Old envelopes
(pre-§16, speaker entries without a type field, no usage field) are unreadable — the user
has rerun mining to produce the new schema.

Public surface:
- `Result` envelope dataclass (5 fields: artifact / transcript / success / warnings / usage)
- `TranscriptEntry` union: `TopicEntry | TurnEntry | SpeakerEntry | ToolCallEntry |
  ArtifactEventEntry | SummaryEntry`
- `ToolCall` / `TurnView` typed views (§13); `TokenUsage` per-LLM-call detail (§16)
- `Result.from_dict / load_json` envelope ↔ Result IO (missing fields → KeyError, no downgrade)
- `Result.tool_calls() / turns() / speakers() / find_finalize_decision()` interpretation views

Design references: OpenAI Agents SDK `RunResult.new_items` (typed `RunItem` union) /
Anthropic `Message.content[ContentBlock]` (typed block union + `usage`) /
inspect_ai `ChatMessage` (typed dispatch).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

ToolCallKind = Literal["artifact", "tracer"]


# =========================================================================
# Transcript entry typed dataclass union (§16)
# =========================================================================
#
# Each transcript entry is a frozen dataclass with an explicit `type` Literal tag.
# When `dataclasses.asdict` serializes to dict, the `type` field is preserved; envelope
# JSON reverse-dispatches to the correct typed class via `_entry_from_dict`.

@dataclass(frozen=True)
class TopicEntry:
    """Topic injected at discussion start (scenario body text); first entry in every transcript."""
    type: Literal["topic"] = "topic"
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class TurnEntry:
    """`<turn X of N>` marker used for turn segmentation. One per expanded (agent, step) turn."""
    type: Literal["turn"] = "turn"
    content: str = ""        # "turn N of M"
    ts: float = 0.0


@dataclass(frozen=True)
class SpeakerEntry:
    """Single LLM reply from an agent. §16 adds explicit `type="speaker"` tag (aligned with other entries)."""
    type: Literal["speaker"] = "speaker"
    speaker: str = ""
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class ToolCallEntry:
    """Non-artifact tool call record (written by `ToolTracer`), e.g. `retrieve_docs`."""
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
    """Artifact tool call record (written by `ArtifactStore`), e.g. `write_section / cast_vote / finalize_artifact`."""
    type: Literal["artifact_event"] = "artifact_event"
    tool: str = ""
    caller: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    ts: float = 0.0


@dataclass(frozen=True)
class SummaryEntry:
    """Placeholder when SummaryMemory merges history while projecting LLM messages; not in real transcript (internal to build_messages only)."""
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
    """Single envelope dict entry → typed `TranscriptEntry`.

    Strict dispatch: missing `type` or unregistered `type` raises directly; no fallback to
    implicit rules like "if speaker field exists, treat as speaker" (§16 strict schema).
    """
    entry_type = d["type"]
    cls = _ENTRY_BY_TYPE[entry_type]
    return cls(**d)


# =========================================================================
# Token usage (§16)
# =========================================================================

@dataclass(frozen=True)
class TokenUsage:
    """Token / latency detail for a single LLM call.

    Per-call detail rather than aggregate: consumers can `sum(u.input_tokens for u in result.usage)`
    for totals while still slicing by `model` / `caller` (per-agent / per-model cost analysis).

    When streaming calls provide no usage (some backends lack fields on the final stream chunk),
    fill 0; `play/evals/metrics/efficiency.py` degrades cost calculation to 0.0, consistent
    with historical behavior.
    """
    model: str
    caller: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    duration_ms: int = 0
    ts: float = 0.0


# =========================================================================
# §13 typed views (minor adjustments for typed entries)
# =========================================================================

@dataclass(frozen=True)
class ToolCall:
    """Typed view of one tool call in the transcript.

    `kind="artifact"` comes from `ArtifactStore` `artifact_event` entries (six artifact
    tools); `kind="tracer"` comes from `ToolTracer` `tool_call` entries (non-artifact tools
    like `retrieve_docs`). The two event shapes differ, but this layer normalizes to typed
    `(tool, caller, arguments)` for consumers.
    """

    tool: str
    caller: str
    arguments: dict[str, Any]
    kind: ToolCallKind
    ts: float | None = None


@dataclass(frozen=True)
class TurnView:
    """Typed view of one turn segment in the transcript (all entries between `<turn X of N>` markers).

    `turn_idx` is 1-based, aligned with the `turn N of M` marker written by `Discussion.run`.
    `start_offset` is the 0-based global index of the segment's first entry in the original
    `transcript` list — `play/agent_sft/data/extractor.py` needs it to map local segment idx
    back to global transcript positions for context slicing; other consumers may ignore it.
    """

    turn_idx: int
    start_offset: int
    entries: tuple[TranscriptEntry, ...]

    def attempts(self, agent: str) -> list[list[TranscriptEntry]]:
        """Split segment into attempts by `agent` SpeakerEntry — each speaker entry starts a new attempt.

        Aligned with `Discussion._run_turn` retry loop: first SpeakerEntry is attempt 0;
        after require_tool miss triggers nudge, second SpeakerEntry is attempt 1, and so on.

        Conventions:
          - A turn usually belongs to one agent, so "other speakers" are rare; if they appear
            they are absorbed as trailing events of the previous attempt without affecting counts
          - Segment with no speaker entries → 0 attempts (caller was completely silent)
          - Events before the first speaker are dropped
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
        """Tool calls within the segment (`ToolCallEntry` + `ArtifactEventEntry`); same convention as `Result.tool_calls`."""
        return _entries_to_tool_calls(self.entries)


def _entry_to_tool_call(entry: TranscriptEntry) -> ToolCall | None:
    """Normalize typed `TranscriptEntry` to `ToolCall`; returns None for non-tool events."""
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

        §16 onward is strict: any missing field raises `KeyError` directly. Old envelopes
        (pre-§16) are unreadable; rerun mining to rebuild historical data first.
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
        """Load Result from a file written by `cli.py --save-result-json`."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---- transcript views ----------------------------------------------

    def tool_calls(self) -> list[ToolCall]:
        """All tool calls in transcript merged in time order (`ToolCallEntry` ∪ `ArtifactEventEntry`)."""
        return _entries_to_tool_calls(self.transcript)

    def turns(self) -> list[TurnView]:
        """Segment by `TurnEntry` markers; `turn_idx` starts at 1; markers themselves dropped; pre-turn debris dropped.

        Engine appends a turn marker before each expanded (agent, step) turn (`Discussion.run`);
        segment count = total turn count; segment entries are tupleized for immutability.
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
        """Set of speaker names that actually spoke in the transcript (deduplicated)."""
        return {e.speaker for e in self.transcript if isinstance(e, SpeakerEntry)}

    def find_finalize_decision(self) -> str | None:
        """Scan tool calls for the last `finalize_artifact`; strip `arguments['decision']`.

        `finalize_artifact` is designed to be idempotent (re-entry returns error), so in theory
        at most one successful call exists in the transcript; if multiple appear (edge case),
        return the **last** decision — closest to "sealed state" semantics.
        Returns None and continues searching when decision is missing / empty / non-str.
        """
        decision: str | None = None
        for tc in self.tool_calls():
            if tc.tool != "finalize_artifact":
                continue
            d = tc.arguments.get("decision")
            if isinstance(d, str) and d.strip():
                decision = d.strip()
        return decision
