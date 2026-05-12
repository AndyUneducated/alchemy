from __future__ import annotations

from typing import Iterable

from .result import (
    ArtifactEventEntry,
    SpeakerEntry,
    SummaryEntry,
    TokenUsage,
    TopicEntry,
    ToolCallEntry,
    TranscriptEntry,
    TurnEntry,
)


def _is_pinned(entry: TranscriptEntry) -> bool:
    """`topic` / `turn` / `artifact_event` 在 window/summary 截断后仍要保留."""
    return isinstance(entry, (TopicEntry, TurnEntry, ArtifactEventEntry))


def _is_visible(entry: TranscriptEntry) -> bool:
    """ToolCallEntry 默认 visible=False（tracer 内部记录），其它 entry 一律可见."""
    if isinstance(entry, ToolCallEntry):
        return entry.visible
    return True


def _render(entries: Iterable[TranscriptEntry], owner: str) -> list[dict]:
    """typed entry → LLM messages 的 `[{role, content}, ...]` 投影."""
    messages: list[dict] = []
    for entry in entries:
        if not _is_visible(entry):
            continue
        if isinstance(entry, SpeakerEntry):
            if entry.speaker == owner:
                messages.append({"role": "assistant", "content": entry.content})
            else:
                content = (
                    f'<message from="{entry.speaker}">\n'
                    f'{entry.content}\n</message>'
                )
                messages.append({"role": "user", "content": content})
            continue
        if isinstance(entry, TopicEntry):
            tag = "topic"
            content = entry.content
        elif isinstance(entry, TurnEntry):
            tag = "turn"
            content = entry.content
        elif isinstance(entry, ArtifactEventEntry):
            tag = "artifact_event"
            content = entry.content
        elif isinstance(entry, SummaryEntry):
            tag = "summary"
            content = entry.content
        elif isinstance(entry, ToolCallEntry):
            tag = "tool_call"
            content = entry.result
        else:  # pragma: no cover - exhaustive union
            continue
        messages.append({
            "role": "user",
            "content": f"<{tag}>\n{content}\n</{tag}>",
        })
    return messages


class ConversationMemory:
    def build_messages(
        self, history: list[TranscriptEntry], owner: str,
    ) -> list[dict]:
        raise NotImplementedError

    def drain_usage(self) -> list[TokenUsage]:
        """Memory 子类如内部触发了 LLM 调用（如 SummaryMemory），返该批次产生的 usage.

        默认无内部 LLM 调用，返空 list. `Agent.respond` 在 `build_messages` 之后
        drain，把 usage append 到主 LLM 调用的 usage 之前一并交给 Discussion.
        """
        return []


class FullHistory(ConversationMemory):
    def build_messages(
        self, history: list[TranscriptEntry], owner: str,
    ) -> list[dict]:
        return _render(history, owner)


class WindowMemory(ConversationMemory):
    def __init__(self, max_recent: int) -> None:
        self.max_recent = max_recent

    def build_messages(
        self, history: list[TranscriptEntry], owner: str,
    ) -> list[dict]:
        pinned = [i for i, e in enumerate(history) if _is_pinned(e)]
        speech = [i for i, e in enumerate(history) if isinstance(e, SpeakerEntry)]
        kept_idx = sorted(set(pinned) | set(speech[-self.max_recent:]))
        return _render((history[i] for i in kept_idx), owner)


DEFAULT_SUMMARIZER_PROMPT = (
    "You compress multi-speaker discussions into structured notes. "
    "Preserve each speaker's stance, key claims, numbers, and points of disagreement. "
    "Write in the SAME language as the input conversation. "
    "Be compact; prefer brevity over completeness."
)

DEFAULT_SUMMARIZE_INSTRUCTION = (
    "Compress the conversation above into a structured summary: "
    "stance, claims, key numbers, disagreements. "
    "Stay in the conversation's language. Be compact. "
    "If <previous_summary> is present, merge the new turns into it and rewrite."
)


class SummaryMemory(ConversationMemory):
    def __init__(
        self,
        max_recent: int,
        *,
        client,
        summary_model: str,
        summary_max_tokens: int,
        summary_temperature: float,
        summarizer_prompt: str = DEFAULT_SUMMARIZER_PROMPT,
        summarize_instruction: str = DEFAULT_SUMMARIZE_INSTRUCTION,
    ) -> None:
        self.max_recent = max_recent
        self._client = client
        self._summary_model = summary_model
        self._summary_max_tokens = summary_max_tokens
        self._summary_temperature = summary_temperature
        self._summarizer_prompt = summarizer_prompt
        self._summarize_instruction = summarize_instruction
        self._summary_text: str = ""
        self._summarized_up_to: int = 0
        self._pending_usage: list[TokenUsage] = []

    def build_messages(
        self, history: list[TranscriptEntry], owner: str,
    ) -> list[dict]:
        speech = [i for i, e in enumerate(history) if isinstance(e, SpeakerEntry)]
        if len(speech) <= self.max_recent:
            return _render(history, owner)

        recent_cutoff = speech[-self.max_recent]
        stale_new = sum(
            1 for i in speech if self._summarized_up_to <= i < recent_cutoff
        )
        if stale_new >= self.max_recent:
            new_prefix = history[self._summarized_up_to:recent_cutoff]
            self._summary_text = self._run_summarizer(new_prefix)
            self._summarized_up_to = recent_cutoff

        result: list[TranscriptEntry] = []
        summary_shown = False
        for i, entry in enumerate(history):
            if i < self._summarized_up_to:
                if _is_pinned(entry):
                    result.append(entry)
                continue
            if self._summary_text and not summary_shown:
                result.append(SummaryEntry(content=self._summary_text))
                summary_shown = True
            result.append(entry)
        return _render(result, owner)

    def _run_summarizer(self, prefix_entries: list[TranscriptEntry]) -> str:
        messages = _render(prefix_entries, owner="_summarizer")
        if self._summary_text:
            messages.insert(
                0,
                {
                    "role": "user",
                    "content": (
                        f"<previous_summary>\n{self._summary_text}\n"
                        f"</previous_summary>"
                    ),
                },
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<instruction>\n{self._summarize_instruction}\n"
                    f"</instruction>"
                ),
            }
        )
        text, usage = self._client.chat(
            model=self._summary_model,
            system_prompt=self._summarizer_prompt,
            messages=messages,
            temperature=self._summary_temperature,
            max_tokens=self._summary_max_tokens,
            stream=False,
            tools=None,
        )
        self._pending_usage.append(usage)
        return text

    def drain_usage(self) -> list[TokenUsage]:
        out, self._pending_usage = self._pending_usage, []
        return out
