from __future__ import annotations

from typing import Iterable

PINNED_TYPES: frozenset[str] = frozenset({"topic", "turn", "artifact_event"})


def _render(entries: Iterable[dict], owner: str) -> list[dict]:
    messages: list[dict] = []
    for entry in entries:
        if entry.get("visible") is False:
            continue
        speaker = entry.get("speaker")
        if speaker is None:
            tag = entry.get("type", "topic")
            content = f"<{tag}>\n{entry['content']}\n</{tag}>"
            messages.append({"role": "user", "content": content})
        elif speaker == owner:
            messages.append({"role": "assistant", "content": entry["content"]})
        else:
            content = f'<message from="{speaker}">\n{entry["content"]}\n</message>'
            messages.append({"role": "user", "content": content})
    return messages


class ConversationMemory:
    def build_messages(self, history: list[dict], owner: str) -> list[dict]:
        raise NotImplementedError


class FullHistory(ConversationMemory):
    def build_messages(self, history: list[dict], owner: str) -> list[dict]:
        return _render(history, owner)


class WindowMemory(ConversationMemory):
    def __init__(self, max_recent: int) -> None:
        self.max_recent = max_recent

    def build_messages(self, history: list[dict], owner: str) -> list[dict]:
        pinned = [i for i, e in enumerate(history) if e.get("type") in PINNED_TYPES]
        speech = [i for i, e in enumerate(history) if "speaker" in e]
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

    def build_messages(self, history: list[dict], owner: str) -> list[dict]:
        speech = [i for i, e in enumerate(history) if "speaker" in e]
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

        result: list[dict] = []
        summary_shown = False
        for i, entry in enumerate(history):
            if i < self._summarized_up_to:
                if entry.get("type") in PINNED_TYPES:
                    result.append(entry)
                continue
            if self._summary_text and not summary_shown:
                result.append({"type": "summary", "content": self._summary_text})
                summary_shown = True
            result.append(entry)
        return _render(result, owner)

    def _run_summarizer(self, prefix_entries: list[dict]) -> str:
        messages = _render(prefix_entries, owner="_summarizer")
        if self._summary_text:
            messages.insert(
                0,
                {
                    "role": "user",
                    "content": f"<previous_summary>\n{self._summary_text}\n</previous_summary>",
                },
            )
        messages.append(
            {
                "role": "user",
                "content": f"<instruction>\n{self._summarize_instruction}\n</instruction>",
            }
        )
        return self._client.chat(
            model=self._summary_model,
            system_prompt=self._summarizer_prompt,
            messages=messages,
            temperature=self._summary_temperature,
            max_tokens=self._summary_max_tokens,
            stream=False,
            tools=None,
        )
