"""Anthropic SDK wrapper — drop-in replacement for ollama_client / openai_client."""

import sys

import anthropic

from config import ANTHROPIC_API_KEY, MAX_TOKENS, TEMPERATURE

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _merge_consecutive(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role for API compatibility."""
    if not messages:
        return messages
    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n\n" + msg["content"]}
        else:
            merged.append(msg)
    return merged


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True) -> str:
    """Send a chat request to Claude and return the assistant reply.

    When *stream* is True the tokens are printed to stdout as they arrive.
    """
    filtered = _merge_consecutive(messages)

    if not stream:
        response = _client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=filtered,
        )
        text = response.content[0].text
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return text

    chunks: list[str] = []
    with _client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=filtered,
    ) as stream_resp:
        for text in stream_resp.text_stream:
            if text:
                chunks.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks)
