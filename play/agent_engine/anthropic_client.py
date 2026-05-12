"""Anthropic SDK wrapper — drop-in replacement for ollama_client / openai_client."""

from __future__ import annotations

import sys
import time
from typing import Callable

import anthropic

from .config import ANTHROPIC_API_KEY, MAX_TOKENS, TEMPERATURE
from .result import TokenUsage

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MAX_TOOL_ROUNDS = 5


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


def _convert_tools(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool defs to Anthropic format."""
    out = []
    for t in openai_tools:
        fn = t["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        })
    return out


def _extract_usage(resp) -> tuple[int, int, int]:
    """Anthropic Message → (input_tokens, output_tokens, cached_tokens)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return (0, 0, 0)
    return (
        int(getattr(u, "input_tokens", 0) or 0),
        int(getattr(u, "output_tokens", 0) or 0),
        int(getattr(u, "cache_read_input_tokens", 0) or 0),
    )


def chat(
    model: str,
    *,
    system_prompt: str = "",
    messages: list[dict],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    stream: bool = True,
    tools: list[dict] | None = None,
    tool_handler: Callable[[str, dict], str] | None = None,
    caller: str = "",
) -> tuple[str, TokenUsage]:
    """Send a chat request to Claude and return `(text, TokenUsage)`.

    `TokenUsage` aggregates token counts across tool-loop rounds.
    """
    msgs = list(messages)
    anthropic_tools = _convert_tools(tools) if tools else None
    t0 = time.monotonic()
    in_tok = out_tok = cached_tok = 0

    for _ in range(MAX_TOOL_ROUNDS):
        filtered = _merge_consecutive(msgs)
        kwargs: dict = dict(
            model=model, max_tokens=max_tokens,
            temperature=temperature, system=system_prompt,
            messages=filtered,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if anthropic_tools or not stream:
            response = _client.messages.create(**kwargs)
            a, b, c = _extract_usage(response)
            in_tok += a
            out_tok += b
            cached_tok += c

            if response.stop_reason == "tool_use" and tool_handler:
                msgs.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = tool_handler(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                msgs.append({"role": "user", "content": tool_results})
                continue

            text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            return text, TokenUsage(
                model=model, caller=caller,
                input_tokens=in_tok, output_tokens=out_tok,
                cached_tokens=cached_tok,
                duration_ms=int((time.monotonic() - t0) * 1000),
                ts=time.time(),
            )

        # Streaming path (no tools)
        chunks: list[str] = []
        with _client.messages.stream(**kwargs) as stream_resp:
            for text in stream_resp.text_stream:
                if text:
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            final_msg = stream_resp.get_final_message()

        a, b, c = _extract_usage(final_msg)
        in_tok += a
        out_tok += b
        cached_tok += c

        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(chunks), TokenUsage(
            model=model, caller=caller,
            input_tokens=in_tok, output_tokens=out_tok,
            cached_tokens=cached_tok,
            duration_ms=int((time.monotonic() - t0) * 1000),
            ts=time.time(),
        )

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text, TokenUsage(
        model=model, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok,
        cached_tokens=cached_tok,
        duration_ms=int((time.monotonic() - t0) * 1000),
        ts=time.time(),
    )
