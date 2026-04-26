"""Anthropic SDK wrapper — drop-in replacement for ollama_client / openai_client."""

from __future__ import annotations

import json
import sys
from typing import Callable

import anthropic

from .config import ANTHROPIC_API_KEY, MAX_TOKENS, TEMPERATURE

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


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True,
         tools: list[dict] | None = None,
         tool_handler: Callable[[str, dict], str] | None = None) -> str:
    """Send a chat request to Claude and return the assistant reply.

    When *tools* are provided the function enters a non-streaming tool loop.
    """
    msgs = list(messages)
    anthropic_tools = _convert_tools(tools) if tools else None

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

            if response.stop_reason == "tool_use" and tool_handler:
                # Build assistant message with all content blocks
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
            return text

        # Streaming path (no tools)
        chunks: list[str] = []
        with _client.messages.stream(**kwargs) as stream_resp:
            for text in stream_resp.text_stream:
                if text:
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(chunks)

    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text
