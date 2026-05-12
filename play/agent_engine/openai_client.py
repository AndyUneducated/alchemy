"""OpenAI SDK wrapper — drop-in replacement for ollama_client."""

from __future__ import annotations

import json
import sys
import time
from typing import Callable

from openai import OpenAI

from .config import MAX_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from .result import TokenUsage

_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

MAX_TOOL_ROUNDS = 5


def _extract_usage(resp) -> tuple[int, int, int]:
    """OpenAI ChatCompletion / chunk → (input_tokens, output_tokens, cached_tokens)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return (0, 0, 0)
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return (
        int(getattr(u, "prompt_tokens", 0) or 0),
        int(getattr(u, "completion_tokens", 0) or 0),
        int(cached),
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
    """Send a chat request via any OpenAI-compatible endpoint.

    Returns `(reply_text, TokenUsage)`. `TokenUsage` aggregates token counts
    across all tool-loop rounds (one chat() call = one logical agent turn,
    even if the model used multiple LLM round-trips for tool calls).
    """
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
    t0 = time.monotonic()
    in_tok = out_tok = cached_tok = 0

    for _ in range(MAX_TOOL_ROUNDS):
        kwargs: dict = dict(
            model=model, messages=msgs,
            temperature=temperature, max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["stream"] = False
        else:
            kwargs["stream"] = stream
            if stream:
                kwargs["stream_options"] = {"include_usage": True}

        response = _client.chat.completions.create(**kwargs)

        if kwargs.get("stream"):
            chunks: list[str] = []
            last_chunk = None
            for chunk in response:
                last_chunk = chunk
                if chunk.choices:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        chunks.append(token)
                        sys.stdout.write(token)
                        sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            if last_chunk is not None:
                a, b, c = _extract_usage(last_chunk)
                in_tok += a
                out_tok += b
                cached_tok += c
            return "".join(chunks), TokenUsage(
                model=model, caller=caller,
                input_tokens=in_tok, output_tokens=out_tok,
                cached_tokens=cached_tok,
                duration_ms=int((time.monotonic() - t0) * 1000),
                ts=time.time(),
            )

        a, b, c = _extract_usage(response)
        in_tok += a
        out_tok += b
        cached_tok += c

        msg = response.choices[0].message
        if msg.tool_calls and tool_handler:
            msgs.append(msg.model_dump())
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = tool_handler(tc.function.name, args)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        text = msg.content or ""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return text, TokenUsage(
            model=model, caller=caller,
            input_tokens=in_tok, output_tokens=out_tok,
            cached_tokens=cached_tok,
            duration_ms=int((time.monotonic() - t0) * 1000),
            ts=time.time(),
        )

    text = msg.content or ""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text, TokenUsage(
        model=model, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok,
        cached_tokens=cached_tok,
        duration_ms=int((time.monotonic() - t0) * 1000),
        ts=time.time(),
    )
