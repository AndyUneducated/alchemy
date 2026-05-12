"""Lightweight Ollama /api/chat wrapper using only stdlib."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from typing import Callable

from .config import MAX_TOKENS, OLLAMA_BASE_URL, TEMPERATURE
from .result import TokenUsage

MAX_TOOL_ROUNDS = 5


def _call(
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    stream: bool,
    tools: list[dict] | None = None,
) -> dict:
    """Single Ollama /api/chat round-trip; returns the final JSON frame."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if tools:
        body["tools"] = tools
        body["stream"] = False
        stream = False

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )

    chunks: list[str] = []
    last_data: dict = {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            data = json.loads(line)
            last_data = data
            token = data.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
                if stream:
                    sys.stdout.write(token)
                    sys.stdout.flush()
            if data.get("done"):
                break

    text = "".join(chunks)
    if stream:
        sys.stdout.write("\n")
    elif text:
        sys.stdout.write(text + "\n")
    sys.stdout.flush()

    last_data["_text"] = text
    return last_data


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
    """Send a chat request to Ollama and return `(text, TokenUsage)`.

    Ollama's `/api/chat` final frame carries `prompt_eval_count` /
    `eval_count` token counters. `TokenUsage` aggregates across tool-loop
    rounds. Ollama doesn't expose prompt cache; `cached_tokens=0`.
    """
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
    t0 = time.monotonic()
    in_tok = out_tok = 0

    for _ in range(MAX_TOOL_ROUNDS):
        data = _call(model, msgs, temperature=temperature,
                     max_tokens=max_tokens, stream=stream, tools=tools)
        in_tok += int(data.get("prompt_eval_count", 0) or 0)
        out_tok += int(data.get("eval_count", 0) or 0)

        tool_calls = data.get("message", {}).get("tool_calls")
        if not tool_calls or not tool_handler:
            return data["_text"], TokenUsage(
                model=model, caller=caller,
                input_tokens=in_tok, output_tokens=out_tok,
                cached_tokens=0,
                duration_ms=int((time.monotonic() - t0) * 1000),
                ts=time.time(),
            )

        msgs.append(data["message"])
        for tc in tool_calls:
            fn = tc["function"]
            result = tool_handler(fn["name"], fn.get("arguments", {}))
            msgs.append({"role": "tool", "content": result})

    return data["_text"], TokenUsage(
        model=model, caller=caller,
        input_tokens=in_tok, output_tokens=out_tok,
        cached_tokens=0,
        duration_ms=int((time.monotonic() - t0) * 1000),
        ts=time.time(),
    )
