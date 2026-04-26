"""OpenAI SDK wrapper — drop-in replacement for ollama_client."""

from __future__ import annotations

import json
import sys
from typing import Callable

from openai import OpenAI

from config import MAX_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE

_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

MAX_TOOL_ROUNDS = 5


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True,
         tools: list[dict] | None = None,
         tool_handler: Callable[[str, dict], str] | None = None) -> str:
    """Send a chat request via any OpenAI-compatible endpoint.

    When *tools* are provided the function enters a non-streaming tool loop.
    """
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)

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

        response = _client.chat.completions.create(**kwargs)

        if kwargs.get("stream"):
            chunks: list[str] = []
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    chunks.append(token)
                    sys.stdout.write(token)
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(chunks)

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
        return text

    text = msg.content or ""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text
