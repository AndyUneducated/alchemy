"""Google Gemini SDK wrapper — drop-in replacement for ollama_client / openai_client."""

from __future__ import annotations

import json
import sys
from typing import Callable

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MAX_TOKENS, TEMPERATURE

_client = genai.Client(api_key=GEMINI_API_KEY)

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


def _convert_tools(openai_tools: list[dict]) -> list[types.Tool]:
    """Convert OpenAI-format tool defs to Gemini FunctionDeclarations."""
    decls = []
    for t in openai_tools:
        fn = t["function"]
        params = fn.get("parameters", {})
        decls.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=params,
        ))
    return [types.Tool(function_declarations=decls)]


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True,
         tools: list[dict] | None = None,
         tool_handler: Callable[[str, dict], str] | None = None) -> str:
    """Send a chat request to Gemini and return the assistant reply.

    When *tools* are provided the function enters a non-streaming tool loop.
    """
    merged = _merge_consecutive(messages)
    contents: list[types.Content] = []
    for m in merged:
        role = "model" if m["role"] == "assistant" else m["role"]
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt or None,
    )
    if tools:
        config.tools = _convert_tools(tools)

    for _ in range(MAX_TOOL_ROUNDS):
        if tools or not stream:
            response = _client.models.generate_content(
                model=model, contents=contents, config=config,
            )

            fc_parts = [
                p for p in (response.candidates[0].content.parts or [])
                if p.function_call
            ]
            if fc_parts and tool_handler:
                contents.append(response.candidates[0].content)
                fn_responses = []
                for p in fc_parts:
                    fc = p.function_call
                    result_str = tool_handler(fc.name, dict(fc.args))
                    fn_responses.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response=json.loads(result_str),
                        ),
                    ))
                contents.append(types.Content(role="user", parts=fn_responses))
                continue

            text = response.text or ""
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            return text

        # Streaming path (no tools)
        chunks: list[str] = []
        for chunk in _client.models.generate_content_stream(
            model=model, contents=contents, config=config,
        ):
            token = chunk.text or ""
            if token:
                chunks.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(chunks)

    text = response.text or ""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return text
