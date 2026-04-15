"""Google Gemini SDK wrapper — drop-in replacement for ollama_client / openai_client."""

import sys

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MAX_TOKENS, TEMPERATURE

_client = genai.Client(api_key=GEMINI_API_KEY)


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
    """Send a chat request to Gemini and return the assistant reply.

    When *stream* is True the tokens are printed to stdout as they arrive.
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

    if not stream:
        response = _client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return text

    chunks: list[str] = []
    for chunk in _client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        token = chunk.text or ""
        if token:
            chunks.append(token)
            sys.stdout.write(token)
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks)
