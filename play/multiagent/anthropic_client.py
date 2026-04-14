"""Anthropic SDK wrapper — drop-in replacement for ollama_client / openai_client."""

import sys

import anthropic

from config import ANTHROPIC_API_KEY

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def chat(model: str, messages: list[dict], *, temperature: float = 0.7,
         max_tokens: int = 512, stream: bool = True) -> str:
    """Send a chat request to Claude and return the assistant reply.

    When *stream* is True the tokens are printed to stdout as they arrive.
    """
    system_prompt = ""
    filtered: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered.append({"role": m["role"], "content": m["content"]})

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
