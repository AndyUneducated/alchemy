"""OpenAI SDK wrapper — drop-in replacement for ollama_client."""

import sys

from openai import OpenAI

from config import MAX_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE

_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True) -> str:
    """Send a chat request via any OpenAI-compatible endpoint.

    When *stream* is True the tokens are printed to stdout as they arrive.
    """
    full = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    response = _client.chat.completions.create(
        model=model,
        messages=full,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if not stream:
        text = response.choices[0].message.content or ""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return text

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
