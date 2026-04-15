"""Lightweight Ollama /api/chat wrapper using only stdlib."""

import json
import sys
import urllib.request

from config import BASE_URL, MAX_TOKENS, TEMPERATURE


def chat(model: str, *, system_prompt: str = "", messages: list[dict],
         temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
         stream: bool = True) -> str:
    """Send a chat request to Ollama and return the assistant reply.

    When *stream* is True the tokens are printed to stdout as they arrive.
    """
    full = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages
    url = f"{BASE_URL}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": full,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    chunks: list[str] = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
                if stream:
                    sys.stdout.write(token)
                    sys.stdout.flush()
            if data.get("done"):
                break

    full = "".join(chunks)
    if stream:
        sys.stdout.write("\n")
    else:
        sys.stdout.write(full + "\n")
    sys.stdout.flush()

    return full
