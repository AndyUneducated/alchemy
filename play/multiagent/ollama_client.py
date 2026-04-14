"""Lightweight Ollama /api/chat wrapper using only stdlib."""

import json
import sys
import urllib.request

from config import BASE_URL


def chat(model: str, messages: list[dict], *, temperature: float = 0.7,
         max_tokens: int = 512, stream: bool = True) -> str:
    """Send a chat request to Ollama and return the assistant reply.

    When *stream* is True the tokens are printed to stdout as they arrive.
    """
    url = f"{BASE_URL}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
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

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(chunks)
