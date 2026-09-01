"""Ollama adapter: stdlib only / /api/generate direct prompt (no chat template).

Why /api/generate is not /api/chat (the reverse of [`play/agent_engine/ollama_client.py`](play/agent_engine/ollama_client.py)):
  The lm-eval philosophy requires that task fully own the prompt literal string. /api/chat will be packaged according to the model chat template
  user/assistant role + system prompt, destroy the prompt literal to reproduce; /api/generate directly dial raw prompt.

Only implement generate_until (the only request type used in phase 3); loglikelihood adopts ABC default
Throw NotImplementedError——phase 9 calibration and restart (then ollama will use /api/embeddings or
HF transformers tokenizer (direct calculation).

Why not reuse play/agent_engine/ollama_client.py:
  play/ sub-projects do not import each other (grep verifies zero-crossing), keeping evals self-consistent. Here stdlib /api/generate
  The package itself is extremely thin (< 60 lines) and the cost of repeated implementation is much lower than coupling across subprojects."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import ClassVar

from ..api import Request, Response, Usage
from .base import LM


class OllamaLM(LM):
    """`OllamaLM(model="qwen3.6:27b")` → Use local ollama HTTP, the name is `ollama:<model>`.

    `base_url` priority: construction parameters > env `EVALS_OLLAMA_BASE_URL` > default `localhost:11434`.
    `temperature=0.0` + `seed=0` makes the test more certain by default (ollama options.seed transparently transmitted).

    `think=False` defaults to qwen3.x and other reasoning models to use chat mode (thinking trace is not output);
    Otherwise, the response field is empty and the content is all inserted into the thinking field, and generate_until will get empty text. By
    env `EVALS_OLLAMA_THINK=true/1` Explicit opt-in. Ollama < 0.4 Ignore this field, for qwen2.5
    There is no harm in waiting for non-reasoning models (the key is discarded if it is not recognized)."""

    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        request_timeout: float = 120.0,
        think: bool | None = None,
    ) -> None:
        self.model = model
        env_url = os.environ.get("EVALS_OLLAMA_BASE_URL")
        self.base_url = (base_url or env_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.request_timeout = request_timeout
        if think is None:
            think = os.environ.get("EVALS_OLLAMA_THINK", "").lower() in {"1", "true", "yes"}
        self.think = think
        self.name = f"ollama:{model}"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        """Call /api/generate serially; phase 1+ concurrency optimization is done at the runner layer (uniformly applied to all LMs).

        Fill in Response.usage / latency_ms starting from phase 6:
          - `prompt_eval_count` → Usage.tokens_in (missing field → None)
          - `eval_count` → Usage.tokens_out (missing field → None)
          - `total_duration` (ns) → latency_ms (ns / 1e6); end-to-end time reported by ollama,
            More accurate than perf_counter (excluding Python call stack/urllib socket queuing).
        Older versions of ollama services may not return these fields - getattr style .get(...) returns None,
        Naturally compatible with the efficiency_aggregated "non-None collection" protocol."""
        out: list[Response] = []
        for req in requests:
            options: dict = {
                "temperature": self.temperature,
                "num_predict": req.max_tokens,
            }
            if self.seed is not None:
                options["seed"] = self.seed
            if req.until:
                options["stop"] = list(req.until)

            body = {
                "model": self.model,
                "prompt": req.prompt,
                "stream": False,
                "think": self.think,
                "options": options,
            }
            payload = json.dumps(body).encode("utf-8")
            http_req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=self.request_timeout) as resp:
                data = json.loads(resp.read())
            text = data.get("response", "") or ""
            tokens_in = data.get("prompt_eval_count")
            tokens_out = data.get("eval_count")
            usage: Usage | None = None
            if tokens_in is not None or tokens_out is not None:
                usage = Usage(tokens_in=tokens_in, tokens_out=tokens_out)
            total_duration_ns = data.get("total_duration")
            latency_ms: float | None = None
            if total_duration_ns is not None:
                latency_ms = float(total_duration_ns) / 1_000_000.0
            out.append(
                Response(
                    doc_id=req.doc_id,
                    text=text,
                    latency_ms=latency_ms,
                    usage=usage,
                )
            )
        return out
