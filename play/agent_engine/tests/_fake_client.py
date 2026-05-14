"""LLM-less backend client stand-in for agent_engine integration tests.

Real backend clients (`anthropic_client / openai_client / gemini_client /
ollama_client`) all expose the same `chat(model, *, system_prompt, messages,
temperature, max_tokens, stream, tools, tool_handler, caller) -> (text, TokenUsage)`
shape. This module ships a scriptable drop-in for that signature so the
end-to-end Engine.invoke pipeline can be exercised without an LLM, without a
network, and without provider SDKs — letting tests catch breakages in
`engine.py / discussion.py / agent.py / memory.py / artifact.py / tracer.py /
scenario.py` wiring.

Wiring: tests `monkeypatch.setattr(agent_engine.agent._client, "chat",
fake.chat)`. `scenario._backend_client` and `agent._client` reference the same
module object, so a single patch covers both Agent.respond and
SummaryMemory's internal summarizer call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_engine.result import TokenUsage


@dataclass
class Script:
    """One scripted Agent.respond reply.

    `tools` is invoked **before** returning the reply text — this matches the
    real chat() loop where tool_calls are resolved within the same logical
    `respond` and emit ArtifactEventEntry / ToolCallEntry into the discussion's
    drain queues before the SpeakerEntry is appended.
    """
    text: str = "ok"
    tools: list[dict[str, Any]] = field(default_factory=list)  # [{name, args}]
    input_tokens: int = 1
    output_tokens: int = 2
    cached_tokens: int = 0


class FakeBackendClient:
    """Per-caller scripted client. Falls back to a noop reply when unscripted."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Script]] = {}
        self.calls: list[dict[str, Any]] = []

    def script(self, caller: str, *scripts: Script) -> None:
        self._queues.setdefault(caller, []).extend(scripts)

    def chat(
        self,
        model: str,
        *,
        system_prompt: str = "",
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 0,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_handler: Callable[[str, dict], str] | None = None,
        caller: str = "",
    ) -> tuple[str, TokenUsage]:
        self.calls.append({
            "caller": caller,
            "model": model,
            "system_prompt": system_prompt,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "tools": tools,
        })
        queue = self._queues.get(caller) or []
        script = queue.pop(0) if queue else Script()
        if script.tools and tool_handler is not None:
            for call in script.tools:
                tool_handler(call["name"], dict(call.get("args") or {}))
        usage = TokenUsage(
            model=model,
            caller=caller,
            input_tokens=script.input_tokens,
            output_tokens=script.output_tokens,
            cached_tokens=script.cached_tokens,
        )
        return script.text, usage
