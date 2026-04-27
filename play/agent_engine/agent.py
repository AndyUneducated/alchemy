from __future__ import annotations

from typing import Callable

from .config import BACKEND, DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE
from .memory import ConversationMemory, FullHistory

if BACKEND == "anthropic":
    from . import anthropic_client as _client
elif BACKEND == "openai":
    from . import openai_client as _client
elif BACKEND == "gemini":
    from . import gemini_client as _client
else:
    from . import ollama_client as _client


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        tools: list[dict] | None = None,
        tool_handler: Callable[[str, dict], str] | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools
        self.tool_handler = tool_handler
        self.memory = memory or FullHistory()

    def respond(
        self,
        history: list[dict],
        *,
        instruction: str | None = None,
        stream: bool = True,
        artifact_view: str | None = None,
    ) -> str:
        messages = self.memory.build_messages(history, self.name)
        if artifact_view is not None:
            messages.append({"role": "user", "content": f"<artifact>\n{artifact_view}\n</artifact>"})
        if instruction:
            messages.append({"role": "user", "content": f"<instruction>\n{instruction}\n</instruction>"})
        return _client.chat(
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            tools=self.tools,
            tool_handler=self.tool_handler,
        )
