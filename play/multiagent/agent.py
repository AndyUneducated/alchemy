"""Agent: a persona that can respond to a shared conversation."""

from __future__ import annotations

from config import BACKEND, DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE

if BACKEND == "anthropic":
    import anthropic_client as _client
elif BACKEND == "openai":
    import openai_client as _client
elif BACKEND == "gemini":
    import gemini_client as _client
else:
    import ollama_client as _client


class Agent:
    """A discussion participant with a fixed persona."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def respond(self, history: list[dict], *, stream: bool = True) -> str:
        """Generate a reply given the shared conversation history.

        Builds a per-agent view: own past replies become ``assistant``,
        other speakers and system injections become ``user``.
        """
        messages: list[dict] = []
        for entry in history:
            speaker = entry.get("speaker")
            if speaker is None:
                tag = entry.get("type", "topic")
                content = f"<{tag}>\n{entry['content']}\n</{tag}>"
                messages.append({"role": "user", "content": content})
            elif speaker == self.name:
                messages.append({"role": "assistant", "content": entry["content"]})
            else:
                content = f'<message from="{speaker}">\n{entry["content"]}\n</message>'
                messages.append({"role": "user", "content": content})
        return _client.chat(
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
        )
