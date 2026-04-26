"""``Callback``: hook surface for ``Engine.invoke`` / future stream().

Today only ``on_run_finished`` is fired by ``Engine.invoke`` (sync path).
Subclasses can override any subset of ``on_xxx`` — defaults are no-ops so
adding a new event type later (e.g. ``on_step_start``) is non-breaking.

Naming follows LangChain's ``BaseCallbackHandler`` (``on_<event>`` methods)
but we pass the structured ``Event`` dataclass (events.py) instead of LC's
loose ``**kwargs`` — easier to type-hint & evolve.
"""

from __future__ import annotations

from .events import (
    AgentReply,
    ArtifactUpdate,
    RunFinished,
    StepStart,
    ToolCall,
)


class Callback:
    """Subclass and override the events you care about; defaults pass."""

    def on_step_start(self, event: StepStart) -> None: ...
    def on_agent_reply(self, event: AgentReply) -> None: ...
    def on_tool_call(self, event: ToolCall) -> None: ...
    def on_artifact_update(self, event: ArtifactUpdate) -> None: ...
    def on_run_finished(self, event: RunFinished) -> None: ...
