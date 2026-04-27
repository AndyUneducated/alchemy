from __future__ import annotations

from .events import (
    AgentReply,
    ArtifactUpdate,
    RunFinished,
    StepStart,
    ToolCall,
)


class Callback:
    def on_step_start(self, event: StepStart) -> None: ...
    def on_agent_reply(self, event: AgentReply) -> None: ...
    def on_tool_call(self, event: ToolCall) -> None: ...
    def on_artifact_update(self, event: ArtifactUpdate) -> None: ...
    def on_run_finished(self, event: RunFinished) -> None: ...
