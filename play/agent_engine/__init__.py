"""``play/agent_engine``: a step-driven multi-agent discussion runtime.

Public API (LangChain Runnable–style; plan §5.1):

    from agent_engine import Engine, Scenario, Result, Callback
    from agent_engine.events import (
        Event, StepStart, AgentReply, ToolCall, ArtifactUpdate, RunFinished,
    )

    scenario = Scenario.from_yaml("scenarios/qa_discuss.md")
    engine = Engine(scenario)
    result = engine.invoke(
        initial_artifact={"Requirements": "..."},
        transcript_path="/tmp/transcript.json",
    )

CLI entry point: ``python -m agent_engine <scenario.md> [--save-artifact PATH] ...``
"""

from .callbacks import Callback
from .engine import Engine
from .result import Result
from .scenario import Scenario

__all__ = [
    "Callback",
    "Engine",
    "Result",
    "Scenario",
]
