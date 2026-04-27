from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


@dataclass
class Event:
    ts: str = field(default_factory=_utc_now)


@dataclass
class StepStart(Event):
    step_id: str | None = None
    agent: str = ""


@dataclass
class AgentReply(Event):
    agent: str = ""
    content: str = ""


@dataclass
class ToolCall(Event):
    name: str = ""
    args: dict = field(default_factory=dict)
    result: str | None = None
    duration_ms: int = 0


@dataclass
class ArtifactUpdate(Event):
    section: str = ""
    mode: str = ""


@dataclass
class RunFinished(Event):
    success: bool = True
