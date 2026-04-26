"""Streaming event types for ``Engine.stream()`` / ``Engine.astream()``.

Today only ``RunFinished`` is emitted (by ``Engine.invoke`` after a sync run)
so callbacks have a working hook point. The other subclasses are signature
placeholders — adding them to ``Discussion`` internals (per-step, per-tool,
per-artifact-update emission) is future work that will not break existing
callback consumers (each ``on_xxx`` defaults to no-op).

Field shape stays loose (dataclass-style positional args via ``__init__``):
``ts`` is ISO 8601 UTC (plan §5.4 / §9.1 — W3C-ready, no trace_id today).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> str:
    """Return a millisecond-precision ISO 8601 UTC timestamp."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


@dataclass
class Event:
    """Base streaming event. Subclasses add fields; ``ts`` defaults to now."""

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
