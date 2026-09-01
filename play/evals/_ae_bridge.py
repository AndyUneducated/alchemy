"""Bridge module: Directly import `agent_engine` typed view in the evals process (DECISIONS §13 / §16).

`play/evals` and `play/agent_engine` are sister packages of the same monorepo. Historically, evals was passed through subprocess
+ JSON envelope decoupled from agent_engine (DECISIONS §4 / §11); but transcript / scenario
Decoding views must be in-process (`Result.tool_calls() / .turns()` / `Scenario.expanded_turns()`
It is a pure function-level schema interpretation, each sample must be adjusted, and subprocessing ~1-2s cold start will make the evaluation time explode).

This bridge collects "sys.path injection + centralized import" into one place, and each metric/task module is re-exported from here.
That’s it, no more repeated `sys.path.insert(...)` + `try/finally` cleanup.

Additional re-export starting from §16: `TranscriptEntry` typed union (6 specific entry classes) + `TokenUsage`,
Let the evals consumer use isinstance dispatch to get the fields, no more `entry.get("...")` defense.

The pip install boundary is orthogonal to the import boundary (same idea as DECISIONS §14): requirements.txt of evals
There is no need to list agent_engine as a dependency (it is in the same source code tree); fresh checkout `pip install -r
The import path is automatically reachable after play/evals/requirements.txt`."""
from __future__ import annotations

import sys
from pathlib import Path

# play/evals/_ae_bridge.py → play/
_PLAY_DIR = Path(__file__).resolve().parent.parent
if str(_PLAY_DIR) not in sys.path:
    sys.path.insert(0, str(_PLAY_DIR))

from agent_engine import (  # noqa: E402
    ArtifactEventEntry,
    ExpandedTurn,
    Result,
    Scenario,
    SpeakerEntry,
    SummaryEntry,
    TokenUsage,
    ToolCall,
    ToolCallEntry,
    TopicEntry,
    TranscriptEntry,
    TurnEntry,
    TurnView,
)
from agent_engine.scenario import _resolve_who_names  # noqa: E402

__all__ = [
    "ArtifactEventEntry",
    "ExpandedTurn",
    "Result",
    "Scenario",
    "SpeakerEntry",
    "SummaryEntry",
    "TokenUsage",
    "ToolCall",
    "ToolCallEntry",
    "TopicEntry",
    "TranscriptEntry",
    "TurnEntry",
    "TurnView",
    "_resolve_who_names",
]
