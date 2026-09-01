"""Family 5 sequel: agent_engine require_tool compliance metric — nudge_fire_rate.

The design is the same as [trajectory.py](trajectory.py) (semi-universal, tied to agent_engine envelope schema,
Pure function + closure factory). Provide "model for SFT baseline / Phase 5 retest in require_tool step
First time compliance rate on the "signal" - the lower the better (less trigger nudge = model gets it right in one go).

Core definition:
  - **require_tool turn** = `require_tool: <tool>` field is declared in scenario.steps
    an expanded (agent, step) tuple
  - **nudge fire** = The first attempt of this turn has no output `(caller=agent, tool=required_tool)`
    artifact event, the engine prints `🔁` and initiates a retry
  - **nudge_fire_rate** = nudge_fire number / require_tool turn total number ∈ [0,1] ↓

Data contract (doc.metadata standard key, injected by NudgeFireRate task):
  - `trajectory` dict envelope form `{transcript, artifact,
                                                     warnings, success, usage}`(agent_engine
                                                     §16 typed entry / TokenUsage specification)
  - `expected_require_tool_turns` list[dict] `[{turn_idx, agent, step_id, tool}, ...]`
                                                     By process_docs/load_prediction from
                                                     scenario YAML automatically derived
  - `scenario_path` str is only used to get the id when reporting by_scenario breakdown

Phase 1 failure mode taxonomy (introduced in 1.C; Category 3):
  - `missed` The first attempt of this caller did not call any tools at all
  - In the first attempt of `wrong_tool`, the caller called another tool (non-required_tool)
  - `wrong_args` (**deferred to Phase 5**) adjusted the correct tool but was rejected by the schema - artifact
                  The handler currently does not send events in the error path, so it cannot be judged based on transcript alone; it is required
                  Agent_engine then adds `{ok: false}` event to the dispatch error path and then enables it.
                  Under the current implementation, this bucket is always 0, and the document is marked honestly."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .._ae_bridge import (
    ArtifactEventEntry,
    Result,
    Scenario,
    ToolCallEntry,
    TranscriptEntry,
)
from ..api import Doc, Response


def derive_expected_turns(scenario_path: str | Path) -> list[dict[str, Any]]:
    """Parse scenario → `[{turn_idx, agent, step_id, tool}, ...]` (only turn of require_tool).

    DECISIONS §13: Internal delegate `agent_engine.Scenario.expanded_turns()`; turn_idx 1-based,
    Aligned with the `turn N of total` marker written by `discussion.run`."""
    expanded = Scenario.from_yaml(str(scenario_path)).expanded_turns()
    return [
        {
            "turn_idx": e.turn_idx,
            "agent": e.agent,
            "step_id": e.step_id,
            "tool": str(e.require_tool),
        }
        for e in expanded
        if e.require_tool
    ]


# ---------- failure mode classification (attempt → mode) ---------------------------

def classify_failure_mode(
    first_attempt_events: list[TranscriptEntry],
    agent: str,
    required_tool: str,
) -> str:
    """Failure classification when First attempt does not meet require_tool (Phase 1: missed / wrong_tool).

    The `wrong_args` bucket (tool required but schema rejected) is currently undetectable - artifact dispatch in error
    The path does not send events and cannot be seen purely by transcript; deferred to Phase 5 (agent_engine is in
    The error path is enabled after adding `{ok: false}` event).

    `required_tool` is not currently used - the argument is reserved to distinguish between "required_tool" and "wrong_args" when the wrong_args bucket is enabled
    args does not match "vs" wrong_tool subclasses."""
    _ = required_tool  # reserved for wrong_args extension
    called_any_tool = any(
        isinstance(e, (ArtifactEventEntry, ToolCallEntry)) and e.caller == agent
        for e in first_attempt_events
    )
    return "wrong_tool" if called_any_tool else "missed"


# ---------- Main entrance: compute_nudge_fire_rate ----------------------------

# Phase 1 supported failure modes (taxonomy starting category 2 + 1 deferred placeholder = 3 total buckets;
# See the wrong_args footnote in the module documentation).
FAILURE_MODES: tuple[str, ...] = ("missed", "wrong_tool", "wrong_args")


def compute_nudge_fire_rate(
    envelope: dict,
    expected_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """envelope dict + expectation table → metric dictionary.

    envelope takes `Result.from_dict` typed deserialization (§16); `turns()` / `attempts()` all
    typed dispatch.

    Return structure (doc level; multi-doc aggregation is done in task.aggregation):
        {
            "nudge_fire_rate": float | None, # fires / total (total=0 → None)
            "nudge_fire_count": int,
            "require_tool_total": int,
            "by_tool": {tool: {"fired": int, "total": int}, ...},
            "by_failure_mode": {mode: int, ...}, # Accumulate the number of occurrences of each mode
            "per_turn": [
                {turn_idx, agent, step_id, tool, fired, mode | None, n_attempts}, ...
            ],
        }

    Boundary:
      - expected_turns is empty (such as brainstorm/debate/roundtable) → rate=None, total=0,
        All other buckets are empty - it means "this doc does not participate in the nudge measurement" (it is naturally ignored by total weighting during aggregation).
      - Number of segments < expected_turn.turn_idx (subprocess crashes/scenario truncation)→
        The turn is marked as fired=True (mode='missed', n_attempts=0) and is included in the denominator - a conservative failure."""
    result = Result.from_dict(envelope)
    turns = result.turns()
    per_turn: list[dict[str, Any]] = []
    by_tool: dict[str, dict[str, int]] = {}
    mode_counter: Counter[str] = Counter()
    fire_count = 0

    for exp in expected_turns:
        idx = int(exp["turn_idx"]) - 1
        agent = str(exp["agent"])
        tool = str(exp["tool"])
        step_id = exp.get("step_id")

        bucket = by_tool.setdefault(tool, {"fired": 0, "total": 0})
        bucket["total"] += 1

        if idx >= len(turns):
            # The turn did not run (the subprocess crashed midway / the scenario was shorter than expected), so it is considered missed.
            fire_count += 1
            bucket["fired"] += 1
            mode_counter["missed"] += 1
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": True, "mode": "missed", "n_attempts": 0,
            })
            continue

        attempts = turns[idx].attempts(agent)
        if not attempts:
            # The agent did not speak at all in this segment - considered missed.
            fire_count += 1
            bucket["fired"] += 1
            mode_counter["missed"] += 1
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": True, "mode": "missed", "n_attempts": 0,
            })
            continue

        first_satisfied = any(
            isinstance(e, (ArtifactEventEntry, ToolCallEntry))
            and e.caller == agent and e.tool == tool
            for e in attempts[0]
        )
        if first_satisfied:
            per_turn.append({
                "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
                "tool": tool, "fired": False, "mode": None, "n_attempts": len(attempts),
            })
            continue

        # nudge triggered
        fire_count += 1
        bucket["fired"] += 1
        mode = classify_failure_mode(attempts[0], agent, tool)
        mode_counter[mode] += 1
        per_turn.append({
            "turn_idx": exp["turn_idx"], "agent": agent, "step_id": step_id,
            "tool": tool, "fired": True, "mode": mode, "n_attempts": len(attempts),
        })

    total = len(expected_turns)
    rate: float | None = (fire_count / total) if total > 0 else None

    # Explicitly list all three buckets of FAILURE_MODES so that the 0 count is also visible - there is no shortage of columns in the downstream breakdown table
    by_failure_mode = {m: int(mode_counter.get(m, 0)) for m in FAILURE_MODES}

    return {
        "nudge_fire_rate": rate,
        "nudge_fire_count": int(fire_count),
        "require_tool_total": int(total),
        "by_tool": by_tool,
        "by_failure_mode": by_failure_mode,
        "per_turn": per_turn,
    }


# ---------- closure factories (same shape as trajectory.py protocol) ---------------

def nudge_fire_rate_metric() -> Callable[[Doc, Response], float | None]:
    """Factory: (Doc, Response) → nudge_fire_rate for this doc.

    Depends on doc.metadata['trajectory'] (envelope dict) + doc.metadata['expected_require_tool_turns']
    Have been injected by process_docs / load_prediction. Returns None ("not measured") if the field is missing."""

    def _score(doc: Doc, _response: Response) -> float | None:
        envelope = doc.metadata.get("trajectory", {}) or {}
        expected = doc.metadata.get("expected_require_tool_turns") or []
        if not envelope or "transcript" not in envelope:
            return None
        result = compute_nudge_fire_rate(envelope, expected)
        return result["nudge_fire_rate"]

    return _score
