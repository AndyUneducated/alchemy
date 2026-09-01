"""agent_traj run-path live e2e: agent_engine subprocess + ollama double gate.

Goal: When both gates are satisfied, put "evals → subprocess → agent_engine → JSON envelope → trajectory
metric" run brainstorm.md end-to-end (minimum scenario, no artifacts, 3 agents × 2 steps).

Why use brainstorm instead of panel:
  - The panel has 4 members + moderator + 11 step + tool chains - single operation ~ minute level
  - brainstorm only 2 steps, ~10-30s (qwen3.6:27b on M-series Mac); CI friendly
  - The online path of phase 5 is mainly envelope contract + run_fn passed; the behavior details are covered by the score matrix

When skipped by gate (no ollama/model not pulled) conftest will clearly prompt the user how to pull ollama."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evals.api import Doc
from evals.models.agent_engine_run import make_run_fn
from evals.tasks.agent_traj import AgentTraj

from .conftest import agent_engine_required, ollama_required


@ollama_required
@agent_engine_required
def test_run_brainstorm_e2e_pins_trajectory():
    """e2e: Run brainstorm.md → trajectory and inject metadata + 5 metrics to calculate it.

    Success conditions:
      - The subprocess exits normally (no RuntimeError is thrown)
      - doc.metadata['trajectory'] 7 keys all pinned
      - At least 1 speaker entered the transcript (agent_engine really responded with LLM)
      - All 5 metrics can be evaluated (no throwing / value in [0,1])"""
    run_fn = make_run_fn()
    task = AgentTraj(run_fn=run_fn)

    doc = Doc(
        id="brainstorm_live",
        input="团建活动方案讨论",
        target=None,
        metadata={
            "scenario_path": "agent_engine/scenarios/brainstorm.md",
            "success_predicate": "speakers_covered",
            "expected_speakers": ["前端", "后端", "PM"],
            "gold_tool_seq": [],
            "gold_tool_calls": [],
            "coverage_kind": "speakers",
        },
    )

    [enriched] = task.process_docs([doc])

    traj = enriched.metadata.get("trajectory")
    assert traj is not None, "process_docs 未注入 trajectory"
    for key in ("transcript", "artifact", "warnings", "success",
                "tool_calls", "tool_seq", "decision"):
        assert key in traj, f"trajectory missing {key!r}"

    # At least 1 speaker should have actually made a sound (brainstorm does not require_tool, model slowdown 0
    # The probability is extremely low; if it does happen, it means there is a problem with the entire link of agent_engine, which is more critical than the metric value)
    speakers = {e["speaker"] for e in traj["transcript"] if "speaker" in e}
    assert len(speakers) >= 1, f"transcript 没有 speaker 条目: {traj['transcript']}"

    # All 5 metrics can be calculated (no locking of values, only locking without throwing + falling [0,1])
    from evals.api import Response
    sr = task.process_results(enriched, Response(doc_id=doc.id))
    for k in ("task_success", "tool_call_set_f1", "argument_correctness",
              "trajectory_match", "trajectory_coverage"):
        assert k in sr.metrics, f"metric {k!r} missing"
        assert 0.0 <= sr.metrics[k] <= 1.0, f"{k}={sr.metrics[k]} out of [0,1]"
