"""agent_traj run-path live e2e：agent_engine subprocess + ollama 双 gate.

目标：在双 gate 都满足时把"evals → subprocess → agent_engine → JSON envelope → trajectory
metric"端到端跑一遍 brainstorm.md（最小 scenario，无 artifact，3 agents × 2 steps）.

为什么用 brainstorm 而不是 panel：
  - panel 有 4 名 member + moderator + 11 个 step + tool 链——单跑 ~分钟级
  - brainstorm 仅 2 step，~10-30s（qwen2.5:32b 在 M-series Mac 上）；CI 友好
  - phase 5 在线路径主要是 envelope contract + run_fn 通了；行为细节由 score 矩阵覆盖

被 gate 跳过时（无 ollama / 模型未拉）conftest 会清晰提示用户怎么 ollama pull.
"""

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
    """e2e：跑 brainstorm.md → trajectory 注入 metadata + 5 个 metric 都能算出.

    成功条件:
      - subprocess 正常退出（不抛 RuntimeError）
      - doc.metadata['trajectory'] 7 个 key 全 pin 上
      - 至少 1 个 speaker 进了 transcript（agent_engine 真有 LLM 回应）
      - 5 个 metric 都能 evaluate（不抛 / 数值在 [0,1]）
    """
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

    # 至少 1 名 speaker 应该真出过声（brainstorm 没有 require_tool，模型怠工 0
    # 几率极低；若真发生说明 agent_engine 整段链路有问题，比 metric 数值更关键）
    speakers = {e["speaker"] for e in traj["transcript"] if "speaker" in e}
    assert len(speakers) >= 1, f"transcript 没有 speaker 条目: {traj['transcript']}"

    # 5 个 metric 都能算（不锁数值，仅锁不抛 + 落 [0,1]）
    from evals.api import Response
    sr = task.process_results(enriched, Response(doc_id=doc.id))
    for k in ("task_success", "tool_call_set_f1", "argument_correctness",
              "trajectory_match", "trajectory_coverage"):
        assert k in sr.metrics, f"metric {k!r} missing"
        assert 0.0 <= sr.metrics[k] <= 1.0, f"{k}={sr.metrics[k]} out of [0,1]"
