"""agent_traj × 4 stub predictions × 3 docs = 12 sample 矩阵：核心教学叙事的数值守门员.

跑 evaluate_score 端到端：load gold.jsonl + pred.jsonl → metrics → aggregated
（不引 LM；judge_lm=None）。锁定四态故事:
  - perfect          → 全 1.0
  - partial          → process 正非满分 / outcome=0
  - **wrong_decision** → process 全满分但 outcome=0（**phase 5 教学核心**）
  - garbage          → process 跌至 0（除 brainstorm vacuous 1/3）

不锁绝对小数（数值随 fixture 微调可能漂移），只锁:
  ① 4 态在 task_success 上的 0/1 分布
  ② wrong_decision 的 process metrics ≈ perfect 的 process metrics（关键叙事）
  ③ garbage 的 task_success / coverage 全 0 + tool_call set f1 严格 < perfect
"""

from __future__ import annotations

from pathlib import Path

from evals.runner import evaluate_score
from evals.tasks.agent_traj import AgentTraj

PRED_ROOT = Path(__file__).resolve().parent.parent / "data" / "agent_traj" / "predictions"


def _score(pred_name: str) -> dict[str, float]:
    """跑某 pred 的 4 metric aggregated（无 judge）."""
    task = AgentTraj()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return dict(result.aggregated)


# ---------- 上下界 sanity（2 条）-------------------------------------------

def test_perfect_all_metrics_are_one():
    agg = _score("perfect")
    for k in (
        "task_success",
        "tool_call_set_f1",
        "argument_correctness",
        "trajectory_match",
        "trajectory_coverage",
    ):
        assert agg[k] == 1.0, f"{k}={agg[k]}"


def test_garbage_outcome_and_coverage_are_zero():
    agg = _score("garbage")
    assert agg["task_success"] == 0.0
    assert agg["trajectory_coverage"] == 0.0
    # process metric 还有 brainstorm 的 vacuous 1/3 残值（gold 空 vs pred 空 = 1.0）
    # 但绝对值必须 < perfect
    assert agg["tool_call_set_f1"] < 1.0
    assert agg["trajectory_match"] < 1.0


# ---------- partial：process 正非满分 / outcome=0（2 条）-------------------

def test_partial_outcome_zero():
    agg = _score("partial")
    assert agg["task_success"] == 0.0


def test_partial_process_in_middle_band():
    """partial 的 process metric 应在 (0, 1) 之间——既不下界也不上界."""
    agg = _score("partial")
    for k in ("tool_call_set_f1", "argument_correctness", "trajectory_match", "trajectory_coverage"):
        assert 0.0 < agg[k] < 1.0, f"{k}={agg[k]}"


# ---------- wrong_decision：phase 5 教学核心（3 条）------------------------

def test_wrong_decision_outcome_is_zero():
    """所有工具都调到位 + decision 不在白名单 → task_success=0."""
    agg = _score("wrong_decision")
    assert agg["task_success"] == 0.0


def test_wrong_decision_process_metrics_match_perfect():
    """wrong_decision 的 (tool_call_set_f1 / trajectory_match / coverage) ≈ perfect.

    这是 phase 5 反向叙事："tool 调用全对 ≠ 任务对"——process metric 单独看会得到
    完全错误的结论；只有把 outcome metric 一起摆出来才能识破"对程序对决策错"型 agent.
    """
    perfect = _score("perfect")
    wrong = _score("wrong_decision")
    for k in ("tool_call_set_f1", "trajectory_match", "trajectory_coverage"):
        assert wrong[k] == perfect[k], (
            f"{k}: wrong={wrong[k]} vs perfect={perfect[k]}; "
            "wrong_decision 故事核心要求两者在 process 维度完全等价"
        )


def test_wrong_decision_outcome_breaks_process_correlation():
    """同时锁定：wrong_decision 在 process 维度 = perfect，但 outcome 维度 ≠ perfect.

    这是单条断言里同时验证"反向叙事"两侧——进程对、结果错.
    """
    perfect = _score("perfect")
    wrong = _score("wrong_decision")
    assert wrong["tool_call_set_f1"] == perfect["tool_call_set_f1"]
    assert wrong["trajectory_match"] == perfect["trajectory_match"]
    assert wrong["task_success"] != perfect["task_success"]
    assert wrong["task_success"] == 0.0


# ---------- 矩阵 monotonicity（1 条）---------------------------------------

def test_metric_ordering_perfect_above_partial_above_garbage():
    """phase 5 故事矩阵要求 process 维度有粗略阶梯：perfect > partial >= garbage.

    用 trajectory_match 当代表（最 IR 口味、最稳）；不锁绝对小数避免 fixture 漂移.
    """
    perfect = _score("perfect")["trajectory_match"]
    partial = _score("partial")["trajectory_match"]
    garbage = _score("garbage")["trajectory_match"]
    assert perfect > partial >= garbage


# ---------- single-doc 维度 drill-down（1 条）------------------------------

def test_brainstorm_partial_speakers_coverage():
    """brainstorm/partial 只有 1/3 speakers 发言 → coverage(speakers) = 1/3.

    通过 evaluate_score 把 SampleResult.metrics 翻出来锁 per-doc 数值（不只 aggregated）.
    """
    task = AgentTraj()
    result = evaluate_score(task, str(PRED_ROOT / "partial.jsonl"))
    # per_sample 顺序按 gold.jsonl 行序：panel / brainstorm / example
    brainstorm = next(s for s in result.per_sample if s.doc_id == "brainstorm")
    assert abs(brainstorm.metrics["trajectory_coverage"] - 1 / 3) < 1e-9
    assert brainstorm.metrics["task_success"] == 0.0  # 仅 1 speaker，speakers_covered fail
