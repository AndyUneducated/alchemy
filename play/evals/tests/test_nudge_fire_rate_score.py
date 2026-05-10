"""nudge_fire_rate × 3 stub × 7 docs = 21 sample 矩阵：task 端到端守门员.

跑 evaluate_score 端到端：load gold.jsonl + pred.jsonl → metrics → aggregated.
锁三态故事 + breakdown 字典 + reverse metric 方向：

  | 预测       | nudge_fire_rate | by_failure_mode 主桶          | 故事 |
  |---|---|---|---|
  | perfect    | 0.0             | 三桶皆 0                       | 上界 sanity（模型 100% 第一次到位）|
  | all_nudged | 1.0             | missed 主导                    | 下界 sanity（模型完全沉默）|
  | mixed      | (0,1)           | missed + wrong_tool 各有       | 中间态 + 失败分类信号 |

不锁绝对小数（fixture 微调可能漂移），只锁：
  ① 上下界严格（perfect=0 / all_nudged=1）
  ② mixed 严格 ∈ (0, 1)
  ③ by_scenario 维度 brainstorm/debate/roundtable 永远 None（vacuous）
  ④ by_tool 字典含 cast_vote / append_section / retrieve_docs（后者来自 1.B
     新增的 require_tool 密集 scenario，依赖 agent_engine DECISIONS §12 解除
     "require_tool 仅 artifact 工具"限制）
  ⑤ by_failure_mode 三桶都列出（含 wrong_args=0 的 deferred 占位）
  ⑥ 总 require_tool turn 数 = 20（panel 4 + example 3 + tool_chain 5 + code_review 8）
"""

from __future__ import annotations

from pathlib import Path

from evals.runner import evaluate_score
from evals.tasks.nudge_fire_rate import NudgeFireRate

PRED_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "nudge_fire_rate" / "predictions"
)


def _score(pred_name: str) -> dict:
    """跑某 pred 的 aggregated 字典（含 nested by_* breakdowns）."""
    task = NudgeFireRate()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return dict(result.aggregated)


def _per_sample_rates(pred_name: str) -> dict[str, float | None]:
    """跑某 pred 的 per-doc nudge_fire_rate 字典（drill-down 锁 by_scenario）."""
    task = NudgeFireRate()
    result = evaluate_score(task, str(PRED_ROOT / f"{pred_name}.jsonl"))
    return {s.doc_id: s.metrics["nudge_fire_rate"] for s in result.per_sample}


# ---------- 上下界 sanity（4 条）-------------------------------------------

def test_perfect_aggregated_rate_is_zero():
    agg = _score("perfect")
    assert agg["nudge_fire_rate"] == 0.0
    assert agg["nudge_fire_count"] == 0.0


def test_perfect_failure_mode_buckets_all_zero():
    agg = _score("perfect")
    by_mode = agg["by_failure_mode"]
    assert by_mode == {"missed": 0, "wrong_tool": 0, "wrong_args": 0}


def test_all_nudged_aggregated_rate_is_one():
    agg = _score("all_nudged")
    assert agg["nudge_fire_rate"] == 1.0
    # 20 个 require_tool turn（example 3 + panel 4 + tool_chain 5 + code_review 8）
    # 后两者由 agent_sft phase 1.B 引入，依赖 agent_engine DECISIONS §12 解除范围限制
    assert agg["nudge_fire_count"] == 20.0
    assert agg["require_tool_total"] == 20.0


def test_all_nudged_dominated_by_missed_mode():
    agg = _score("all_nudged")
    by_mode = agg["by_failure_mode"]
    assert by_mode["missed"] == 20
    assert by_mode["wrong_tool"] == 0
    assert by_mode["wrong_args"] == 0


# ---------- mixed 中间态（3 条）-------------------------------------------

def test_mixed_aggregated_rate_in_open_interval():
    agg = _score("mixed")
    rate = agg["nudge_fire_rate"]
    assert 0.0 < rate < 1.0


def test_mixed_failure_modes_have_both_missed_and_wrong_tool():
    """mixed stub 设计上 missed + wrong_tool 都该有计数（验证两桶都活跃）."""
    agg = _score("mixed")
    by_mode = agg["by_failure_mode"]
    assert by_mode["missed"] > 0
    assert by_mode["wrong_tool"] > 0
    assert by_mode["wrong_args"] == 0  # Phase 1 deferred，恒 0


def test_mixed_strictly_between_perfect_and_all_nudged():
    """单调性：perfect ≤ mixed < all_nudged ∈ {0, x ∈ (0,1), 1}."""
    perfect = _score("perfect")["nudge_fire_rate"]
    mixed = _score("mixed")["nudge_fire_rate"]
    nudged = _score("all_nudged")["nudge_fire_rate"]
    assert perfect == 0.0
    assert nudged == 1.0
    assert perfect < mixed < nudged


# ---------- by_scenario / by_tool breakdown（3 条）-------------------------

def test_by_scenario_vacuous_scenarios_are_none():
    """3 个无 require_tool 的 scenario 永远 None（vacuous）→ breakdown 显式渲染."""
    agg = _score("all_nudged")
    by_scn = agg["by_scenario"]
    assert by_scn["brainstorm"] is None
    assert by_scn["debate"] is None
    assert by_scn["roundtable"] is None
    # 4 个有 require_tool 的 scenario 都是 1.0
    assert by_scn["example"] == 1.0
    assert by_scn["panel"] == 1.0
    assert by_scn["tool_chain"] == 1.0
    assert by_scn["code_review"] == 1.0


def test_by_tool_breakdown_contains_three_tools():
    """all_nudged 下 by_tool 三键齐全：append_section + cast_vote + retrieve_docs.
    retrieve_docs 是非 artifact 工具，依赖 DECISIONS §12 才能进入 require_tool 检查面."""
    agg = _score("all_nudged")
    by_tool = agg["by_tool"]
    assert set(by_tool.keys()) == {"append_section", "cast_vote", "retrieve_docs"}
    assert by_tool["append_section"] == 1.0
    assert by_tool["cast_vote"] == 1.0
    assert by_tool["retrieve_docs"] == 1.0


def test_perfect_per_sample_rates_correctly_split():
    """per-sample drill-down：perfect 下 4 个 require_tool scenario rate=0、3 个 vacuous None."""
    rates = _per_sample_rates("perfect")
    assert rates == {
        "brainstorm": None, "debate": None, "roundtable": None,
        "example": 0.0, "panel": 0.0, "tool_chain": 0.0, "code_review": 0.0,
    }


# ---------- 数据契约 sanity（2 条）-----------------------------------------

def test_docs_loading_order_matches_gold():
    """gold.jsonl 行序：vacuous 先（smoke 友好）→ require_tool 后；require_tool
    内部按 turn 数升序（example 3 → panel 4 → tool_chain 5 → code_review 8）.
    --limit 1 命中 brainstorm（最快）, --limit 7 全跑."""
    docs = list(NudgeFireRate().docs())
    assert [d.id for d in docs] == [
        "brainstorm", "debate", "roundtable", "example", "panel",
        "tool_chain", "code_review",
    ]


def test_higher_is_better_marks_rate_as_lower_better():
    """nudge_fire_rate 是反向 metric（越低越好），与项目其它 metric 高=好约定相反."""
    h = NudgeFireRate().higher_is_better()
    assert h["nudge_fire_rate"] is False
    assert h["nudge_fire_count"] is False
