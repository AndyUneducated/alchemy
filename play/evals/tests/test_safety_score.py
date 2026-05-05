"""Phase 7 safety task 的 score 路径矩阵锁（5 份 predictions）.

wave 3（DECISIONS §7.2）：撤销 cross-cutting AOP；safety task 自己 own metrics。
访问路径从嵌套 `r.aggregated["safety"]["x"]` 改为 flat 顶层 `r.aggregated["x"]`；
sample.metrics 从嵌套 `metrics["safety"]["refusal_detected"]` 改为 flat
`metrics["refusal_detected"]`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.api import Request, Response
from evals.models.base import LM
from evals.runner import evaluate_score
from evals.tasks.safety import Safety

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "safety" / "predictions"


class _ConstJudgeLM(LM):
    """返回固定 1-5 分，锁住 judge_safety_score 聚合行为."""

    def __init__(self, score: int) -> None:
        self.name = f"fake:judge:{score}"
        self._text = str(score)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        return [Response(doc_id=r.doc_id, text=self._text) for r in requests]


def _score(pred_name: str, *, judge: LM | None = None):
    task = Safety(judge_lm=judge) if judge is not None else Safety()
    return evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")


def test_safe_predictions_safety_stats():
    r = _score("safe")
    # 当前 safe fixtures 中 10/15 样本命中 refusal 关键词（heuristic 口径）。
    assert r.aggregated["refusal_rate"] == pytest.approx(10.0 / 15.0)
    # jailbreak 切片仍有 1 条未命中 refusal（s11），故成功率为 1/5。
    assert r.aggregated["jailbreak_success_rate"] == pytest.approx(1.0 / 5.0)
    # benign 切片有样本，全部正常回答（无 refuse 关键词）→ over_refusal_rate=0
    assert r.aggregated["over_refusal_rate"] == 0.0
    # 未传 judge_lm → None（None vs 0 区分协议保留）
    assert r.aggregated["judge_safety_score"] is None


def test_over_refuse_predictions_safety_stats():
    r = _score("over_refuse")
    assert r.aggregated["refusal_rate"] == 1.0
    assert r.aggregated["jailbreak_success_rate"] == 0.0
    assert r.aggregated["over_refusal_rate"] == 1.0  # benign 全拒
    assert r.aggregated["judge_safety_score"] is None  # 未传 judge_lm


def test_jailbreak_success_predictions_safety_stats():
    r = _score("jailbreak_success")
    assert r.aggregated["jailbreak_success_rate"] == 1.0
    assert r.aggregated["refusal_rate"] < 0.2
    # benign 切片有样本，全部"绕行风格"回答（无 refuse 关键词）→ over_refusal_rate=0 真值
    assert r.aggregated["over_refusal_rate"] == 0.0
    assert r.aggregated["judge_safety_score"] is None


def test_evasive_predictions_refusal_high_but_judge_low():
    # heuristic 会因"不能/无法"关键词判高拒答，judge（常量 2）给出低安全分，形成反向叙事。
    r = _score("evasive", judge=_ConstJudgeLM(score=2))
    assert r.aggregated["refusal_rate"] >= 0.6
    assert r.aggregated["jailbreak_success_rate"] <= 0.2
    assert r.aggregated["judge_safety_score"] == 2.0


def test_garbage_predictions_real_zero_not_unmeasured():
    """garbage prediction 短文本 < MIN_RESPONSE_LEN → heuristic 真跑出 0；
    refusal_rate=0 是合法值（既不 refuse 也不 jailbreak attempted），不是"未测得"."""
    r = _score("garbage")
    # 真 0 真值（heuristic 跑了，判都不触发）
    assert r.aggregated["refusal_rate"] == 0.0
    assert r.aggregated["jailbreak_success_rate"] == 0.0
    assert r.aggregated["over_refusal_rate"] == 0.0
    # 但 judge 没接 → None
    assert r.aggregated["judge_safety_score"] is None


def test_aggregated_4_stat_present_in_score_mode():
    """safety task 顶层 4 stat 平铺（task-specific metric，与 sentiment_clf 同形）."""
    r = _score("safe")
    assert "refusal_rate" in r.aggregated
    assert "jailbreak_success_rate" in r.aggregated
    assert "over_refusal_rate" in r.aggregated
    assert "judge_safety_score" in r.aggregated
    # 不应再有嵌套 "safety" 子组（wave 3 删除）
    assert "safety" not in r.aggregated


def test_per_sample_metrics_carry_safety_keys_flat():
    """sample.metrics flat 平铺：直接访问 refusal_detected / jailbreak_attempted（不嵌套）."""
    r = _score("safe")
    s0 = r.per_sample[0]
    assert "refusal_detected" in s0.metrics
    assert "jailbreak_attempted" in s0.metrics
    # 不应再有嵌套 "safety" 子组
    assert "safety" not in s0.metrics
