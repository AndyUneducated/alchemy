"""Phase 7 safety task 的 score 路径矩阵锁（5 份 predictions）。"""

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
    s = r.aggregated["safety"]
    # 当前 safe fixtures 中 10/15 样本命中 refusal 关键词（heuristic 口径）。
    assert s["refusal_rate"] == pytest.approx(10.0 / 15.0)
    # jailbreak 切片仍有 1 条未命中 refusal（s11），故成功率为 1/5。
    assert s["jailbreak_success_rate"] == pytest.approx(1.0 / 5.0)
    # benign 切片有样本，全部正常回答（无 refuse 关键词）→ over_refusal_rate=0
    assert s["over_refusal_rate"] == 0.0
    # 未传 judge_lm → None（phase 7 audit P2，与"真 0 分"区分）
    assert s["judge_safety_score"] is None


def test_over_refuse_predictions_safety_stats():
    r = _score("over_refuse")
    s = r.aggregated["safety"]
    assert s["refusal_rate"] == 1.0
    assert s["jailbreak_success_rate"] == 0.0
    assert s["over_refusal_rate"] == 1.0  # benign 全拒
    assert s["judge_safety_score"] is None  # 未传 judge_lm


def test_jailbreak_success_predictions_safety_stats():
    r = _score("jailbreak_success")
    s = r.aggregated["safety"]
    assert s["jailbreak_success_rate"] == 1.0
    assert s["refusal_rate"] < 0.2
    # benign 切片有样本，全部"绕行风格"回答（无 refuse 关键词）→ over_refusal_rate=0 真值
    assert s["over_refusal_rate"] == 0.0
    assert s["judge_safety_score"] is None


def test_evasive_predictions_refusal_high_but_judge_low():
    # heuristic 会因“不能/无法”关键词判高拒答，judge（常量 2）给出低安全分，形成反向叙事。
    r = _score("evasive", judge=_ConstJudgeLM(score=2))
    s = r.aggregated["safety"]
    assert s["refusal_rate"] >= 0.6
    assert s["jailbreak_success_rate"] <= 0.2
    assert s["judge_safety_score"] == 2.0


def test_garbage_predictions_real_zero_not_unmeasured():
    """garbage prediction 短文本 < MIN_RESPONSE_LEN → heuristic 真跑出 0；
    refusal_rate=0 是合法值（既不 refuse 也不 jailbreak attempted），不是"未测得"."""
    r = _score("garbage")
    s = r.aggregated["safety"]
    # 真 0 真值（heuristic 跑了，判都不触发）
    assert s["refusal_rate"] == 0.0
    assert s["jailbreak_success_rate"] == 0.0
    assert s["over_refusal_rate"] == 0.0
    # 但 judge 没接 → None（phase 7 audit P2 区分）
    assert s["judge_safety_score"] is None


def test_aggregated_safety_subgroup_present_in_score_mode():
    r = _score("safe")
    assert "safety" in r.aggregated
    assert set(r.aggregated["safety"].keys()) == {
        "refusal_rate",
        "jailbreak_success_rate",
        "over_refusal_rate",
        "judge_safety_score",
    }


def test_per_sample_metrics_carry_safety_subgroup():
    r = _score("safe")
    s0 = r.per_sample[0]
    assert "safety" in s0.metrics
    sub = s0.metrics["safety"]
    assert isinstance(sub, dict)
    assert "refusal_detected" in sub
    assert "jailbreak_attempted" in sub
