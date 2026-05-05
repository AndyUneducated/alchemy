"""Phase 7 safety task 的 run 路径锁.

wave 3（DECISIONS §7.2）：safety = standalone task；aggregated 顶层 flat 平铺
（refusal_rate / jailbreak_success_rate / over_refusal_rate / judge_safety_score），
不再嵌套在 `aggregated["safety"]` 子组下.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evals.api import Request, Response
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.safety import Safety

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "safety" / "predictions"

# safety task 的 4 stat 顶层 key 集合（wave 3 起 flat 平铺）
_SAFETY_4_STAT_KEYS = {
    "refusal_rate",
    "jailbreak_success_rate",
    "over_refusal_rate",
    "judge_safety_score",
}


class _ConstJudgeLM(LM):
    def __init__(self, score: int) -> None:
        self.name = f"fake:judge:{score}"
        self._text = str(score)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        return [Response(doc_id=r.doc_id, text=self._text) for r in requests]


def test_run_mock_gold_parity_perfect_predictions():
    """safety task parity：run(mock:gold) 与 score(由 gold 即时生成的 perfect predictions) 相等."""
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "prediction": d.target}, ensure_ascii=False) + "\n")
        path = Path(f.name)
    r_score = evaluate_score(Safety(), path)

    # 4 stat 平铺顶层均存在
    assert _SAFETY_4_STAT_KEYS <= r_run.aggregated.keys()
    assert _SAFETY_4_STAT_KEYS <= r_score.aggregated.keys()

    # task-specific metrics 在双模式下相等
    for k in _SAFETY_4_STAT_KEYS:
        assert r_run.aggregated[k] == r_score.aggregated[k], f"parity mismatch on {k!r}"

    # run 路径多 efficiency（call class）；score 无
    assert "efficiency" in r_run.aggregated
    assert "efficiency" not in r_score.aggregated


def test_run_with_judge_outputs_judge_score():
    docs = list(Safety().docs())
    task = Safety(judge_lm=_ConstJudgeLM(score=4))
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert r.aggregated["judge_safety_score"] == 4.0


def test_run_safety_4_stat_present_both_modes():
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    assert _SAFETY_4_STAT_KEYS <= r_run.aggregated.keys()
    assert _SAFETY_4_STAT_KEYS <= r_score.aggregated.keys()


def test_safety_aggregated_subgroup_no_longer_present():
    """wave 3 decisive 锁：aggregated 不再有嵌套 'safety' 子组（task-specific 平铺）."""
    docs = list(Safety().docs())
    r_run = evaluate_run(Safety(), MockLM(mode="gold", docs=docs))
    r_score = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    assert "safety" not in r_run.aggregated
    assert "safety" not in r_score.aggregated


def test_safety_self_handles_long_answer_correctly():
    """A1 wave 3 修复核心：safety task 自己跑 heuristic（与 cross-cutting AOP 删除无关），
    长答案在 safety task 内仍能正常 refusal_detected / jailbreak_attempted——这条锁
    safety task 本身的功能不被 wave 3 删 cross-cutting 影响."""
    r = evaluate_score(Safety(), PRED_DIR / "safe.jsonl")
    # 至少有一些样本 refusal_detected=1（safe.jsonl 含拒答样本）
    refusal_set = {s.metrics.get("refusal_detected") for s in r.per_sample}
    assert 1.0 in refusal_set
