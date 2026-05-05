"""Phase 7 runner.safety 横切注入锁（非 safety task 也生效）。"""

from __future__ import annotations

from pathlib import Path

from evals.api import Request, Response, Usage
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"


class _CountingFakeLM(LM):
    """报受控 latency/usage，便于验证 safety 与 efficiency 子组不会互相覆盖."""

    def __init__(self) -> None:
        self.name = "fake:counter"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        out: list[Response] = []
        for i, req in enumerate(requests):
            out.append(
                Response(
                    doc_id=req.doc_id,
                    text="positive",
                    latency_ms=float(100 + i * 10),
                    usage=Usage(tokens_in=10, tokens_out=5),
                )
            )
        return out


def test_run_aggregated_contains_safety_subgroup():
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert "safety" in r.aggregated
    assert set(r.aggregated["safety"].keys()) == {
        "refusal_rate",
        "jailbreak_success_rate",
        "over_refusal_rate",
        "judge_safety_score",
    }


def test_run_non_safety_task_unmeasured_stats_are_none():
    """非 safety task（无 safety_category 切片 / 无 judge_lm）→
    refusal_rate=0.0 是真值（heuristic 跑了 mock 的 prediction "positive" 不触发拒答）；
    其它 3 stat None 占位（phase 7 audit P2，区分未测得 vs 真 0）."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    s = r.aggregated["safety"]
    assert s["refusal_rate"] == 0.0  # 真值：mock "positive" 不触发 heuristic
    assert s["jailbreak_success_rate"] is None  # 无 jailbreak 切片样本
    assert s["over_refusal_rate"] is None  # 无 benign 切片样本
    assert s["judge_safety_score"] is None  # 未接 judge_lm


def test_score_aggregated_contains_safety_subgroup():
    r = evaluate_score(SentimentClf(), PRED_DIR / "perfect.jsonl")
    assert "safety" in r.aggregated
    # content class 双路径都挂
    assert set(r.aggregated["safety"].keys()) == {
        "refusal_rate",
        "jailbreak_success_rate",
        "over_refusal_rate",
        "judge_safety_score",
    }


def test_per_sample_metrics_carry_safety_keys_run():
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    sub = r.per_sample[0].metrics["safety"]
    assert isinstance(sub, dict)
    assert sub["refusal_detected"] == 0.0
    assert sub["jailbreak_attempted"] == 0.0


def test_per_sample_metrics_carry_safety_keys_score():
    r = evaluate_score(SentimentClf(), PRED_DIR / "perfect.jsonl")
    sub = r.per_sample[0].metrics["safety"]
    assert isinstance(sub, dict)
    assert sub["refusal_detected"] == 0.0
    assert sub["jailbreak_attempted"] == 0.0


def test_safety_inject_does_not_overwrite_efficiency_subgroup():
    import warnings

    task = SentimentClf()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*fake:counter.*")
        r = evaluate_run(task, _CountingFakeLM())

    m = r.per_sample[0].metrics
    assert "safety" in m and "efficiency" in m
    assert m["safety"]["refusal_detected"] == 0.0
    assert m["efficiency"]["latency_ms"] == 100.0


# ---------- 结构性焊接锁（plan §4.4 重构锁） ----------

def test_inner_helper_invoked_by_both_modes(monkeypatch):
    """Plan §4.4 重构锁：score / run 双入口都必须经过 _evaluate_inner.

    用 monkey-patch 替换 runner._evaluate_inner，分别调 evaluate_score / evaluate_run
    后该 patched helper 必被调用——结构性保证"中段不可能被绕过"，防止后续有人在某个
    入口直接 `return EvalResult(...)` 跳过 cross-cutting injectors.
    """
    from pathlib import Path as _Path

    from evals import runner as _runner_mod

    calls: list[str] = []
    real_inner = _runner_mod._evaluate_inner

    def _spy_inner(*args, **kwargs):
        calls.append(kwargs.get("mode", "?"))
        return real_inner(*args, **kwargs)

    monkeypatch.setattr(_runner_mod, "_evaluate_inner", _spy_inner)

    task = SentimentClf()
    docs = list(task.docs())

    _runner_mod.evaluate_score(
        task, _Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions" / "perfect.jsonl"
    )
    _runner_mod.evaluate_run(task, MockLM(mode="gold", docs=docs))

    assert "score" in calls, "evaluate_score 必须经过 _evaluate_inner"
    assert "run" in calls, "evaluate_run 必须经过 _evaluate_inner"


def test_safety_inject_runs_before_efficiency(monkeypatch):
    """Plan §4.4 顺序锁：content class（safety）必须先于 call class（efficiency）注入.

    监听两 injector 的调用顺序：safety 调完之后再调 efficiency.
    顺序错了会导致 efficiency 子组覆盖（理论上）safety 字段——虽然 nested 派下不会真覆盖，
    但顺序约定本身是 ontology 二分的代码层落点，必须焊死.
    """
    from evals import runner as _runner_mod

    call_order: list[str] = []
    real_safety = _runner_mod.inject_per_sample_safety
    real_eff = _runner_mod.inject_per_sample_efficiency

    def _spy_safety(srs, resps):
        call_order.append("safety")
        return real_safety(srs, resps)

    def _spy_eff(srs, resps, model_label):
        call_order.append("efficiency")
        return real_eff(srs, resps, model_label)

    monkeypatch.setattr(_runner_mod, "inject_per_sample_safety", _spy_safety)
    monkeypatch.setattr(_runner_mod, "inject_per_sample_efficiency", _spy_eff)

    task = SentimentClf()
    docs = list(task.docs())
    evaluate_run(task, MockLM(mode="gold", docs=docs))

    assert call_order == ["safety", "efficiency"], (
        f"顺序违反 ontology 二分约定：content class (safety) 必须先于 call class (efficiency)，"
        f"实际 {call_order}"
    )
