"""runner._evaluate_inner 挂 efficiency.judge.* 子组的端到端锁（DECISIONS §7.3）.

锁定：
  1. task 没接 judge_lm → aggregated 不出现 efficiency.judge 子组
  2. task 接 judge_lm + run 路径 → efficiency 含 task 部分（latency_ms 等）+ judge 子组
  3. task 接 judge_lm + score 路径 → efficiency 仅含 judge 子组（无 task 部分）
  4. judge 调用次数与 sample 是 N:M 关系（pointwise 1:1 / g_eval n_dim×n_samples / RAG n_claim+1 等）
  5. efficiency.judge 4 子组 schema 与 efficiency 顶层同形（latency_ms / tokens_in / tokens_out / cost_usd）
"""

from __future__ import annotations

from pathlib import Path

from evals.api import Request, Response, Usage
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.qa_open import QAOpen

QA_PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


class _FakeJudgeLM(LM):
    """返回固定 4 分 + 受控 latency / usage。"""

    def __init__(self, label: str = "fake:judge") -> None:
        self.name = label
        self._counter = 0

    def generate_until(self, requests: list[Request]) -> list[Response]:
        out: list[Response] = []
        for r in requests:
            self._counter += 1
            out.append(
                Response(
                    doc_id=r.doc_id,
                    text="4",
                    latency_ms=200.0,
                    usage=Usage(tokens_in=30, tokens_out=2),
                )
            )
        return out


def test_no_judge_no_judge_subgroup():
    """task 没接 judge_lm → aggregated 不应出现 efficiency.judge 子组."""
    docs = list(QAOpen().docs())
    r = evaluate_run(QAOpen(), MockLM(mode="gold", docs=docs))
    assert "efficiency" in r.aggregated
    assert "judge" not in r.aggregated["efficiency"]


def test_run_with_judge_has_both_task_and_judge_efficiency():
    """run 路径 + judge：efficiency 顶层含 task 4 子组 + judge 子组."""
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    # task 部分（被测物 call class）
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd"} <= eff.keys()
    # judge 部分（评估工具 call class）
    assert "judge" in eff
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd"} <= eff["judge"].keys()


def test_score_with_judge_has_only_judge_efficiency():
    """score 路径 + judge：efficiency 仅含 judge 子组（无 task 部分）—— DECISIONS §7.3 wave 3."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_score(task, QA_PRED_DIR / "perfect.jsonl")
    assert "efficiency" in r.aggregated
    eff = r.aggregated["efficiency"]
    assert "judge" in eff
    # task 部分不应出现（被测物 call class 仅 run 挂）
    assert "latency_ms" not in eff
    assert "tokens_in" not in eff
    # judge 子组数值非全 0（_FakeJudgeLM 报了 latency=200 + tokens 30/2）
    assert eff["judge"]["latency_ms"]["mean"] == 200.0
    assert eff["judge"]["tokens_in"]["total"] > 0


def test_judge_efficiency_call_count_matches_sample_count_for_pointwise():
    """qa_open 是 pointwise judge：1 sample = 1 judge call —— tokens_in.total = sample 数 × 30."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_score(task, QA_PRED_DIR / "perfect.jsonl")
    judge_eff = r.aggregated["efficiency"]["judge"]
    assert judge_eff["tokens_in"]["total"] == r.n * 30
    assert judge_eff["tokens_out"]["total"] == r.n * 2


def test_judge_efficiency_schema_matches_task_efficiency():
    """efficiency.judge 4 子组形态与 efficiency 顶层（被测物）同形 schema-on-write."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    docs = list(QAOpen().docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    judge_eff = eff["judge"]
    # latency_ms 4 stat
    assert {"mean", "p50", "p95", "max"} == set(judge_eff["latency_ms"].keys())
    # tokens_in/out 双 stat
    assert {"total", "mean"} == set(judge_eff["tokens_in"].keys())
    assert {"total", "mean"} == set(judge_eff["tokens_out"].keys())
    # cost_usd 双 stat
    assert {"total", "mean"} == set(judge_eff["cost_usd"].keys())
