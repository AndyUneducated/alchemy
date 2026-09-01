"""runner._evaluate_inner holds an end-to-end lock on efficiency.judge.* subgroups (DECISIONS §7.3).

Lock:
  1. task does not receive judge_lm → aggregated does not appear in the efficiency.judge subgroup
  2. task connects to judge_lm + run path → efficiency contains task part (latency_ms, etc.) + judge subgroup
  3. task connects to judge_lm + score path → efficiency only contains the judge subgroup (no task part)
  4. The relationship between the number of judge calls and sample is N:M (pointwise 1:1 / g_eval n_dim×n_samples / RAG n_claim+1, etc.)
  5. efficiency.judge 4 subgroup schema is the same as the efficiency top level (latency_ms / tokens_in / tokens_out / cost_usd)"""

from __future__ import annotations

from pathlib import Path

from evals.api import Request, Response, Usage
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.qa_open import QAOpen

QA_PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


class _FakeJudgeLM(LM):
    """Returns fixed 4 points + controlled latency / usage."""

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
    """task is not connected to judge_lm → aggregated and the efficiency.judge subgroup should not appear."""
    docs = list(QAOpen().docs())
    r = evaluate_run(QAOpen(), MockLM(mode="gold", docs=docs))
    assert "efficiency" in r.aggregated
    assert "judge" not in r.aggregated["efficiency"]


def test_run_with_judge_has_both_task_and_judge_efficiency():
    """run path + judge:efficiency top level contains task 4 subgroup + judge subgroup."""
    docs = list(QAOpen().docs())
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    # task part (object under test call class)
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd"} <= eff.keys()
    # judge part (evaluation tool call class)
    assert "judge" in eff
    assert {"latency_ms", "tokens_in", "tokens_out", "cost_usd"} <= eff["judge"].keys()


def test_score_with_judge_has_only_judge_efficiency():
    """score path + judge:efficiency contains only the judge subgroup (no task part) - DECISIONS §7.3 wave 3."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_score(task, QA_PRED_DIR / "perfect.jsonl")
    assert "efficiency" in r.aggregated
    eff = r.aggregated["efficiency"]
    assert "judge" in eff
    # The task part should not appear (the object under test call class only runs)
    assert "latency_ms" not in eff
    assert "tokens_in" not in eff
    # The value of the judge subgroup is not all 0 (_FakeJudgeLM reported latency=200 + tokens 30/2)
    assert eff["judge"]["latency_ms"]["mean"] == 200.0
    assert eff["judge"]["tokens_in"]["total"] > 0


def test_judge_efficiency_call_count_matches_sample_count_for_pointwise():
    """qa_open is pointwise judge: 1 sample = 1 judge call —— tokens_in.total = number of samples × 30."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    r = evaluate_score(task, QA_PRED_DIR / "perfect.jsonl")
    judge_eff = r.aggregated["efficiency"]["judge"]
    assert judge_eff["tokens_in"]["total"] == r.n * 30
    assert judge_eff["tokens_out"]["total"] == r.n * 2


def test_judge_efficiency_schema_matches_task_efficiency():
    """The efficiency.judge 4 subgroup shape is the same as the efficiency top level (the object under test) schema-on-write."""
    task = QAOpen(judge_lm=_FakeJudgeLM())
    docs = list(QAOpen().docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    judge_eff = eff["judge"]
    # latency_ms 4 stat
    assert {"mean", "p50", "p95", "max"} == set(judge_eff["latency_ms"].keys())
    # tokens_in/out double stat
    assert {"total", "mean"} == set(judge_eff["tokens_in"].keys())
    assert {"total", "mean"} == set(judge_eff["tokens_out"].keys())
    # cost_usd double stat
    assert {"total", "mean"} == set(judge_eff["cost_usd"].keys())
