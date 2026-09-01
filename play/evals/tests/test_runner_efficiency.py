"""Phase 6 runner.efficiency end-to-end lock:

  - run mode: aggregated contains efficiency subgroups (schema always 4 subgroups)
  - score mode: aggregated without efficiency (cross-cutting only run injection)
  - LM reports real numbers (fake LM) → aggregated mathematically correct + per-sample SampleResult.metrics copy
  - LM is not reported (MockLM) → the subgroup key values are all 0 but the schema is

Does not rely on ollama live; use fake LM to inject controlled latency/usage fields."""

from __future__ import annotations

import json
from pathlib import Path

from evals.api import Request, Response, Usage
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"


# ---------- fake LM: Report controlled latency/usage but the answer is arbitrary ----------

class _CountingFakeLM(LM):
    """Each request returns fixed text + auto-increment latency_ms + fixed tokens_in/out → aggregated and can be calculated by hand."""

    def __init__(self, label: str = "fake:counter") -> None:
        self.name = label

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


# ---------- run mode ----------

def test_run_aggregated_contains_efficiency_subgroup():
    """Regardless of whether LM is reported or not in the run path, aggregated always has an efficiency subgroup (schema is stable)."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert "efficiency" in r.aggregated
    eff = r.aggregated["efficiency"]
    assert set(eff.keys()) == {"latency_ms", "tokens_in", "tokens_out", "cost_usd"}


def test_run_with_mock_lm_reports_zero_efficiency():
    """MockLM does not report latency / usage → 4 subgroup key values ​​are all 0 (including audit §1.1 cost.mean / §1.2 latency.max)."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    assert eff["latency_ms"]["mean"] == 0.0
    assert eff["latency_ms"]["max"] == 0.0  # audit §1.2
    assert eff["tokens_in"]["total"] == 0    # audit §1.5: int semantics
    assert eff["cost_usd"]["total"] == 0.0
    assert eff["cost_usd"]["mean"] == 0.0    # audit §1.1


def test_run_with_real_signals_aggregates_correctly():
    """Fake LM reports latency=100,110,...; tokens_in=10×N, tokens_out=5×N → hand calculation oracle lock."""
    import warnings
    task = SentimentClf()
    docs = list(task.docs())
    n = len(docs)
    # fake:counter not in PRICE_TABLE → triggers audit §1.4 unknown-pricing-model warning (expected)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*fake:counter.*")
        r = evaluate_run(task, _CountingFakeLM())

    eff = r.aggregated["efficiency"]
    expected_mean_latency = sum(100 + i * 10 for i in range(n)) / n
    assert eff["latency_ms"]["mean"] == expected_mean_latency
    assert eff["latency_ms"]["max"] == float(100 + (n - 1) * 10)  # The last one is the biggest
    assert eff["tokens_in"]["total"] == 10 * n
    assert eff["tokens_in"]["mean"] == 10.0
    assert eff["tokens_out"]["total"] == 5 * n
    # fake:counter not in PRICE_TABLE → cost 0.0 (warning triggered, see test_metrics_efficiency)
    assert eff["cost_usd"]["total"] == 0.0
    assert eff["cost_usd"]["mean"] == 0.0


def test_run_per_sample_metrics_carry_efficiency_fields():
    """The per-sample measured values ​​go into the SampleResult.metrics["efficiency"] subgroup (phase 7 §7.D nested)."""
    import warnings
    task = SentimentClf()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*fake:counter.*")
        r = evaluate_run(task, _CountingFakeLM())
    s0 = r.per_sample[0]
    eff = s0.metrics["efficiency"]
    assert isinstance(eff, dict)
    assert eff["latency_ms"] == 100.0
    assert eff["tokens_in"] == 10.0
    assert eff["tokens_out"] == 5.0
    assert eff["cost_usd"] == 0.0


def test_run_mock_lm_per_sample_metrics_carry_zero_padded_efficiency():
    """audit §1.3 Option A: mock path per-sample.metrics["efficiency"] subgroup also writes 4 placeholders (schema-on-write is consistent at both levels).
    phase 7 §7.D nested faction: access the path s.metrics["efficiency"]["latency_ms"] to avoid KeyError."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    s0 = r.per_sample[0]
    eff = s0.metrics["efficiency"]
    assert isinstance(eff, dict)
    assert eff["latency_ms"] == 0.0
    assert eff["tokens_in"] == 0.0
    assert eff["tokens_out"] == 0.0
    assert eff["cost_usd"] == 0.0


# ---------- score mode ----------

def test_score_aggregated_lacks_efficiency_subgroup():
    """Score path has no LM calls → does not inject the efficiency subgroup (explicit yield instead of all 0 placeholders)."""
    task = SentimentClf()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert "efficiency" not in r.aggregated


# ---------- result.json serialization ----------

def test_run_result_json_round_trip_preserves_efficiency(tmp_path: Path):
    """Drop result.json → re-load → efficiency to keep the nested structure intact (asdict shallow copy is enough)."""
    from dataclasses import asdict

    task = SentimentClf()
    r = evaluate_run(task, _CountingFakeLM())
    p = tmp_path / "result.json"
    p.write_text(json.dumps(asdict(r), default=str), encoding="utf-8")

    loaded = json.loads(p.read_text(encoding="utf-8"))
    eff = loaded["aggregated"]["efficiency"]
    assert "latency_ms" in eff and "p50" in eff["latency_ms"]
    assert eff["tokens_in"]["total"] > 0
