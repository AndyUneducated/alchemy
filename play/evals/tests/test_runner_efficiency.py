"""Phase 6 runner.efficiency 端到端锁：

  - run 模式：aggregated 包含 efficiency 子组（schema 永远 4 子组）
  - score 模式：aggregated 不含 efficiency（cross-cutting 仅 run 注入）
  - LM 报实数（fake LM）→ aggregated 数学正确 + per-sample SampleResult.metrics 拷贝
  - LM 不报（MockLM）→ 子组键值全 0 但 schema 在

不依赖 ollama live；用 fake LM 注入受控 latency/usage 字段.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.api import Request, Response, Usage
from evals.models.base import LM
from evals.models.mock import MockLM
from evals.runner import evaluate_run, evaluate_score
from evals.tasks.sentiment_clf import SentimentClf

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "predictions"


# ---------- fake LM：报受控 latency/usage 但答案随便 ----------

class _CountingFakeLM(LM):
    """每条 request 回固定文本 + 自增 latency_ms + 固定 tokens_in/out → aggregated 可手算."""

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


# ---------- run 模式 ----------

def test_run_aggregated_contains_efficiency_subgroup():
    """run 路径无论 LM 是否报，aggregated 永远有 efficiency 子组（schema 稳定）."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    assert "efficiency" in r.aggregated
    eff = r.aggregated["efficiency"]
    assert set(eff.keys()) == {"latency_ms", "tokens_in", "tokens_out", "cost_usd"}


def test_run_with_mock_lm_reports_zero_efficiency():
    """MockLM 不报 latency / usage → 4 子组键值全 0（含 audit §1.1 cost.mean / §1.2 latency.max）."""
    task = SentimentClf()
    docs = list(task.docs())
    r = evaluate_run(task, MockLM(mode="gold", docs=docs))
    eff = r.aggregated["efficiency"]
    assert eff["latency_ms"]["mean"] == 0.0
    assert eff["latency_ms"]["max"] == 0.0  # audit §1.2
    assert eff["tokens_in"]["total"] == 0    # audit §1.5: int 语义
    assert eff["cost_usd"]["total"] == 0.0
    assert eff["cost_usd"]["mean"] == 0.0    # audit §1.1


def test_run_with_real_signals_aggregates_correctly():
    """fake LM 报 latency=100,110,...; tokens_in=10×N, tokens_out=5×N → 手算 oracle 锁住."""
    import warnings
    task = SentimentClf()
    docs = list(task.docs())
    n = len(docs)
    # fake:counter 不在 PRICE_TABLE → 触发 audit §1.4 unknown-pricing-model warning（预期）
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*fake:counter.*")
        r = evaluate_run(task, _CountingFakeLM())

    eff = r.aggregated["efficiency"]
    expected_mean_latency = sum(100 + i * 10 for i in range(n)) / n
    assert eff["latency_ms"]["mean"] == expected_mean_latency
    assert eff["latency_ms"]["max"] == float(100 + (n - 1) * 10)  # 最后一条最大
    assert eff["tokens_in"]["total"] == 10 * n
    assert eff["tokens_in"]["mean"] == 10.0
    assert eff["tokens_out"]["total"] == 5 * n
    # fake:counter 不在 PRICE_TABLE → cost 0.0（warning 已触发，见 test_metrics_efficiency）
    assert eff["cost_usd"]["total"] == 0.0
    assert eff["cost_usd"]["mean"] == 0.0


def test_run_per_sample_metrics_carry_efficiency_fields():
    """per-sample 实测值进 SampleResult.metrics["efficiency"] 子组（phase 7 §7.D nested 派）."""
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
    """audit §1.3 选项 A：mock 路径 per-sample.metrics["efficiency"] 子组也写 4 占位（schema-on-write 两层一致）.
    phase 7 §7.D nested 派：访问路径 s.metrics["efficiency"]["latency_ms"]，避免 KeyError.
    """
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


# ---------- score 模式 ----------

def test_score_aggregated_lacks_efficiency_subgroup():
    """score 路径无 LM 调用 → 不注入 efficiency 子组（显式让步而非全 0 占位）."""
    task = SentimentClf()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert "efficiency" not in r.aggregated


# ---------- result.json 序列化 ----------

def test_run_result_json_round_trip_preserves_efficiency(tmp_path: Path):
    """落 result.json → re-load → efficiency 嵌套结构完整保留（asdict 浅拷贝足够）."""
    from dataclasses import asdict

    task = SentimentClf()
    r = evaluate_run(task, _CountingFakeLM())
    p = tmp_path / "result.json"
    p.write_text(json.dumps(asdict(r), default=str), encoding="utf-8")

    loaded = json.loads(p.read_text(encoding="utf-8"))
    eff = loaded["aggregated"]["efficiency"]
    assert "latency_ms" in eff and "p50" in eff["latency_ms"]
    assert eff["tokens_in"]["total"] > 0
