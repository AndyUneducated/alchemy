"""Phase 6 metrics/efficiency 单元锁：

cost lookup table 形状（per 1M tokens × (in, out) 二元组）+ 边界（None / 未命中 / mock）
+ aggregated schema 永远 4 子组 + percentile 实现与 numpy linear interp 行为一致.

不引 numpy/tokencost：stdlib statistics 实现 percentile，几个数值断言用手算 oracle 锁住.

phase 6 audit follow-up（2026-05-04）：
  - 1.1 cost_usd 子组加 mean
  - 1.2 latency_ms 子组加 max
  - 1.4 unknown model fail-loud UserWarning
  - 1.5 tokens.{total} 用 int 而非 float（整数语义）
"""

from __future__ import annotations

import statistics
import warnings

import pytest

from evals.api import SampleResult
from evals.metrics.efficiency import (
    _PRICE_PER_1M_TOKENS,
    _warn_unknown_pricing_model,
    compute_cost_usd,
    efficiency_aggregated,
)


# ---------- 价格表 ----------

def test_price_table_entries_are_in_out_tuple():
    """每条 entry 必须是 (input_price, output_price) 二元 float tuple，单位 USD/1M tokens."""
    assert _PRICE_PER_1M_TOKENS  # 至少 1 条
    for model, price in _PRICE_PER_1M_TOKENS.items():
        assert isinstance(model, str), f"key 必须是 str，got {type(model)}"
        assert isinstance(price, tuple) and len(price) == 2, (
            f"price 必须是 2-tuple，got {price!r} for {model}"
        )
        in_p, out_p = price
        assert isinstance(in_p, (int, float)) and in_p >= 0
        assert isinstance(out_p, (int, float)) and out_p >= 0


def test_price_table_includes_canonical_models():
    """phase 6 立的 4 个调试 SKU 永远在表里（cli.py EXTERNAL_PROVIDERS 三家全覆盖 + 默认 ollama）."""
    assert "ollama:qwen2.5:32b" in _PRICE_PER_1M_TOKENS
    assert "openai:gpt-4o-mini" in _PRICE_PER_1M_TOKENS
    assert "anthropic:claude-3-5-haiku-20241022" in _PRICE_PER_1M_TOKENS
    assert "gemini:gemini-1.5-flash" in _PRICE_PER_1M_TOKENS


# ---------- compute_cost_usd ----------

def test_compute_cost_returns_none_for_missing_tokens():
    """tokens_in 或 tokens_out 任一 None → cost None（保持非 None 收集协议）."""
    assert compute_cost_usd("ollama:qwen2.5:32b", None, 100) is None
    assert compute_cost_usd("ollama:qwen2.5:32b", 100, None) is None
    assert compute_cost_usd("ollama:qwen2.5:32b", None, None) is None


def test_compute_cost_returns_zero_for_unknown_model():
    """未填的 model（如 mock:gold / 未上架的 ollama tag）→ cost 0.0；audit §1.4 起 fail-loud warning."""
    _warn_unknown_pricing_model.cache_clear()  # 清 lru_cache 让 warning 重发
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compute_cost_usd("mock:gold", 1000, 500) == 0.0
        assert compute_cost_usd("ollama:not-pulled:7b", 1000, 500) == 0.0
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("mock:gold" in m for m in msgs)
    assert any("ollama:not-pulled:7b" in m for m in msgs)


def test_compute_cost_unknown_model_warning_dedups_lru():
    """同 unknown model 多次调用同进程内只 warn 一次（lru_cache 防刷屏）."""
    _warn_unknown_pricing_model.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            compute_cost_usd("dedup:test:model", 100, 50)
    msgs = [str(w.message) for w in caught if "dedup:test:model" in str(w.message)]
    assert len(msgs) == 1, f"expected 1 warning for dedup model, got {len(msgs)}: {msgs}"


def test_compute_cost_known_model_no_warning():
    """命中 model 不应 warn（避免在正常路径喷信号）."""
    _warn_unknown_pricing_model.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_cost_usd("ollama:qwen2.5:32b", 100, 50)
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"


def test_compute_cost_uses_per_1m_unit():
    """openai:gpt-4o-mini = (0.15, 0.60) per 1M → 1M in + 1M out = 0.15 + 0.60 = 0.75 USD."""
    cost = compute_cost_usd("openai:gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == 0.75


def test_compute_cost_handles_distinct_in_out_prices():
    """anthropic:claude-3-5-haiku = (1.00, 5.00) → 1M in + 1M out = 6.00 USD（output 5x input 锁）."""
    cost = compute_cost_usd("anthropic:claude-3-5-haiku-20241022", 1_000_000, 1_000_000)
    assert cost == 6.00


def test_compute_cost_small_call_scales_correctly():
    """ollama:qwen2.5:32b = (0.80, 0.80) per 1M → 100 in + 200 out = (100*0.8 + 200*0.8)/1e6."""
    cost = compute_cost_usd("ollama:qwen2.5:32b", 100, 200)
    expected = (100 * 0.80 + 200 * 0.80) / 1_000_000.0
    assert cost == expected


# ---------- efficiency_aggregated schema 永远 4 子组 ----------

def _sr(metrics: dict[str, float]) -> SampleResult:
    return SampleResult(doc_id="x", prediction="p", target="t", metrics=metrics)


def test_efficiency_aggregated_empty_inputs_returns_zero_schema():
    """完全无 efficiency 信号（MockLM 路径）→ 4 子组键值全 0，schema 仍存在.
    audit §1.1: cost_usd 加 mean；§1.2: latency_ms 加 max；§1.5: tokens.total 用 int.
    """
    agg = efficiency_aggregated([_sr({"accuracy": 1.0})])
    assert set(agg.keys()) == {"latency_ms", "tokens_in", "tokens_out", "cost_usd"}
    assert agg["latency_ms"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert agg["tokens_in"] == {"total": 0, "mean": 0.0}
    assert agg["tokens_out"] == {"total": 0, "mean": 0.0}
    assert agg["cost_usd"] == {"total": 0.0, "mean": 0.0}


def test_efficiency_aggregated_handles_empty_sample_list():
    """0 sample（边界）→ 同 zero schema，不爆 statistics.mean 空序列错."""
    agg = efficiency_aggregated([])
    assert agg["latency_ms"]["p50"] == 0.0
    assert agg["latency_ms"]["max"] == 0.0
    assert agg["tokens_in"]["total"] == 0
    assert agg["cost_usd"]["mean"] == 0.0


def test_efficiency_aggregated_aggregates_real_signals():
    """3 个 sample 都报 latency / tokens / cost → 数学正确（含新加的 max / cost.mean）."""
    srs = [
        _sr({"latency_ms": 100.0, "tokens_in": 10.0, "tokens_out": 5.0, "cost_usd": 0.01}),
        _sr({"latency_ms": 200.0, "tokens_in": 20.0, "tokens_out": 10.0, "cost_usd": 0.02}),
        _sr({"latency_ms": 300.0, "tokens_in": 30.0, "tokens_out": 15.0, "cost_usd": 0.03}),
    ]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == statistics.mean([100.0, 200.0, 300.0])
    assert agg["latency_ms"]["max"] == 300.0  # audit §1.2
    assert agg["tokens_in"]["total"] == 60  # int 语义 (audit §1.5)
    assert isinstance(agg["tokens_in"]["total"], int)
    assert agg["tokens_in"]["mean"] == 20.0
    assert agg["tokens_out"]["total"] == 30
    assert agg["cost_usd"]["total"] == pytest.approx(0.06)
    assert agg["cost_usd"]["mean"] == pytest.approx(0.02)  # audit §1.1


def test_efficiency_aggregated_with_zero_padded_signals():
    """phase 6 audit §1.3 选项 A：mock 路径 sample.metrics 写 0 占位 → aggregated 仍走 0 路径.
    数学行为不变（0 序列的 mean/total/max 都是 0），schema-on-write 协议在两层一致.
    """
    srs = [_sr({"latency_ms": 0.0, "tokens_in": 0.0, "tokens_out": 0.0, "cost_usd": 0.0}) for _ in range(5)]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 0.0
    assert agg["latency_ms"]["max"] == 0.0
    assert agg["tokens_in"]["total"] == 0
    assert agg["cost_usd"]["mean"] == 0.0


def test_efficiency_aggregated_skips_none_signals_per_sample():
    """部分 sample 报、部分不报（None）→ 只对报的求 mean / total（非 None 收集协议）.
    注意：phase 6 audit §1.3 选项 A 后，runner injector 不再产生 None（永远写 0 占位）；
    本测试守住 efficiency_aggregated 自身的 None-skipping 行为（直接构造 metrics 时仍合法）.
    """
    srs = [
        _sr({"latency_ms": 100.0}),  # 只报 latency
        _sr({"tokens_in": 50.0}),    # 只报 tokens_in
        _sr({}),                      # 啥都不报
    ]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 100.0
    assert agg["latency_ms"]["max"] == 100.0
    assert agg["tokens_in"]["total"] == 50
    assert agg["tokens_in"]["mean"] == 50.0
    assert agg["tokens_out"]["total"] == 0


def test_efficiency_aggregated_single_sample_percentile_safe():
    """单 sample 不能爆 statistics.quantiles 要求 n>=2 的 ValueError（边界）."""
    srs = [_sr({"latency_ms": 42.0})]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 42.0
    assert agg["latency_ms"]["p50"] == 42.0
    assert agg["latency_ms"]["p95"] == 42.0
    assert agg["latency_ms"]["max"] == 42.0


def test_efficiency_aggregated_two_sample_percentile_safe():
    """n=2 是 statistics.quantiles 接受的最小输入；audit 测试覆盖缺口补齐."""
    srs = [_sr({"latency_ms": 10.0}), _sr({"latency_ms": 20.0})]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 15.0
    assert agg["latency_ms"]["max"] == 20.0
    assert 10.0 <= agg["latency_ms"]["p50"] <= 20.0
    assert 10.0 <= agg["latency_ms"]["p95"] <= 20.0


def test_efficiency_aggregated_p95_stays_below_max():
    """p95 单调性：100 个等差 latency → p95 < max；max 准确等于序列最大值."""
    srs = [_sr({"latency_ms": float(i)}) for i in range(1, 101)]  # 1..100
    agg = efficiency_aggregated(srs)
    # p50 ≈ 50.5（中位数）
    assert 50.0 <= agg["latency_ms"]["p50"] <= 51.0
    # p95 应落在 95 附近
    assert 94.0 <= agg["latency_ms"]["p95"] <= 96.0
    # max 严格等于序列最大值
    assert agg["latency_ms"]["max"] == 100.0
    # 严格 p95 < max（验证 max 的独立 worst-case 信号价值）
    assert agg["latency_ms"]["p95"] < agg["latency_ms"]["max"]
