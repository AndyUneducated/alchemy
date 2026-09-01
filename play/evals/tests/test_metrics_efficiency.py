"""Phase 6 metrics/efficiency unit lock:

cost lookup table shape (per 1M tokens × (in, out) tuples) + bounds (None / miss / mock)
+ aggregated schema always has 4 subgroups + percentile implementation consistent with numpy linear interp behavior.

Not citing numpy/tokencost: stdlib statistics implements percentile, and several numerical assertions are calculated by hand oracle locked.

phase 6 audit follow-up (2026-05-04):
  - 1.1 cost_usd subgroup plus mean
  - 1.2 latency_ms subgroup plus max
  - 1.4 unknown model fail-loud UserWarning
  - 1.5 tokens.{total} uses int instead of float (integer semantics)"""

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


# ---------- Price list ----------

def test_price_table_entries_are_in_out_tuple():
    """Each entry must be (input_price, output_price) binary float tuple, unit USD/1M tokens."""
    assert _PRICE_PER_1M_TOKENS  # at least 1
    for model, price in _PRICE_PER_1M_TOKENS.items():
        assert isinstance(model, str), f"key 必须是 str，got {type(model)}"
        assert isinstance(price, tuple) and len(price) == 2, (
            f"price 必须是 2-tuple，got {price!r} for {model}"
        )
        in_p, out_p = price
        assert isinstance(in_p, (int, float)) and in_p >= 0
        assert isinstance(out_p, (int, float)) and out_p >= 0


def test_price_table_includes_canonical_models():
    """The 4 debugging SKUs established in phase 6 + plan A. The default pair of qwen3.x after switching is always in the table.
    (cli.py EXTERNAL_PROVIDERS three full coverage + current default ollama tag)."""
    assert "ollama:qwen3.6:27b" in _PRICE_PER_1M_TOKENS
    assert "ollama:qwen3.5:9b" in _PRICE_PER_1M_TOKENS
    assert "openai:gpt-4o-mini" in _PRICE_PER_1M_TOKENS
    assert "anthropic:claude-3-5-haiku-20241022" in _PRICE_PER_1M_TOKENS
    assert "gemini:gemini-1.5-flash" in _PRICE_PER_1M_TOKENS


# ---------- compute_cost_usd ----------

def test_compute_cost_returns_none_for_missing_tokens():
    """tokens_in or tokens_out either None → cost None (keep non-None collection protocol)."""
    assert compute_cost_usd("ollama:qwen3.6:27b", None, 100) is None
    assert compute_cost_usd("ollama:qwen3.6:27b", 100, None) is None
    assert compute_cost_usd("ollama:qwen3.6:27b", None, None) is None


def test_compute_cost_returns_zero_for_unknown_model():
    """Unfilled model (such as mock:gold / unlisted ollama tag) → cost 0.0; audit §1.4 and above fail-loud warning."""
    _warn_unknown_pricing_model.cache_clear()  # Clear lru_cache to allow warnings to be reissued
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compute_cost_usd("mock:gold", 1000, 500) == 0.0
        assert compute_cost_usd("ollama:not-pulled:7b", 1000, 500) == 0.0
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("mock:gold" in m for m in msgs)
    assert any("ollama:not-pulled:7b" in m for m in msgs)


def test_compute_cost_unknown_model_warning_dedups_lru():
    """If the unknown model is called multiple times in the same process, only warn once (lru_cache prevents screen refresh)."""
    _warn_unknown_pricing_model.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            compute_cost_usd("dedup:test:model", 100, 50)
    msgs = [str(w.message) for w in caught if "dedup:test:model" in str(w.message)]
    assert len(msgs) == 1, f"expected 1 warning for dedup model, got {len(msgs)}: {msgs}"


def test_compute_cost_known_model_no_warning():
    """Hit model should not warn (avoid spraying signals on normal paths)."""
    _warn_unknown_pricing_model.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_cost_usd("ollama:qwen3.6:27b", 100, 50)
    assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"


def test_compute_cost_uses_per_1m_unit():
    """openai:gpt-4o-mini = (0.15, 0.60) per 1M → 1M in + 1M out = 0.15 + 0.60 = 0.75 USD."""
    cost = compute_cost_usd("openai:gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == 0.75


def test_compute_cost_handles_distinct_in_out_prices():
    """anthropic:claude-3-5-haiku = (1.00, 5.00) → 1M in + 1M out = 6.00 USD (output 5x input lock)."""
    cost = compute_cost_usd("anthropic:claude-3-5-haiku-20241022", 1_000_000, 1_000_000)
    assert cost == 6.00


def test_compute_cost_small_call_scales_correctly():
    """ollama:qwen3.6:27b = (0.80, 0.80) per 1M → 100 in + 200 out = (100*0.8 + 200*0.8)/1e6."""
    cost = compute_cost_usd("ollama:qwen3.6:27b", 100, 200)
    expected = (100 * 0.80 + 200 * 0.80) / 1_000_000.0
    assert cost == expected


# ---------- efficiency_aggregated schema always 4 subgroups ----------

def _sr(eff: dict[str, float] | None = None) -> SampleResult:
    """Construct SampleResult for testing.

    phase 7 §7.D onwards sample.metrics nested: efficiency key value embedded metrics["efficiency"] subgroup.
    `eff=None` means that the sample has no efficiency signal at all (mock path or score path)."""
    metrics: dict[str, float | dict[str, float]] = {}
    if eff is not None:
        metrics["efficiency"] = eff
    return SampleResult(doc_id="x", prediction="p", target="t", metrics=metrics)


def test_efficiency_aggregated_empty_inputs_returns_zero_schema():
    """No efficiency signal at all (MockLM path) → 4 subgroup key values are all 0, schema still exists.
    audit §1.1: cost_usd plus mean; §1.2: latency_ms plus max; §1.5: tokens.total uses int."""
    agg = efficiency_aggregated([_sr(None)])
    assert set(agg.keys()) == {"latency_ms", "tokens_in", "tokens_out", "cost_usd"}
    assert agg["latency_ms"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert agg["tokens_in"] == {"total": 0, "mean": 0.0}
    assert agg["tokens_out"] == {"total": 0, "mean": 0.0}
    assert agg["cost_usd"] == {"total": 0.0, "mean": 0.0}


def test_efficiency_aggregated_handles_empty_sample_list():
    """0 sample (boundary) → Same as zero schema, does not cause statistics.mean empty sequence error."""
    agg = efficiency_aggregated([])
    assert agg["latency_ms"]["p50"] == 0.0
    assert agg["latency_ms"]["max"] == 0.0
    assert agg["tokens_in"]["total"] == 0
    assert agg["cost_usd"]["mean"] == 0.0


def test_efficiency_aggregated_aggregates_real_signals():
    """All three samples report latency / tokens / cost → mathematically correct (including the newly added max / cost.mean)."""
    srs = [
        _sr({"latency_ms": 100.0, "tokens_in": 10.0, "tokens_out": 5.0, "cost_usd": 0.01}),
        _sr({"latency_ms": 200.0, "tokens_in": 20.0, "tokens_out": 10.0, "cost_usd": 0.02}),
        _sr({"latency_ms": 300.0, "tokens_in": 30.0, "tokens_out": 15.0, "cost_usd": 0.03}),
    ]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == statistics.mean([100.0, 200.0, 300.0])
    assert agg["latency_ms"]["max"] == 300.0  # audit §1.2
    assert agg["tokens_in"]["total"] == 60  # int semantics (audit §1.5)
    assert isinstance(agg["tokens_in"]["total"], int)
    assert agg["tokens_in"]["mean"] == 20.0
    assert agg["tokens_out"]["total"] == 30
    assert agg["cost_usd"]["total"] == pytest.approx(0.06)
    assert agg["cost_usd"]["mean"] == pytest.approx(0.02)  # audit §1.1


def test_efficiency_aggregated_with_zero_padded_signals():
    """phase 6 audit §1.3 Option A: mock path sample.metrics writes 0 placeholder → aggregated still takes 0 path.
    The mathematical behavior is unchanged (the mean/total/max of the 0 sequence are all 0), and the schema-on-write protocol is consistent at both layers."""
    srs = [_sr({"latency_ms": 0.0, "tokens_in": 0.0, "tokens_out": 0.0, "cost_usd": 0.0}) for _ in range(5)]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 0.0
    assert agg["latency_ms"]["max"] == 0.0
    assert agg["tokens_in"]["total"] == 0
    assert agg["cost_usd"]["mean"] == 0.0


def test_efficiency_aggregated_skips_none_signals_per_sample():
    """Some samples are reported, some are not reported (None) → only the mean / total is calculated for the reported samples (non-None collection protocol).
    Note: After phase 6 audit §1.3 option A, the runner injector no longer generates None (always writes 0 placeholder);
    This test respects the None-skipping behavior of efficiency_aggregated itself (it is still legal when directly constructing metrics)."""
    srs = [
        _sr({"latency_ms": 100.0}),  # Only report latency
        _sr({"tokens_in": 50.0}),    # Only report tokens_in
        _sr(None),                    # Report nothing
    ]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 100.0
    assert agg["latency_ms"]["max"] == 100.0
    assert agg["tokens_in"]["total"] == 50
    assert agg["tokens_in"]["mean"] == 50.0
    assert agg["tokens_out"]["total"] == 0


def test_efficiency_aggregated_single_sample_percentile_safe():
    """Single sample cannot explode the ValueError (bound) of statistics.quantiles which requires n>=2."""
    srs = [_sr({"latency_ms": 42.0})]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 42.0
    assert agg["latency_ms"]["p50"] == 42.0
    assert agg["latency_ms"]["p95"] == 42.0
    assert agg["latency_ms"]["max"] == 42.0


def test_efficiency_aggregated_two_sample_percentile_safe():
    """n=2 is the minimum input accepted by statistics.quantiles; audit tests for coverage gap filling."""
    srs = [_sr({"latency_ms": 10.0}), _sr({"latency_ms": 20.0})]
    agg = efficiency_aggregated(srs)
    assert agg["latency_ms"]["mean"] == 15.0
    assert agg["latency_ms"]["max"] == 20.0
    assert 10.0 <= agg["latency_ms"]["p50"] <= 20.0
    assert 10.0 <= agg["latency_ms"]["p95"] <= 20.0


def test_efficiency_aggregated_p95_stays_below_max():
    """p95 monotonicity: 100 arithmetic latency → p95 < max; max is exactly equal to the sequence maximum."""
    srs = [_sr({"latency_ms": float(i)}) for i in range(1, 101)]  # 1..100
    agg = efficiency_aggregated(srs)
    # p50 ≈ 50.5 (median)
    assert 50.0 <= agg["latency_ms"]["p50"] <= 51.0
    # p95 should fall around 95
    assert 94.0 <= agg["latency_ms"]["p95"] <= 96.0
    # max is strictly equal to the maximum value of the sequence
    assert agg["latency_ms"]["max"] == 100.0
    # Strict p95 < max (validates the independent worst-case signal value of max)
    assert agg["latency_ms"]["p95"] < agg["latency_ms"]["max"]


# ---------- phase 7 §7.D nested Dispatch write position lock ----------

def test_efficiency_aggregated_ignores_legacy_flat_keys():
    """From phase 7 §7.D metrics["efficiency"] must be dict; old phase 6 flat writing method
    (metrics["latency_ms"] = 999) is no longer recognized (aggregator cannot see → all 0).

    This is the expected behavior of supersede: the old result.json can be loaded after deserialization but the efficiency data is "invisible",
    Need to rerun. (No exception is thrown / no crash is required, the aggregator silently skips the non-dict efficiency key)."""
    sr_legacy = SampleResult(
        doc_id="x", prediction="p", target="t",
        metrics={"latency_ms": 999.0},  # Old way of writing flat
    )
    agg = efficiency_aggregated([sr_legacy])
    assert agg["latency_ms"]["mean"] == 0.0


def test_inject_per_sample_efficiency_writes_nested_subgroup():
    """phase 7 §7.D Lock: inject_per_sample_efficiency is written to metrics["efficiency"] nested subgroup,
    Does not pollute the top level (task-specific scalar and cross-cutting are layered by ontology)."""
    from evals.api import Response, Usage
    from evals.metrics.efficiency import inject_per_sample_efficiency

    sr = SampleResult(doc_id="x", prediction="p", target="t", metrics={"acc": 1.0})
    resp = Response(
        doc_id="x", text="positive",
        latency_ms=123.0,
        usage=Usage(tokens_in=10, tokens_out=5),
    )
    [out] = inject_per_sample_efficiency([sr], [resp], "ollama:qwen3.6:27b")

    # task-specific top level is not affected
    assert out.metrics["acc"] == 1.0
    # cross-cutting falls within nested subgroups
    assert isinstance(out.metrics["efficiency"], dict)
    assert out.metrics["efficiency"]["latency_ms"] == 123.0
    assert out.metrics["efficiency"]["tokens_in"] == 10.0
    assert out.metrics["efficiency"]["tokens_out"] == 5.0
    # There are no 4 flat keys at the top level (to prevent falling back to phase 6 behavior)
    assert "latency_ms" not in out.metrics
    assert "tokens_in" not in out.metrics
