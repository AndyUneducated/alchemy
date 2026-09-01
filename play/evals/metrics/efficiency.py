"""Phase 6 efficiency crosscutting dimension metric module.

Trigger new as per README guideline #3: cross-cutting dimensions span all tasks, math/price list helper required for runner injection.

Design points:
  - **runner automatic collection** (cross-cutting AOP style): task does not change process_results / aggregation;
    runner._evaluate_inner hangs in run mode `aggregated["efficiency"] = efficiency_aggregated(srs)`
    (Starting from phase 7, _finalize is merged into _evaluate_inner, and cross-cutting injectors + aggregation + packaging are unified in the middle).
  - **Nested subgroup**: `aggregated["efficiency"]` subtree is phase 7+ crosscut (safety/calibration/
    robustness) reserved extension bits (HELM 7 dimensions for ontology); and OpenAI / Anthropic / inspect_ai
    SDK's nested usage object style alignment.
  - **schema-on-write two-layer consistency** (phase 6 audit follow-up; phase 7 §7.D from nested faction unification):
    `aggregated["efficiency"]` subgroup always hangs 4 subgroups, `SampleResult.metrics["efficiency"]` subgroup always
    Write 4 efficiency keys; missing value 0.0 placeholder (semantic "not measured"), let downstream drill-down (CLI/dashboard/
    SQL) write a schema without branch judgment KeyError. The CLI rendering layer judges "all 0" and folds it into a line of `<not measured>`.
    Avoid visual misleading.
  - **per 1M tokens unit** (same unit as OpenAI / Anthropic / Together / Fireworks public offer,
    entry directly copy and paste without brain deletion 1000).
  - **fail-loud unknown model** (phase 6 audit follow-up): `compute_cost_usd` is not in model
    `_PRICE_PER_1M_TOKENS` when `warnings.warn` (lru_cache anti-swipe); distinguish between "true free / untested /
    There are three cost=0 states not in the table.
  - **stdlib calculates percentile**: `statistics.quantiles(data, n=100, method='inclusive')`,
    Do not import numpy (project phase 1-5 existing code 0 explicitly imports numpy).

Data contract (per-sample, phase 7 §7.D nested faction):
  inject_per_sample_efficiency Use dataclasses.replace to write the following 4 keys
  SampleResult.metrics["efficiency"] Nested subgroups (always 4 keys, None / 0.0 placeholder for missing values):
    - latency_ms from Response.latency_ms
    - tokens_in from Response.usage.tokens_in
    - tokens_out from Response.usage.tokens_out
    - cost_usd is derived from compute_cost_usd(model_label, tokens_in, tokens_out)

  Access path: `s.metrics["efficiency"]["latency_ms"]` (the first version of phase 6 is flat `s.metrics["latency_ms"]`,
  phase 7 §7.D supersede is a nested subgroup, consistent with aggregated["efficiency"] / Response.usage three layers).

Industry Benchmarking:
  - HELM efficiency dimensions: mean / p50 / p95 / max are standard (latency_ms subgroup 4 stat of this module)
  - inspect_ai ModelUsage: input_tokens / output_tokens tiling (this module uses tokens_in / tokens_out subgroups)
  - tokencost / litellm: cost lookup table full model; this module stub 4 entry, phase 3+ enabled external
    Consider cutting tokencost when providing provider"""

from __future__ import annotations

import functools
import statistics
import warnings

from ..api import Response, SampleResult

# CLI rendering layer folding protocol (phase 7 audit P1, trait faction):
# True = all 0 subgroups are collapsed into a single line of `<dim>: <not measured>` to avoid visual misleading
# False = all 0 is a legal metric value (content class) and does not fold
# efficiency is call class - all 0 is almost equivalent to mock / output_type='none' path,
# Folding is honest UX; safety and other content classes go False (heuristic really runs out of 0 which is a legal value).
# CLI _print_aggregated queries this constant via evals.cli._should_fold_when_all_zero.
FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO = True


# ---------- Price list (per 1M tokens) ----------

# tuple = (input_price_per_1M, output_price_per_1M) USD
# Industry practice input != output (output is autoregressive decode, 4-5x input price; open source platform 1:1)
# Data as of 2026-05; manual synchronization or consider cutting tokencost when price changes
# (https://github.com/AgentOps-AI/tokencost)
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # ollama Local Reasoning: Using Together AI/Fireworks public quotes as a "how much does it cost to run on the cloud" analogy
    # Keep conftest.py history DEFAULT_TEST_MODEL + current qwen3.x default pair (plan A: env-driven default
    # Switch to qwen3.5:9b / qwen3.6:27b, the old qwen2.5:32b remains compatible with agent_sft v1 history result.json);
    # Other local tags miss when passing EVALS_TEST_OLLAMA_MODEL override → take 0 branch (no damage; add as needed)
    "ollama:qwen3.6:27b": (0.80, 0.80),
    "ollama:qwen3.5:9b": (0.80, 0.80),
    "ollama:qwen2.5:32b": (0.80, 0.80),
    # Each external provider leaves one SKU for debugging (the cheapest SKU; phase 3 NotImplementedError cannot be run yet.
    # But the entry is not damaged and can be used when phase 3+ is enabled; cli.py::EXTERNAL_PROVIDERS covers all three companies)
    "openai:gpt-4o-mini": (0.15, 0.60),
    "anthropic:claude-3-5-haiku-20241022": (1.00, 5.00),
    "gemini:gemini-1.5-flash": (0.075, 0.30),
    # mock:* not prefilled - always 0 by design; compute_cost_usd miss → 0.0
}


@functools.lru_cache(maxsize=128)
def _warn_unknown_pricing_model(model: str) -> None:
    """Warn only once for each missed model in the same process (lru_cache prevents screen refresh)."""
    warnings.warn(
        f"unknown pricing model {model!r} (not in _PRICE_PER_1M_TOKENS); cost reported as 0.0. "
        f"Add an entry to _PRICE_PER_1M_TOKENS to enable cost tracking.",
        UserWarning,
        stacklevel=3,
    )


def compute_cost_usd(
    model: str,
    tokens_in: int | None,
    tokens_out: int | None,
) -> float | None:
    """Derive cost_usd from PRICE_TABLE.

    Return value convention:
      - tokens_in / tokens_out either None → None (unmeasured, keep None untainted)
      - model is not in table → 0.0 + UserWarning (fail-loud): Let users distinguish "real free vs unconfigured pricing";
        warning uses lru_cache to prevent screen refresh. Each unknown model is only warned once in the same process.
      - hit → (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    Note (phase 7 audit P3): The score path is divided into ontology bisections (DECISIONS §7.A call class
    Only run (only run) does not hang the efficiency subgroup → this function is not adjusted, so `preds:*` and other score paths
    model_label will never enter the price list query and will not trigger the unknown-model warning. This is correct behavior
    (preds:* is the file label, not LM) instead of fail-silent."""
    if tokens_in is None or tokens_out is None:
        return None
    if model not in _PRICE_PER_1M_TOKENS:
        _warn_unknown_pricing_model(model)
    in_per_m, out_per_m = _PRICE_PER_1M_TOKENS.get(model, (0.0, 0.0))
    return (tokens_in * in_per_m + tokens_out * out_per_m) / 1_000_000.0


# ---------- Aggregation helpers (stdlib only) ----------

def _percentile(data: list[float], pct: float) -> float:
    """linear interpolation percentile (consistent with numpy's default 'linear' method).

    `statistics.quantiles(data, n=100, method='inclusive')[i-1]` when i ∈ [1,99]
    Equivalent to numpy.percentile(data, i), but requires len(data) >= 2; this helper covers the details
    Single element/empty list scenario ensures that efficiency_aggregated does not explode in small batches."""
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct!r}")
    # method='inclusive' gives 99 cutpoints at n=100; index = round(pct) - 1
    # But exact linear interp requires (n-1) * pct/100 quadratic interpolation; statistics is implemented.
    quantiles = statistics.quantiles(sorted(data), n=100, method="inclusive")
    idx = max(0, min(98, int(round(pct)) - 1))
    return float(quantiles[idx])


def _collect(srs: list[SampleResult], key: str) -> list[float]:
    """Collect non-None values from SampleResult.metrics["efficiency"] nested subgroups (phase 7 §7.D nested pie).

    The first version of phase 6 is flat `s.metrics.get(key)`; §7.D supersede is nested path
    `s.metrics.get("efficiency", {}).get(key)`, symmetrical with the inject writing path."""
    out: list[float] = []
    for s in srs:
        sub = s.metrics.get("efficiency")
        if not isinstance(sub, dict):
            continue
        v = sub.get(key)
        if v is not None:
            out.append(float(v))
    return out


# ---------- run-only entry (runner._evaluate_inner call) ----------

def efficiency_aggregated(sample_results: list[SampleResult]) -> dict[str, dict[str, float | int]]:
    """Generates `aggregated["efficiency"]` nested subtrees.

    Return fixed 4-subgroup schema (retain schema even if all are missing):

        {
          "latency_ms": {"mean": ..., "p50": ..., "p95": ..., "max": ...},
          "tokens_in": {"total": <int>, "mean": <float>},
          "tokens_out": {"total": <int>, "mean": <float>},
          "cost_usd": {"total": <float>, "mean": <float>},
        }

    Type convention (phase 6 audit follow-up):
      - tokens.total uses `int` (token is a discrete count); mean is still float (avg can have decimals)
      - latency_ms / cost_usd all float

    Coverage (HELM efficiency dimension benchmarking):
      - latency_ms 4 stat: mean/p50/p95/**max** (small N for worst-case exposed entry; HELM/
        inspect_ai all reports max; audit §1.2)
      - cost_usd double stat: total / **mean** (per-call average cost, with tokens.{total,mean} format
        Alignment; audit §1.1)

    Missing value (MockLM reports None / output_type='none' task / score mode is skipped) → subgroup key value 0.0;
    Ensure that the efficiency schema of the run mode is always consistent, and downstream consumption (CLI _fmt_row recursive printing / W&B
    dashboard / cross-run JSON_EXTRACT) does not require branch judgment None.

    This function is not called in score mode (runner._evaluate_inner is only in the update subtree of the mode='run' branch)."""
    latency = _collect(sample_results, "latency_ms")
    tokens_in = _collect(sample_results, "tokens_in")
    tokens_out = _collect(sample_results, "tokens_out")
    cost = _collect(sample_results, "cost_usd")

    return {
        "latency_ms": {
            "mean": float(statistics.mean(latency)) if latency else 0.0,
            "p50": _percentile(latency, 50.0),
            "p95": _percentile(latency, 95.0),
            "max": float(max(latency)) if latency else 0.0,
        },
        "tokens_in": {
            "total": int(sum(tokens_in)),
            "mean": float(statistics.mean(tokens_in)) if tokens_in else 0.0,
        },
        "tokens_out": {
            "total": int(sum(tokens_out)),
            "mean": float(statistics.mean(tokens_out)) if tokens_out else 0.0,
        },
        "cost_usd": {
            "total": float(sum(cost)),
            "mean": float(statistics.mean(cost)) if cost else 0.0,
        },
    }


# ---------- judge subgroup aggregation (DECISIONS §7.3 wave 3: evaluation tool call class, both paths hang) -

def efficiency_judge_aggregated(
    judge_responses: list[Response],
    judge_model_label: str | None,
) -> dict[str, dict[str, float | int]]:
    """Generates `aggregated["efficiency"]["judge"]` nested subtrees.

    Same shape as `efficiency_aggregated` 4 subgroups (latency_ms / tokens_in / tokens_out / cost_usd),
    But the sources are different:
      - efficiency_aggregated collected from `sample.metrics["efficiency"]` nested subgroup (task LM)
      - efficiency_judge_aggregated directly collects list[Response] from `task.collect_judge_responses()`
        (Evaluation tool judge LM call record)

    Why doesn't the judge attach the sample layer? Because the judge call and sample have an N:M relationship - a sample may trigger
    Multiple judge calls (such as RAG faithfulness: claim extract + per-claim NLI = 1 + N times; g_eval
    Multi-dimensional multi-sampling = D × n_samples times), unlike the object under test, task LM and sample have a 1:1 relationship, which is suitable for sharing
    sample.metrics. So judge efficiency is only exposed in the aggregated layer.

    DECISIONS §7.3 Evaluation tool call class (divided into the call class of the object under test): both score / run paths are hung.

    Missing value handling:
      - judge_responses is empty / model_label is None → 4 subgroups are all 0 placeholders (with efficiency_aggregated
        The schema-on-write protocol is consistent; the CLI folding protocol can be folded accordingly to `<not measured>`)
      - Single response missing latency_ms / usage → skip (consistent with _collect None-skipping)"""
    if not judge_responses or judge_model_label is None:
        return {
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "tokens_in": {"total": 0, "mean": 0.0},
            "tokens_out": {"total": 0, "mean": 0.0},
            "cost_usd": {"total": 0.0, "mean": 0.0},
        }

    latency = [r.latency_ms for r in judge_responses if r.latency_ms is not None]
    tokens_in = [
        r.usage.tokens_in
        for r in judge_responses
        if r.usage is not None and r.usage.tokens_in is not None
    ]
    tokens_out = [
        r.usage.tokens_out
        for r in judge_responses
        if r.usage is not None and r.usage.tokens_out is not None
    ]
    cost: list[float] = []
    for r in judge_responses:
        if r.usage is None:
            continue
        c = compute_cost_usd(judge_model_label, r.usage.tokens_in, r.usage.tokens_out)
        if c is not None:
            cost.append(c)

    return {
        "latency_ms": {
            "mean": float(statistics.mean(latency)) if latency else 0.0,
            "p50": _percentile(latency, 50.0),
            "p95": _percentile(latency, 95.0),
            "max": float(max(latency)) if latency else 0.0,
        },
        "tokens_in": {
            "total": int(sum(tokens_in)),
            "mean": float(statistics.mean(tokens_in)) if tokens_in else 0.0,
        },
        "tokens_out": {
            "total": int(sum(tokens_out)),
            "mean": float(statistics.mean(tokens_out)) if tokens_out else 0.0,
        },
        "cost_usd": {
            "total": float(sum(cost)),
            "mean": float(statistics.mean(cost)) if cost else 0.0,
        },
    }


# ---------- runner injector (avoid runner.py import dataclasses.replace directly) ----------

def inject_per_sample_efficiency(
    sample_results: list[SampleResult],
    responses: list[Response],
    model_label: str,
) -> list[SampleResult]:
    """Called in the middle of the run path _evaluate_inner, write per-sample efficiency into the SampleResult.metrics["efficiency"] subgroup.

    nested dispatch position (phase 7 §7.D supersede phase 6 audit §1.5):
      `metrics={..., "efficiency": {"latency_ms": ..., "tokens_in": ..., "tokens_out": ..., "cost_usd": ...}}`
    Exactly the same as `aggregated["efficiency"]` nested subgroup / `Response.usage` nested object three levels
    (OpenAI/Anthropic/inspect_ai parties aligned).

    schema-on-write (audit §1.3 option A + §7.D nested unified): always write `metrics["efficiency"]` subgroup
    Contains 4 efficiency keys, None / 0.0 placeholder for missing values (semantic "unmeasured"), with aggregated.efficiency subgroup always
    The schema philosophy of the 4 subgroups is consistent; downstream drill-down `s.metrics["efficiency"]["latency_ms"]` does not require branches
    throw KeyError.

    CLI rendering layer `_print_aggregated` collapses all 0 efficiency subgroups into `<not measured>` single lines to avoid visual
    Misleading (audit §1.7; recursive form common to phase 7+ crosscutting subgroups).

    Use dataclasses.replace to maintain SampleResult frozen semantics.
    Sequence convention: sample_results[i] ↔ responses[i] (same order as runner._build_request).

    Remove the getattr defense of the first version of phase 6 (audit §1.6): Response is a frozen dataclass field fixed,
    `resp.latency_ms` is taken directly; when schema is renamed, AttributeError will be exposed immediately instead of silent None."""
    from dataclasses import replace as _replace

    if len(sample_results) != len(responses):
        raise RuntimeError(
            f"length mismatch: sample_results={len(sample_results)} vs responses={len(responses)}"
        )

    out: list[SampleResult] = []
    for sr, resp in zip(sample_results, responses):
        usage = resp.usage
        tokens_in = usage.tokens_in if usage is not None else None
        tokens_out = usage.tokens_out if usage is not None else None
        cost = compute_cost_usd(model_label, tokens_in, tokens_out)

        eff_subgroup: dict[str, float] = {
            "latency_ms": float(resp.latency_ms) if resp.latency_ms is not None else 0.0,
            "tokens_in": float(tokens_in) if tokens_in is not None else 0.0,
            "tokens_out": float(tokens_out) if tokens_out is not None else 0.0,
            "cost_usd": float(cost) if cost is not None else 0.0,
        }
        out.append(_replace(sr, metrics={**sr.metrics, "efficiency": eff_subgroup}))
    return out
