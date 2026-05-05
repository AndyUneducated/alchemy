"""Phase 6 efficiency 横切维度 metric 模块.

按 README 指导原则 #3 触发新建：横切维度跨所有 task，runner 注入需要数学/价格表 helper.

设计要点：
  - **runner 自动采集**（cross-cutting AOP 风格）：task 不改 process_results / aggregation；
    runner._evaluate_inner 在 run 模式时挂 `aggregated["efficiency"] = efficiency_aggregated(srs)`
    （phase 7 起 _finalize 合并入 _evaluate_inner，cross-cutting injectors + 聚合 + 打包统一在中段）.
  - **嵌套子组**：`aggregated["efficiency"]` 子树为 phase 7+ 横切（safety / calibration /
    robustness）预留扩展位（HELM 7 维度作 ontology）；与 OpenAI / Anthropic / inspect_ai
    SDK 的 nested usage object 风格对齐.
  - **schema-on-write 两层一致**（phase 6 audit follow-up；phase 7 §7.D 起 nested 派统一）：
    `aggregated["efficiency"]` 子组永远挂 4 子组、`SampleResult.metrics["efficiency"]` 子组永远
    写 4 efficiency 键；缺失值 0.0 占位（语义"未测得"），让下游 drill-down（CLI / dashboard /
    SQL）写一份 schema 不需分支判 KeyError. CLI 渲染层判断"全 0"折叠为 `<not measured>` 一行，
    避免视觉误导.
  - **per 1M tokens 单位**（与 OpenAI / Anthropic / Together / Fireworks 公开报价同单位，
    entry 直接复制粘贴免人脑除 1000）.
  - **fail-loud unknown model**（phase 6 audit follow-up）：`compute_cost_usd` 在 model 不在
    `_PRICE_PER_1M_TOKENS` 时 `warnings.warn`（lru_cache 防刷屏）；区分"真免费 / 未测得 /
    不在表里"三种 cost=0 状态.
  - **stdlib 算 percentile**：`statistics.quantiles(data, n=100, method='inclusive')`，
    不引 numpy（项目 phase 1-5 现有代码 0 处显式 import numpy）.

数据契约（per-sample，phase 7 §7.D nested 派）：
  inject_per_sample_efficiency 用 dataclasses.replace 把以下 4 键写进
  SampleResult.metrics["efficiency"] 嵌套子组（永远 4 键，None / 缺失值 0.0 占位）：
    - latency_ms     来自 Response.latency_ms
    - tokens_in      来自 Response.usage.tokens_in
    - tokens_out     来自 Response.usage.tokens_out
    - cost_usd       由 compute_cost_usd(model_label, tokens_in, tokens_out) 派生

  访问路径：`s.metrics["efficiency"]["latency_ms"]`（phase 6 初版是 flat `s.metrics["latency_ms"]`，
  phase 7 §7.D supersede 为 nested 子组，与 aggregated["efficiency"] / Response.usage 三层一致）.

行业对标：
  - HELM efficiency 维度：mean / p50 / p95 / max 是标配（本模块 latency_ms 子组 4 stat）
  - inspect_ai ModelUsage：input_tokens / output_tokens 平铺（本模块用 tokens_in / tokens_out 子组）
  - tokencost / litellm：cost lookup table 全模型；本模块 stub 4 entry，phase 3+ 启用 external
    provider 时考虑切 tokencost
"""

from __future__ import annotations

import functools
import statistics
import warnings

from ..api import Response, SampleResult

# CLI 渲染层折叠协议（phase 7 audit P1，trait 派）：
#   True  = 全 0 子组折叠为 `<dim>: <not measured>` 单行，避免视觉误导
#   False = 全 0 是合法 metric 值（content class），不折叠
# efficiency 是 call class——全 0 几乎等价 mock / output_type='none' 路径，
# 折叠是诚实 UX；safety 等 content class 走 False（heuristic 真跑出 0 是合法值）。
# CLI _print_aggregated 通过 evals.cli._should_fold_when_all_zero 查询本常量。
FOLD_AS_NOT_MEASURED_WHEN_ALL_ZERO = True


# ---------- 价格表（per 1M tokens） ----------

# tuple = (input_price_per_1M, output_price_per_1M) USD
# 行业惯例 input != output（output 是 autoregressive decode，4-5x input 价；开源平台 1:1）
# 数据 as of 2026-05；价变时手动同步或考虑切 tokencost
# (https://github.com/AgentOps-AI/tokencost)
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    # ollama 本地推理：用 Together AI / Fireworks 公开报价做 "如果在 cloud 跑会花多少" 类比
    # 仅保留 conftest.py DEFAULT_TEST_MODEL；其它 qwen2.5 tag 通过 EVALS_TEST_OLLAMA_MODEL
    # override 时未命中 → 走 0 分支（无伤；用户按需自加）
    "ollama:qwen2.5:32b": (0.80, 0.80),
    # 外部 provider 各留一个调试用 SKU（最便宜 SKU；phase 3 NotImplementedError 暂跑不到，
    # 但 entry 在不破坏，phase 3+ 启用时即用；cli.py::EXTERNAL_PROVIDERS 三家全覆盖）
    "openai:gpt-4o-mini": (0.15, 0.60),
    "anthropic:claude-3-5-haiku-20241022": (1.00, 5.00),
    "gemini:gemini-1.5-flash": (0.075, 0.30),
    # mock:* 不预填——设计上永远 0；compute_cost_usd 未命中 → 0.0
}


@functools.lru_cache(maxsize=128)
def _warn_unknown_pricing_model(model: str) -> None:
    """对每个未命中 model 同进程内只 warn 一次（lru_cache 防刷屏）."""
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
    """根据 PRICE_TABLE 派生 cost_usd.

    返回值约定：
      - tokens_in / tokens_out 任一 None → None（未测得，保持 None 不污染）
      - model 不在 table → 0.0 + UserWarning（fail-loud）：让用户区分"真免费 vs 未配置定价"；
        warning 用 lru_cache 防刷屏，每个 unknown model 同进程内只 warn 一次
      - 命中 → (tokens_in * in_price + tokens_out * out_price) / 1_000_000

    注意（phase 7 audit P3）：score 路径在 ontology 二分（DECISIONS §7.A call class
    仅 run 挂）下不挂 efficiency 子组 → 不调本函数，故 `preds:*` 等 score 路径
    model_label 永不进入价格表查询，不会触发 unknown-model warning。这是正确行为
    （preds:* 是文件 label 不是 LM）而非 fail-silent.
    """
    if tokens_in is None or tokens_out is None:
        return None
    if model not in _PRICE_PER_1M_TOKENS:
        _warn_unknown_pricing_model(model)
    in_per_m, out_per_m = _PRICE_PER_1M_TOKENS.get(model, (0.0, 0.0))
    return (tokens_in * in_per_m + tokens_out * out_per_m) / 1_000_000.0


# ---------- 聚合 helpers（stdlib only） ----------

def _percentile(data: list[float], pct: float) -> float:
    """linear interpolation percentile（与 numpy 默认 'linear' method 一致）.

    `statistics.quantiles(data, n=100, method='inclusive')[i-1]` 在 i ∈ [1,99] 时
    与 numpy.percentile(data, i) 等价，但要求 len(data) >= 2；本 helper 兜底
    单元素 / 空列表场景，保证 efficiency_aggregated 在小 batch 下不爆.
    """
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct!r}")
    # method='inclusive' 在 n=100 时给 99 cutpoints；index = round(pct) - 1
    # 但精确 linear interp 需要 (n-1) * pct/100 二次插值；statistics 已实现.
    quantiles = statistics.quantiles(sorted(data), n=100, method="inclusive")
    idx = max(0, min(98, int(round(pct)) - 1))
    return float(quantiles[idx])


def _collect(srs: list[SampleResult], key: str) -> list[float]:
    """从 SampleResult.metrics["efficiency"] 嵌套子组收集非 None 值（phase 7 §7.D nested 派）.

    phase 6 初版是 flat `s.metrics.get(key)`；§7.D supersede 为 nested 路径
    `s.metrics.get("efficiency", {}).get(key)`，与 inject 写入路径对称.
    """
    out: list[float] = []
    for s in srs:
        sub = s.metrics.get("efficiency")
        if not isinstance(sub, dict):
            continue
        v = sub.get(key)
        if v is not None:
            out.append(float(v))
    return out


# ---------- run-only 入口（runner._evaluate_inner 调用） ----------

def efficiency_aggregated(sample_results: list[SampleResult]) -> dict[str, dict[str, float | int]]:
    """生成 `aggregated["efficiency"]` 嵌套子树.

    返回固定 4 子组形态（即使全缺失也保留 schema）：

        {
          "latency_ms": {"mean": ..., "p50": ..., "p95": ..., "max": ...},
          "tokens_in":  {"total": <int>, "mean": <float>},
          "tokens_out": {"total": <int>, "mean": <float>},
          "cost_usd":   {"total": <float>, "mean": <float>},
        }

    类型约定（phase 6 audit follow-up）：
      - tokens.total 用 `int`（token 是离散计数）；mean 仍 float（avg 可有小数）
      - latency_ms / cost_usd 全 float

    覆盖范围（HELM efficiency 维度对标）：
      - latency_ms 4 stat：mean / p50 / p95 / **max**（小 N 下 worst-case 暴露入口；HELM /
        inspect_ai 都报 max；audit §1.2）
      - cost_usd 双 stat：total / **mean**（per-call 平均成本，与 tokens.{total,mean} 体例
        对齐；audit §1.1）

    缺失值（MockLM 报 None / output_type='none' task / score 模式被跳过）→ 子组键值 0.0；
    保证 run 模式的 efficiency schema 始终一致，下游消费（CLI _fmt_row 递归打印 / W&B
    dashboard / cross-run JSON_EXTRACT）不需要分支判 None.

    score 模式不调用本函数（runner._evaluate_inner 只在 mode='run' 分支 update 子树）.
    """
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


# ---------- runner injector（避免 runner.py 直接 import dataclasses.replace） ----------

def inject_per_sample_efficiency(
    sample_results: list[SampleResult],
    responses: list[Response],
    model_label: str,
) -> list[SampleResult]:
    """run 路径 _evaluate_inner 中段调用，把 per-sample efficiency 写进 SampleResult.metrics["efficiency"] 子组.

    nested 派写位置（phase 7 §7.D supersede phase 6 audit §1.5）：
      `metrics={..., "efficiency": {"latency_ms": ..., "tokens_in": ..., "tokens_out": ..., "cost_usd": ...}}`
    与 `aggregated["efficiency"]` 嵌套子组 / `Response.usage` nested object 三层完全一致
    （OpenAI / Anthropic / inspect_ai 派对齐）.

    schema-on-write（audit §1.3 选项 A + §7.D nested 统一）：永远写 `metrics["efficiency"]` 子组
    含 4 efficiency 键，None / 缺失值 0.0 占位（语义"未测得"），与 aggregated.efficiency 子组永远
    4 子组的 schema 哲学一致；下游 drill-down `s.metrics["efficiency"]["latency_ms"]` 不需要分支
    判 KeyError.

    CLI 渲染层 `_print_aggregated` 对全 0 efficiency 子组折叠为 `<not measured>` 单行避免视觉
    误导（audit §1.7；递归形态对 phase 7+ 横切子组通用）.

    用 dataclasses.replace 保持 SampleResult frozen 语义.
    顺序约定：sample_results[i] ↔ responses[i]（与 runner._build_request 同序）.

    去掉 phase 6 初版的 getattr 防御（audit §1.6）：Response 是 frozen dataclass 字段固定，
    `resp.latency_ms` 直接取；schema rename 时 AttributeError 即时暴露而非 silent None.
    """
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
