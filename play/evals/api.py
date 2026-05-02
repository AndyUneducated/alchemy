"""契约层：跨层唯一数据形状.

5 个 frozen dataclass 组成一条数据流：
    Doc -> Request -> Response -> SampleResult -> EvalResult

所有其他层（Task / LM / Metric / Runner / Storage）都只读/生产这些类型，互相不 import。
选 dataclass 而非 Pydantic：Phase 1 不引依赖；frozen 提供不可变 + hash + asdict。
换 Pydantic v2 时外部 API 不变，只需加 validator。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RequestType = Literal["generate_until", "loglikelihood", "multiple_choice"]
EvalMode = Literal["score", "run"]


@dataclass(frozen=True)
class Doc:
    """数据集一行，Task 产出。

    `id` 用于 de-dup 和 per-sample 追踪 / join predictions。
    Phase 1 `target` 用 str，未来可扩为 Any（MCQ 用 int index，RAG 用 list[str]）。
    """

    id: str
    input: str
    target: str
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    """LM 的调用请求.

    刻意只设三种 request_type，和 lm-evaluation-harness 原版一致。
    不引入 chat messages，让 LM 适配层自己决定怎么封装，保证 prompt 字面可复现。
    """

    doc_id: str
    prompt: str
    request_type: RequestType = "generate_until"
    until: tuple[str, ...] = ()
    max_tokens: int = 64
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Response:
    """LM 的返回.

    `text` 和 `loglikelihoods` 互斥，由 request_type 决定哪个有值。
    `latency_ms` 即使 Phase 1 不用也预留，为 Phase 10 efficiency 维度埋点。
    offline 模式下 text 字段从 predictions JSONL 读进来。
    """

    doc_id: str
    text: str | None = None
    loglikelihoods: tuple[float, ...] | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class SampleResult:
    """单样本评分结果，粒度 = 1 条样本.

    `metrics` 只装 per-sample 可算完的数值（acc=0/1, EM, 单条 BLEU）.
    需要全集统计的（F1、kappa）留给 aggregation，原始 pred/target 用 `_` 前缀私有键传下去。
    """

    doc_id: str
    prediction: str
    target: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvalResult:
    """一次 run 的最终产物，粒度 = 整个 run.

    外层包内层：`per_sample: list[SampleResult]` 提供 drill-down 入口。
    `aggregated` 装必须看全集才能算的指标（f1_macro / kappa / NDCG...）。
    `mode` 区分 score / run，让 storage 能按模式过滤。
    """

    task: str
    model: str
    mode: EvalMode
    n: int
    aggregated: dict[str, float]
    per_sample: tuple[SampleResult, ...]
    run_id: str
    created_at: str  # ISO8601
    elapsed_ms: float
