"""契约层：跨层唯一数据形状.

5 个顶层契约 dataclass 组成一条数据流：
    Doc -> Request -> Response -> SampleResult -> EvalResult

phase 6 起增加 1 个嵌套字段类型 `Usage`（住 `Response.usage`，与 OpenAI / Anthropic /
inspect_ai SDK 同形），不属于顶层契约——它是 Response 的内嵌资源消耗类型。

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
    `target` 由 str 放宽为 `str | None`（Phase 4 引入）：兼顾"老 task 仍传 str"
    与"rag_retrieval / 任何无字符串 gold 的 task 显式传 None"两侧——避免用 ""
    占位污染语义。`metadata` 是 task / pipeline 互通的 free-form bucket：RAG 在
    `process_docs` hook 里把检索产物（retrieved_ids / contexts）注入这里，
    `Response` 保持只装 LM-side 输出（path B+C 决策，详见 DECISIONS §4）。
    """

    id: str
    input: str
    target: str | None = None
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
class Usage:
    """LM 调用的资源消耗（phase 6 引入）.

    与 OpenAI `CompletionUsage` / Anthropic `Usage` / inspect_ai `ModelUsage` 同形：
    nested typed object，避免顶层 `Response` 字段在多模型生态扩展（reasoning_tokens /
    cached_tokens / audio_tokens）时膨胀。

    扩展点（视模型生态按需加，加字段不破老 Response）：
      - reasoning_tokens   o1 / DeepSeek-R1 风格
      - cached_tokens      Anthropic prompt caching / OpenAI cached input
      - audio_tokens       多模态

    设计上 score 路径 / MockLM 永远不填（保持 None）；OllamaLM 等真适配器在
    `generate_until` 内解析 provider response 后填入。
    """

    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True)
class Response:
    """LM 的返回.

    `text` 和 `loglikelihoods` 互斥，由 request_type 决定哪个有值。
    `latency_ms` 顶层时间维度（与 HELM `request_time` / inspect_ai `output.time` 同位）；
    phase 0 起预留，phase 6 起 OllamaLM 真填，runner 不做 batch 时间除以 N 的 fallback——
    显式 None 优于不准估算。
    `usage` 嵌套资源消耗（tokens_in/out 等），phase 6 引入；MockLM / score 路径永远 None。
    score 模式下 text 字段从 predictions JSONL 读进来。
    """

    doc_id: str
    text: str | None = None
    loglikelihoods: tuple[float, ...] | None = None
    latency_ms: float | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class SampleResult:
    """单样本评分结果，粒度 = 1 条样本.

    `metrics` 形态（phase 7 起 nested 派统一，supersede phase 6 audit §1.5）：
      - **task-specific scalar**：永远 flat 顶层（`acc` / `f1_macro` / `cohens_kappa`）—— task 内部命名空间
      - **cross-cutting 横切子组**：嵌套 dict（`metrics["efficiency"]` / `metrics["safety"]`）—— runner 注入的横切 namespace；与 `aggregated["<dim>"]` 嵌套子组 / `Response.usage` nested object 三层完全一致（OpenAI / Anthropic / inspect_ai 派）
      - **`_` 前缀私有键**：仍 task 顶层（如 `_safety_category`），不上聚合面板，aggregation 消费用

    F1/kappa 这种需要全集才能算的留 aggregation 拉原始 pred/target 自己算。

    `artifacts`（Phase 4 引入）装 per-sample **非标量**产物：
      - retrieval task 的 `pred_ids` / `gold_ids`（aggregation 用 ranx 拉）
      - 未来 agent task 的 trajectory steps / tool_calls
      - 任何 diagnostic dump 而非 metric 数值

    与 metrics 的 `dict[str, float | dict[str, float]]` 形成 MLflow / W&B 风格的
    scalar/non-scalar 对偶——防止把 `list[str]` 偷偷塞进 `metrics` 里破坏类型契约。

    防垃圾桶纪律：
      - 装"per-sample 非标量产物"，aggregation 输入 + diagnostic dump 用途
      - 不许装：与 metric 计算无关的状态（log/上报放 metric 闭包内；task 状态走 __init__）

    类型放宽演化（DECISIONS §7.D，supersede §6.1 §1.5）：
      phase 1: dict[str, float]                     —— 严守标量
      phase 7: dict[str, float | dict[str, float]]  —— 横切子组嵌套
    """

    doc_id: str
    prediction: str
    target: str
    metrics: dict[str, float | dict[str, float]]
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """一次 run 的最终产物，粒度 = 整个 run.

    外层包内层：`per_sample: list[SampleResult]` 提供 drill-down 入口。
    `aggregated` 装必须看全集才能算的指标（f1_macro / kappa / NDCG...）。
    `mode` 区分 score / run，让 storage 能按模式过滤。
    `num_fewshot` 仅 run 路径有意义（score 永远 0）；默认值保证旧 result.json 反序列化兼容。

    aggregated 类型 phase 6 起放宽为 `dict[str, Any]`（实际形态 `dict[str, float | dict]`）：
      - 顶层平铺任务自身指标（HELM accuracy 维度：accuracy / f1_macro / em / rouge_l / ...）
      - 嵌套子组装横切维度（HELM 7 维度的另外 6 维），按 cross-cutting ontology 二分挂载（DECISIONS §7.A）：
          aggregated["efficiency"]   phase 6   call class    仅 run 挂
          aggregated["safety"]       phase 7   content class score / run 双挂
          aggregated["calibration"]  phase 9   call class    （计划）
          aggregated["robustness"]   phase 10  content class （计划）
      - 同名指标跨 phase 位置一致（如 cohens_kappa 在 phase 1 / 8 都顶层），保证
        cross-run JSON_EXTRACT 路径不漂移；显式不按"方法学族"内部归类 task-specific 指标。
    """

    task: str
    model: str
    mode: EvalMode
    n: int
    aggregated: dict[str, Any]
    per_sample: tuple[SampleResult, ...]
    run_id: str
    created_at: str  # ISO8601
    elapsed_ms: float
    num_fewshot: int = 0
