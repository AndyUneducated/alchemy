"""Contract layer: unique data shape across layers.

Five top-level contract dataclasses form a data flow:
    Doc -> Request -> Response -> SampleResult -> EvalResult

Starting from phase 6, a nested field type `Usage` (live in `Response.usage`, and OpenAI / Anthropic /
inspect_ai SDK isomorphic), does not belong to the top-level contract - it is an embedded resource consumption type of Response.

All other layers (Task/LM/Metric/Runner/Storage) only read/produce these types and do not import each other.
Choose dataclass instead of Pydantic: Phase 1 does not introduce dependencies; frozen provides immutable + hash + asdict.
When switching to Pydantic v2, the external API remains unchanged and only the validator is added."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RequestType = Literal["generate_until", "loglikelihood", "multiple_choice"]
EvalMode = Literal["score", "run"]


@dataclass(frozen=True)
class Doc:
    """One row of data set, Task output.

    `id` is used for de-dup and per-sample tracking/join predictions.
    `target` is relaxed from str to `str | None` (introduced in Phase 4): taking into account "old tasks still pass str"
    With "rag_retrieval / any task without string gold explicitly pass None" on both sides - avoid using ""
    Placeholders pollute semantics. `metadata` is a free-form bucket for task/pipeline interoperability: RAG is in
    Inject the retrieved products (retrieved_ids / contexts) here in the `process_docs` hook.
    `Response` remains loaded with only LM-side output (path B+C decision, see DECISIONS §4 for details)."""

    id: str
    input: str
    target: str | None = None
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    """LM call request.

    There are only three request_types deliberately set, which are consistent with the original version of lm-evaluation-harness.
    Do not introduce chat messages, let the LM adaptation layer decide how to encapsulate it, and ensure that the prompt literal is reproducible."""

    doc_id: str
    prompt: str
    request_type: RequestType = "generate_until"
    until: tuple[str, ...] = ()
    max_tokens: int = 64
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Usage:
    """Resource consumption of LM calls (introduced in phase 6).

    Identical to OpenAI `CompletionUsage` / Anthropic `Usage` / inspect_ai `ModelUsage`:
    nested typed object to avoid the top-level `Response` field in multi-model ecological expansion (reasoning_tokens/
    cached_tokens/audio_tokens).

    Extension points (view model ecology can be added on demand, adding fields will not break the old Response):
      - reasoning_tokens o1 / DeepSeek-R1 style
      - cached_tokens Anthropic prompt caching / OpenAI cached input
      - audio_tokens multimodal

    By design, the score path / MockLM is never filled in (keep None); OllamaLM and other true adapters are
    `generate_until` parses the provider response and fills it in."""

    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True)
class Response:
    """Return of LM.

    `text` and `loglikelihoods` are mutually exclusive, and request_type determines which one has a value.
    `latency_ms` top-level time dimension (same position as HELM `request_time` / inspect_ai `output.time`);
    Reserved from phase 0, OllamaLM is filled in from phase 6, and the runner does not do the fallback of dividing the batch time by N——
    Explicit None is preferable to imprecise estimation.
    `usage` nested resource consumption (tokens_in/out, etc.), introduced in phase 6; MockLM / score path is always None.
    In score mode, the text field is read from predictions JSONL."""

    doc_id: str
    text: str | None = None
    loglikelihoods: tuple[float, ...] | None = None
    latency_ms: float | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class SampleResult:
    """Single sample scoring results, granularity = 1 sample.

    `metrics` form (from phase 7, nested faction unified, supersede phase 6 audit §1.5):
      - **task-specific scalar**: always flat top level (`acc` / `f1_macro` / `cohens_kappa`) - task internal namespace
      - **cross-cutting cross-cutting subgroup**: nested dict (`metrics["efficiency"]` / `metrics["safety"]`) - cross-cutting namespace injected by runner; exactly the same as `aggregated["<dim>"]` nested subgroup / `Response.usage` nested object three layers (OpenAI / Anthropic / inspect_ai faction)
      - **`_` prefix private key**: still at the top level of task (such as `_safety_category`), not on the aggregation panel, used for aggregation consumption

    For F1/kappa, which requires the complete set to be calculated, leave aggregation to pull the original pred/target and calculate it yourself.

    `artifacts` (introduced in Phase 4) holds per-sample **non-scalar** artifacts:
      - `pred_ids` / `gold_ids` of retrieval task (aggregation is pulled with ranx)
      - Trajectory steps / tool_calls for future agent tasks
      - any diagnostic dump other than metric values

    Formed with `dict[str, float | None | dict[str, float | None]]` of metrics
    MLflow / W&B style scalar/non-scalar duality - prevent sneaking in `list[str]`
    Breaking type contract in `metrics`.

    Anti-trash can discipline:
      - Install "per-sample non-scalar product", aggregation input + diagnostic dump purpose
      - Not allowed to install: status irrelevant to metric calculation (log/reports are placed in the metric closure; task status goes to __init__)

    Type relaxation evolution (DECISIONS §7.D onwards nested; §X wave 4 plus None placeholder):
      phase 1: dict[str, float] - strictly stick to scalars
      phase 7: dict[str, float | dict[str, float]] - crosscut subgroup nesting
      wave 4: dict[str, float | None | dict[str, float | None]] - None means "not measured"
              (judge parse fails / safety slices empty / etc., the same shape as phase 7 wave 2 P2)"""

    doc_id: str
    prediction: str
    target: str
    metrics: dict[str, float | None | dict[str, float | None]]
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """The final product of a run, granularity = entire run.

    Outer package inner layer: `per_sample: list[SampleResult]` provides drill-down entry.
    `aggregated` installs indicators that must be viewed in full to be calculated (f1_macro / kappa / NDCG...).
    `mode` distinguishes score / run, allowing storage to be filtered by mode.
    `num_fewshot` is only meaningful in the run path (score is always 0, `= 0` defaults to the score path construction to save fields).

    The aggregated type is relaxed to `dict[str, Any]` starting from phase 6 (actual form `dict[str, float | dict]`):
      - Top-level tiling task's own indicators (HELM accuracy dimension: accuracy / f1_macro / em / rouge_l / ...)
      - Nested subassembly cross-cutting dimensions (an additional 6 dimensions of HELM 7 dimensions), mounted bipartite by cross-cutting ontology (DECISIONS §7.A):
          aggregated["efficiency"] phase 6 call class only run hangs
          aggregated["safety"] phase 7 content class score / run double hanging
          aggregated["calibration"] phase 9 call class (plan)
          aggregated["robustness"] phase 10 content class (plan)
      - Indicators with the same name have consistent positions across phases (for example, cohens_kappa is at the top level of phases 1 / 8), ensuring
        The cross-run JSON_EXTRACT path does not drift; explicit task-specific metrics are not internally categorized by "method families"."""

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
