"""Task ABC: lm-evaluation-harness original semantics.

A Task = a reproducible evaluation unit that binds dataset + prompt template + answer analysis + aggregation method.

Responsibility demarcation lines for six abstract methods:
  - docs data source (lazy iterator)
  - doc_to_text constructs prompt (only called in run mode)
  - doc_to_target gold answer (symmetrical to doc_to_text, leaving holes for few-shot scenarios)
  - doc_to_choice MCQ-specific, default None
  - process_results per-sample score (unify Response, shared by score/run)
  - aggregation per-sample → global aggregation function dictionary, delayed evaluation
  - higher_is_better indicator direction (for show UI/multi-run sorting)

Two few-shot default methods (added in Phase 2):
  - fewshot_docs example pool, default = self.docs(), subclasses can refer to held-out split
  - format_fewshot_example The string form of an example, the default is doc_to_text + doc_to_target

Three "aligned lm-eval" hooks introduced in Phase 4 (fully implemented by default, without breaking old tasks):
  - load_prediction(doc, row) score path customization JSONL row → (Doc, Response) translation
  - process_docs(docs) run path docs pre-processing before LM call (RAG retrieve / column rename)
  - output_type = "none" adds literal to tell Runner to skip LM calls (used by rag_retrieval)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, ClassVar, Literal

from ..api import Doc, Response, SampleResult

OutputType = Literal[
    "generate_until",
    "multiple_choice",
    "loglikelihood",
    "none",  # Phase 4: Declare that the task does not require LM call (runner jumps to lm.generate_until)
]


class Task(ABC):
    """Base class for all tasks. Subclasses decorate themselves with @register_task."""

    name: ClassVar[str]
    output_type: ClassVar[OutputType]

    @abstractmethod
    def docs(self) -> Iterable[Doc]:
        """Dataset, allowing streaming."""
        ...

    @abstractmethod
    def doc_to_text(self, doc: Doc) -> str:
        """Construct prompt (for run mode). Literal string, do not be overwritten by the provider's system prompt."""
        ...

    @abstractmethod
    def doc_to_target(self, doc: Doc) -> str:
        """gold answer. Symmetrical with doc_to_text, the Runner itself does not touch the target, only process_results."""
        ...

    def doc_to_choice(self, doc: Doc) -> tuple[str, ...] | None:
        """MCQ-specific, default None."""
        return None

    def fewshot_docs(self) -> Iterable[Doc]:
        """few-shot example pool. The default is self.docs() - the Runner excludes the current query when sampling.

        If the subclass has an independent held-out split (train/dev/test style of HF dataset),
        override This method returns another Iterable[Doc]."""
        return self.docs()

    def format_fewshot_example(self, doc: Doc) -> str:
        """A single example is spelled into a string prefixed by prompt. Default = doc_to_text + ' ' + doc_to_target.

        Consistent with lm-eval's default `target_delimiter=' '`; tasks can override the delimiter.
        /Multi-segment structure/Delete instructions to retain input→output short form."""
        return f"{self.doc_to_text(doc)} {self.doc_to_target(doc)}"

    @abstractmethod
    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        """per-sample rating:
        ① normalize model output (case, trim, truncation)
        ② Compare target
        ③Produce per-sample metrics

        Key constraints: Those that require full set statistics (F1, kappa) **don’t** approximate here,
        Stuff the original pred/target with the private keys of `metrics` (`_pred` / `_target`) and give it to aggregation."""
        ...

    @abstractmethod
    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        """{metric_name: fn(list[SampleResult]) -> float | None} Lazy evaluation.

        Why returns a dictionary instead of a number:
        - The same batch of per-sample can be fed to multiple aggregate functions
        - You can replace an aggregate individually during testing
        - key is the final indicator name, Storage uses it directly

        DECISIONS §X wave 4: Returns `float | None` - None means "not measured" (same as phase 7 P2
        safety.judge_safety_score (isomorphic); downstream CLI rendering `<n/a>`, JSON falls into `null`.
        It is also legal for subclasses to return a pure float (no unmeasured scenarios) - Optional is only relaxed, not enforced."""
        ...

    @abstractmethod
    def higher_is_better(self) -> dict[str, bool]:
        """{metric_name: True means bigger is better}. Used for show UI and multi-run comparison and sorting."""
        ...

    # ---- Phase 4 new hooks --------------------------------------------------

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: translate predictions JSONL line into `(enriched_doc, response)`.

        Default implementation: only take `row['prediction']` as Response.text, doc remains unchanged - same as Phase 1
        The old `_load_predictions` behavior is byte identical.

        When overriding the subclass, inject the pipeline data in the row (such as retrieved_ids / contexts)
        `doc.metadata`, install LM-side data into `Response` - follow path B+C: Response only
        Install the LM-side, and the pipeline product lives on the doc side."""
        from dataclasses import replace as _replace

        # By default, doc is left unchanged; if subclasses need to inject metadata, they should _replace themselves in override.
        _ = _replace  # silence vulture
        return doc, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """Run path: Do pre-processing of docs before calling LM (align with lm-eval hook of the same name).

        Typical usage:
        - RAG task calls retrieve_fn here and injects retrieved_ids/contexts into doc.metadata
        - Any task to do batch tokenize/field mapping/column rename/normalize

        Default implementation: identity transparent transmission - old tasks are not affected.

        ⚠️Pure processing discipline (anti-trash can):
        - Signature constraints: Must be `list[Doc] -> list[Doc]`, **not allowed to have "task execution" semantics**
        - Side effects (logging/metric reporting/status writing) should be placed within the metric closure or process_results
        - Initialization unrelated to doc processing (resource preparation/cache warm-up) should be placed in task __init__"""
        return docs

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """Both run / score paths are adjusted and return (judge_responses, judge_model_label).

        Default ([], None)——Tasks without judge/tasks without judge_lm injected will return empty.
        The task holding the judge closure overrides here, pulling the response list from closure._recorder.

        DECISIONS §7.3 Evaluation tool call class: runner collects judge call records in both paths.
        Hang to the `aggregated["efficiency"]["judge"]` subgroup (with the object under test task LM
        `aggregated["efficiency"].{latency_ms, tokens_in, tokens_out, cost_usd}` homogeneous 4 subgroups).

        Implementation guidelines: closure factory (judge_pointwise / g_eval / self_consistency / judge_rag.* 5
        factory) are exposed `closure._recorder.responses + .model_label`; task closes all judges
        The responses can be merged and unified model_label can be obtained."""
        return [], None
