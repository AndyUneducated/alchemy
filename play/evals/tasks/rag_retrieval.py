"""Phase 4 vertical slice: Family 4 RAG retrieval-only task.

8 searches for `play/rag/docs/panel/` corporate governance narrative corpus + 4 stubs
predictions (perfect / good_rerank / weak / garbage), the core narrative is "on IR indicators
See retriever quality ladder":

  | prediction | recall@5 | mrr | ndcg@5 | story |
  |---|---|---|---|---|
  | perfect | 1.0 | 1.0 | 1.0 | upper bound sanity |
  | good_rerank | 1.0 | ~0.5 | medium | recall full / rank inaccurate (rerank rescue scene) |
  | weak | ~0.5 | low | low | weak baseline |
  | garbage | 0.0 | 0.0 | 0.0 | lower bound sanity |

Design points:
  - **output_type='none'** (literal introduced in phase 4): runner automatically jumps to LM call,
    Retrieve step task.process_docs and inject retrieved_ids into doc.metadata.
    Replaces the "fake LM adapter" anti-pattern.
  - **process_docs injection**: pass the run path to retrieve_fn → retrieve all docs at once before calling LM.
    `retrieve_fn` is injected by cli.py during construction, and the task itself does not know whether it is a subprocess or in-process.
  - **load_prediction injection**: score path, translate row['retrieved_ids'] into doc.metadata,
    Response is a placeholder (no LM-side data). This is the pred-side embodiment of path B+C.
  - **artifacts is loaded with non-scalars**: process_results is used to load pred_ids/gold_ids with artifacts to aggregation,
    metrics still only holds scalars (an empty dict here - this task has no per-sample scalar metrics).

Backward compatibility: This task can also work normally in the score path through the default construction of `retrieve_fn=None` (retrieve_fn is not required),
The run path must be injected."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.retrieval import (
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_retrieval" / "gold.jsonl"

# retrieve_fn protocol: query: str -> (doc_ids: list[str], contents: list[str])
RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


@register_task("rag_retrieval")
class RagRetrieval(Task):
    """Independent task in RAG retrieval phase: bearer of 5 ranx IR indicators.

    Construction:
      - `retrieve_fn=None` → only score path is available (read retrieved_ids from predictions)
      - `retrieve_fn=callable` → run path process_docs hook injects retrieved_ids
      - `top_k` → process_docs truncation; score path row has been truncated and will not be processed anymore"""

    name: ClassVar[str] = "rag_retrieval"
    output_type: ClassVar[str] = "none"  # phase 4 literal: runner jumps lm.generate_until

    def __init__(
        self,
        retrieve_fn: RetrieveFn | None = None,
        *,
        top_k: int = 10,
    ) -> None:
        self.data_path = DATA_PATH
        self._retrieve_fn = retrieve_fn
        self._top_k = top_k

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(
                    id=row["id"],
                    input=row["input"],
                    target=None,  # rag_retrieval no string target - semantic honesty after phase 4 widening
                    metadata={"gold_doc_ids": tuple(row["gold_doc_ids"])},
                )

    def doc_to_text(self, doc: Doc) -> str:
        """The runner is not adjusted when output_type='none'; the method is reserved only to satisfy ABC."""
        return ""

    def doc_to_target(self, doc: Doc) -> str:
        """doc_to_target should not be walked by fewshot when target is None - an empty string placeholder is returned."""
        return ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run path: retrieve all docs at once before calling LM; retrieved_ids injects metadata.

        When retrieve_fn is missing (for example, the score path is safe here) → identity transparent transmission,
        The score path relies on load_prediction to take another injection path."""
        if self._retrieve_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            ids, _contents = self._retrieve_fn(d.input)
            out.append(replace(
                d,
                metadata={**d.metadata, "retrieved_ids": tuple(ids[: self._top_k])},
            ))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: row['retrieved_ids'] into doc.metadata; Response placeholder (no LM-side data)."""
        retrieved = tuple(row.get("retrieved_ids", ()))
        enriched = replace(doc, metadata={**doc.metadata, "retrieved_ids": retrieved})
        return enriched, Response(doc_id=doc.id)

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred_ids = list(doc.metadata.get("retrieved_ids", ()))
        gold_ids = list(doc.metadata.get("gold_doc_ids", ()))
        return SampleResult(
            doc_id=doc.id,
            prediction="",  # No string prediction (placeholder)
            target="",       # None string target (placeholder; real gold in artifacts.gold_ids)
            metrics={},      # Strictly adhere to scalar, per-sample scalar-free indicators
            artifacts={"pred_ids": pred_ids, "gold_ids": gold_ids},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        # ranx direct adjustment; pull data from SampleResult.artifacts.{pred_ids, gold_ids}
        return {
            "recall@5": recall_at_k(5),
            "precision@5": precision_at_k(5),
            "mrr": mrr(),
            "ndcg@5": ndcg_at_k(5),
            "map@5": map_at_k(5),
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "recall@5": True,
            "precision@5": True,
            "mrr": True,
            "ndcg@5": True,
            "map@5": True,
        }
