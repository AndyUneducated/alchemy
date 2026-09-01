"""Phase 4 vertical slice: Family 4 RAG end-to-end QA task.

8 Chinese QA + 4 stubs for `play/rag/docs/panel/` corporate governance narrative corpus
predictions (perfect/paraphrase/wrong_fact/garbage). Core of teaching narrative:
"Looking at the generation quality ladder from the grounding dimension"——

  | prediction | em | rouge_l | faithfulness | answer_correctness | story |
  |---|---|---|---|---|---|
  | perfect | 1.0 | ~1.0 | ~1.0 | ~1.0 | upper bound sanity |
  | paraphrase | 0.0 | mid | ~1.0 | ~1.0 | lexical failure / judge rescue (**core narrative**) |
  | wrong_fact | 0.0 | high | low | low | lexical misjudgment / judge grasps the wrong fact (**reverse narrative**) |
  | garbage | 0.0 | low | low | low | lower bound sanity |

Design points:
  - **process_docs injection contexts** (run path): retrieve all docs at once before LM call,
    contexts/retrieved_ids pin into doc.metadata; `doc_to_text` is a pure string construct (0 IO).
  - **load_prediction injects contexts** (score path): extract contexts/retrieved_ids from row
    Go into doc.metadata, prediction into Response.text - score instance of path B+C.
  - **judge_lm optional**: None → lexical only (em/rouge_l), same mode as qa_open's lexical fallback.
    Hang 5 RAG dimensions when giving judge_lm (faithfulness / answer_correctness / context_precision /
    context_recall / answer_relevancy)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.judge_rag import (
    judge_answer_correctness,
    judge_answer_relevancy,
    judge_context_precision,
    judge_context_recall,
    judge_faithfulness,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task
from .mt import _rouge_scorer  # Chinese char-level rouge tokenizer that reuses mt

PROMPT_TEMPLATE = (
    "请依据以下材料回答问题。\n"
    "材料：\n{context}\n\n"
    "问题：{input}\n"
    "回答："
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_qa" / "gold.jsonl"

RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


@register_task("rag_qa")
class RagQA(Task):
    """RAG end-to-end QA: retrieval + generation + grounding evaluation three-in-one.

    Construction:
      - `retrieve_fn=None` → only the score path is available (contexts are read from predictions)
      - `retrieve_fn=callable` → run path process_docs hook automatically retrieve
      - `judge_lm=None` → only lexical baseline (em/rouge_l)
      - `judge_lm=lm` → add 5 RAG dimensions
      - `top_k` → process_docs truncate contexts/ids to top K items"""

    name: ClassVar[str] = "rag_qa"
    output_type: ClassVar[str] = "generate_until"

    def __init__(
        self,
        retrieve_fn: RetrieveFn | None = None,
        judge_lm: LM | None = None,
        *,
        top_k: int = 5,
    ) -> None:
        self.data_path = DATA_PATH
        self._retrieve_fn = retrieve_fn
        self._judge_lm = judge_lm
        self._top_k = top_k

        if judge_lm is not None:
            self._judge_faithfulness = judge_faithfulness(judge_lm)
            self._judge_answer_correctness = judge_answer_correctness(judge_lm)
            self._judge_context_precision = judge_context_precision(judge_lm)
            self._judge_context_recall = judge_context_recall(judge_lm)
            self._judge_answer_relevancy = judge_answer_relevancy(judge_lm)
        else:
            self._judge_faithfulness = None
            self._judge_answer_correctness = None
            self._judge_context_precision = None
            self._judge_context_recall = None
            self._judge_answer_relevancy = None

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
                    target=row["target"],
                    metadata={"gold_doc_ids": tuple(row.get("gold_doc_ids", ()))},
                )

    def doc_to_text(self, doc: Doc) -> str:
        """Pure string construction: rendering prompt from doc.metadata['contexts'], 0 IO.

        `process_docs` has been retrieved once before the LM call; read the injected contexts here.
        If contexts are missing (rarely, there is no retrieve_fn configuration in run mode), fallback to no material prompt."""
        contexts = doc.metadata.get("contexts", ())
        if contexts:
            ctx_block = "\n---\n".join(contexts)
        else:
            ctx_block = "（无可用材料）"
        return PROMPT_TEMPLATE.format(context=ctx_block, input=doc.input)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_docs(self, docs: list[Doc]) -> list[Doc]:
        """run path: retrieve is completed once before LM call, contexts/ids enter doc.metadata."""
        if self._retrieve_fn is None:
            return docs
        out: list[Doc] = []
        for d in docs:
            ids, contents = self._retrieve_fn(d.input)
            out.append(replace(d, metadata={
                **d.metadata,
                "retrieved_ids": tuple(ids[: self._top_k]),
                "contexts": tuple(contents[: self._top_k]),
            }))
        return out

    def load_prediction(self, doc: Doc, row: dict) -> tuple[Doc, Response]:
        """score path: row['contexts'] / ['retrieved_ids'] → doc.metadata; row['prediction'] → Response.text.

        Score example of path B+C: The pipeline product lives on the doc side, and the LM output lives on the Response side."""
        enriched = replace(doc, metadata={
            **doc.metadata,
            "retrieved_ids": tuple(row.get("retrieved_ids", ())),
            "contexts": tuple(row.get("contexts", ())),
        })
        return enriched, Response(doc_id=doc.id, text=row.get("prediction"))

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = (response.text or "").strip()
        target = doc.target or ""
        metrics: dict[str, float | None] = {
            "em": float(pred == target),
            "rouge_l": _per_sample_rouge_l(pred, target),
        }
        artifacts: dict[str, object] = {
            "pred_ids": list(doc.metadata.get("retrieved_ids", ())),
            "gold_ids": list(doc.metadata.get("gold_doc_ids", ())),
        }
        if self._judge_faithfulness is not None:
            # DECISIONS §X wave 4: judge_answer_correctness / judge_answer_relevancy in parse
            # Returns None on failure; the remaining 3 closures still only return float (it is legal to return 0.0 for the degenerate-input path)
            # lowest score); unified use of None-check is both compatible and robust to the None path of future closure upgrades.
            for key, fn in (
                ("faithfulness", self._judge_faithfulness),
                ("answer_correctness", self._judge_answer_correctness),
                ("context_precision", self._judge_context_precision),
                ("context_recall", self._judge_context_recall),
                ("answer_relevancy", self._judge_answer_relevancy),
            ):
                v = fn(doc, response)
                if v is not None:
                    metrics[key] = float(v)
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        agg: dict[str, Callable[[list[SampleResult]], float | None]] = {
            "exact_match": _mean_metric("em"),
            "rouge_l": _mean_metric("rouge_l"),
        }
        if self._judge_lm is not None:
            agg["faithfulness"] = _mean_metric("faithfulness")
            agg["answer_correctness"] = _mean_metric("answer_correctness")
            agg["context_precision"] = _mean_metric("context_precision")
            agg["context_recall"] = _mean_metric("context_recall")
            agg["answer_relevancy"] = _mean_metric("answer_relevancy")
        return agg

    def higher_is_better(self) -> dict[str, bool]:
        out = {"exact_match": True, "rouge_l": True}
        if self._judge_lm is not None:
            out.update({
                "faithfulness": True,
                "answer_correctness": True,
                "context_precision": True,
                "context_recall": True,
                "answer_relevancy": True,
            })
        return out

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3: Aggregate _recorder.responses of 5 RAG judge closures.

        All 5 dimensions share the same judge_lm (the same LM instance is passed to 5 factories during construction), so
        Model_label can be any one (actually the model_labels of the 5 recorders are exactly the same)."""
        if self._judge_lm is None:
            return [], None
        all_responses: list[Response] = []
        label: str | None = None
        for fn in (
            self._judge_faithfulness,
            self._judge_answer_correctness,
            self._judge_context_precision,
            self._judge_context_recall,
            self._judge_answer_relevancy,
        ):
            if fn is None:
                continue
            rec = getattr(fn, "_recorder", None)
            if rec is None:
                continue
            all_responses.extend(rec.responses)
            label = label or rec.model_label
        return all_responses, label


def _per_sample_rouge_l(pred: str, target: str) -> float:
    """Single-sample ROUGE-L F-measure (Chinese char-level; reuse mt._rouge_scorer cache)."""
    if not pred or not target:
        return 0.0
    scorer = _rouge_scorer()
    return float(scorer.score(target, pred)["rougeL"].fmeasure)


def _mean_metric(key: str) -> Callable[[list[SampleResult]], float | None]:
    """Factory: aggregation closure that averages SampleResult.metrics[key].

    DECISIONS §X wave 4: None occupancy "not measured" - key is missing / value=None are filtered;
    Old metrics such as em / rouge_l are always float (not None), and filtering logic transparent transmission does not affect the value;
    judge dimension (answer_correctness / answer_relevancy) parse does not write the key on failure → returns None."""

    def _agg(srs: list[SampleResult]) -> float | None:
        if not srs:
            return None
        vals = [
            s.metrics[key]
            for s in srs
            if key in s.metrics and s.metrics[key] is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    _agg.__name__ = f"mean_{key}"
    return _agg
