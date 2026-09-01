"""Phase 1 vertical slice: Family 1 (Classification + Agreement) MVP.

Three-class sentiment task: positive / negative / neutral.

Displayed indicator bifurcation (teaching story):
  - accuracy vs F1-macro vs F1-micro vs cohens_kappa bifurcation under different prediction distributions
  - macro crash / accuracy OK → Degeneration of "all in one class" (constant_neutral)
  - kappa ≈ 0 / accuracy > 0 → The "depending on luck" part, kappa has eliminated it

The three callables in aggregation extract y_true/y_pred from SampleResult and call them directly.
sklearn.metrics - Phase 1 does not have a dedicated metric abstraction layer, see DECISIONS §1 (Phase 0 architecture) for the reason."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, ClassVar

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from ..api import Doc, Response, SampleResult
from ..registry import register_task
from .base import Task

LABELS = ("positive", "negative", "neutral")

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following text as one of: "
    "positive, negative, neutral.\n"
    "Text: {input}\n"
    "Label:"
)

# Data paths are relative to the project root (play/evals/)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "gold.jsonl"


def _normalize(text: str | None) -> str:
    """Model output is normalized to one of LABELS.

    Real LLM often comes with spaces, Markdown, and explanatory text; phase 1 simple strategy:
      1. Remove whitespace, lowercase, and strip "Label:" prefix
      2. Take the first token and remove the trailing punctuation
      3. If it matches LABELS, return it; otherwise, use the keyword fallback ("pos"→positive, "neg"→negative, other→neutral)

    The goal of Phase 1 is not robust demo but metric teaching, so the fallback is simple enough."""
    if text is None:
        return "neutral"
    s = text.strip().lower()
    if s.startswith("label:"):
        s = s[len("label:") :].strip()
    first = s.split()[0] if s.split() else ""
    first = first.rstrip(".,;:!?'\"")
    if first in LABELS:
        return first
    # fallback: LLM may output "pos" / "negative." / "it's positive"
    if first.startswith("pos"):
        return "positive"
    if first.startswith("neg"):
        return "negative"
    return "neutral"


@register_task("sentiment_clf")
class SentimentClf(Task):
    """Three classification emotion tasks."""

    name: ClassVar[str] = "sentiment_clf"
    output_type: ClassVar[str] = "generate_until"

    # Allow test/Runner to cover the data source (Runner will not be used in score mode, but the interface will be kept consistent)
    data_path: Path = DATA_PATH

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(id=row["id"], input=row["input"], target=row["target"])

    def doc_to_text(self, doc: Doc) -> str:
        return PROMPT_TEMPLATE.format(input=doc.input)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = _normalize(response.text)
        target = doc.target
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics={"acc": float(pred == target)},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        # SampleResult.prediction / .target are top-level fields (strongly typed),
        # aggregation Reading them directly is the embodiment of "two-mode shared Task contract".
        def _accuracy(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            y_t = [s.target for s in srs]
            y_p = [s.prediction for s in srs]
            return float(accuracy_score(y_t, y_p))

        def _f1_macro(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            y_t = [s.target for s in srs]
            y_p = [s.prediction for s in srs]
            _, _, f, _ = precision_recall_fscore_support(
                y_t, y_p, average="macro", labels=list(LABELS), zero_division=0
            )
            return float(f)

        def _cohens_kappa(srs: list[SampleResult]) -> float:
            if not srs:
                return 0.0
            y_t = [s.target for s in srs]
            y_p = [s.prediction for s in srs]
            return float(cohen_kappa_score(y_t, y_p, labels=list(LABELS)))

        return {
            "accuracy": _accuracy,
            "f1_macro": _f1_macro,
            "cohens_kappa": _cohens_kappa,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {"accuracy": True, "f1_macro": True, "cohens_kappa": True}
