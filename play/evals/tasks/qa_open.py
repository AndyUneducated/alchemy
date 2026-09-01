"""Phase 3 vertical slice: Family 3 (LLM-as-judge) open Chinese QA task.

10 factual QA + 4 stub predictions (perfect / paraphrase / wrong_fact / garbage),
The design purpose is to "let the judge come to the rescue or catch the mistake when lexical fails or misjudges" - pointwise at the task layer
Have strong story points (plan §6):

  | prediction | exact_match | rouge_l | judge_pointwise | story |
  |---|---|---|---|---|
  | perfect | 1.0 | ~1.0 | ~5 | upper bound sanity |
  | paraphrase | 0.0 | ~0.4 | ~4 | lexical low / judge high (**core narrative**) |
  | wrong_fact | 0.0 | ~0.9 | ~1-2 | lexical high / judge low (**reverse narrative**) |
  | garbage | 0.0 | ~0.1 | ~1 | lower bound sanity |

Design: judge call occurs in process_results (per-sample), aggregation only means——
In this way, both score / run paths will automatically obtain the judge scoring ability, which is in line with lm-eval's "process_results does not distinguish between sources" principle.

Construction: QAOpen(judge_lm=None) → lexical baseline only (for no network / parity test control branch).
       QAOpen(judge_lm=lm) → add judge_pointwise key."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.judge_core import (
    judge_pointwise as _judge_pointwise_factory,
    self_consistency as _self_consistency,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task
from .mt import _rouge_scorer  # Chinese char-level rouge tokenizer that reuses mt

PROMPT_TEMPLATE = (
    "用一句话回答下列问题。\n"
    "问题：{input}\n"
    "答案："
)

QA_OPEN_JUDGE_TEMPLATE = (
    "请按 1-5 分对回答的整体质量打分（5=完全正确且贴近参考，1=离题或事实错误）。\n"
    "问题：{input}\n"
    "参考答案：{reference}\n"
    "Reference answer: {reference}\n"
    "回答：{response}\n"
    "Response: {response}\n"
    "Score (1-5):"
)
# The Chinese-English mixed template is intentional: FakeJudgeLM's Jaccard rules follow "Reference answer: " /
# "Response: "Literal cutting prompt (same anchor as metrics/judge_core.DEFAULT_POINTWISE_TEMPLATE),
# A real LLM judge can score normally just by looking at the Chinese part. The anchors of both paths are aligned.

DATA_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "qa_open" / "gold.jsonl"


@register_task("qa_open")
class QAOpen(Task):
    """Open Chinese QA. judge_lm optional - returns to lexical baseline when None."""

    name: ClassVar[str] = "qa_open"
    output_type: ClassVar[str] = "generate_until"

    def __init__(
        self,
        judge_lm: LM | None = None,
        *,
        judge_template: str = QA_OPEN_JUDGE_TEMPLATE,
        judge_n_samples: int = 1,
    ) -> None:
        """When `judge_n_samples > 1`, the self_consistency multi-sampling mode wrapper is automatically applied."""
        self.data_path = DATA_PATH
        self._judge_lm = judge_lm
        if judge_lm is not None:
            base = _judge_pointwise_factory(judge_lm, prompt_template=judge_template)
            if judge_n_samples > 1:
                base = _self_consistency(base, n_samples=judge_n_samples)
            self._judge_pointwise_fn: Callable[[Doc, Response], float] | None = base
        else:
            self._judge_pointwise_fn = None

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
        pred = (response.text or "").strip()
        target = doc.target
        metrics: dict[str, float | None] = {"em": float(pred == target)}
        if self._judge_pointwise_fn is not None:
            v = self._judge_pointwise_fn(doc, response)
            # DECISIONS §X wave 4: parse failed → do not write keys (aggregator natural filtering),
            # Same shape as phase 7 P2 "None measured occupancy".
            if v is not None:
                metrics["judge_pointwise"] = float(v)
        return SampleResult(doc_id=doc.id, prediction=pred, target=target, metrics=metrics)

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | None]]:
        agg: dict[str, Callable[[list[SampleResult]], float | None]] = {
            "exact_match": _exact_match,
            "rouge_l": _rouge_l,
        }
        if self._judge_lm is not None:
            agg["judge_pointwise"] = _judge_pointwise_mean
        return agg

    def higher_is_better(self) -> dict[str, bool]:
        out = {"exact_match": True, "rouge_l": True}
        if self._judge_lm is not None:
            out["judge_pointwise"] = True
        return out

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3: Pull the LM call record from the judge closure's _recorder."""
        if self._judge_pointwise_fn is None:
            return [], None
        rec = getattr(self._judge_pointwise_fn, "_recorder", None)
        if rec is None:
            return [], None
        return list(rec.responses), rec.model_label


def _exact_match(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    return sum(s.metrics["em"] for s in srs) / len(srs)


def _rouge_l(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    scorer = _rouge_scorer()
    scores = [scorer.score(s.target, s.prediction)["rougeL"].fmeasure for s in srs]
    return sum(scores) / len(scores)


def _judge_pointwise_mean(srs: list[SampleResult]) -> float | None:
    """DECISIONS §X wave 4: All sample parse failed (key missing) → None "not measured",
    Same as safety.judge_safety_score / phase 7 P2 method; mean is calculated when non-empty."""
    if not srs:
        return None
    vals = [
        s.metrics["judge_pointwise"]
        for s in srs
        if "judge_pointwise" in s.metrics and s.metrics["judge_pointwise"] is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)
