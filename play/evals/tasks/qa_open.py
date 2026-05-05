"""Phase 3 vertical slice：族 3（LLM-as-judge）开放式中文 QA task.

10 条事实型 QA + 4 份 stub predictions（perfect / paraphrase / wrong_fact / garbage），
设计宗旨是"在 lexical 失效或误判时让 judge 救场或抓错"——pointwise 在 task 层
有强故事点（plan §六）：

  | 预测       | exact_match | rouge_l | judge_pointwise | 故事 |
  |---|---|---|---|---|
  | perfect    | 1.0         | ~1.0    | ~5              | 上界 sanity |
  | paraphrase | 0.0         | ~0.4    | ~4              | lexical 低 / judge 高（**核心叙事**） |
  | wrong_fact | 0.0         | ~0.9    | ~1-2            | lexical 高 / judge 低（**反向叙事**） |
  | garbage    | 0.0         | ~0.1    | ~1              | 下界 sanity |

设计：判 judge 调用发生在 process_results（per-sample），aggregation 仅 mean——
这样 score / run 两路径都自动获得 judge 评分能力，符合 lm-eval 的"process_results 不区分来源"原则.

构造：QAOpen(judge_lm=None) → 仅 lexical baseline（用于无网络 / parity test 对照支）.
       QAOpen(judge_lm=lm)  → 加 judge_pointwise key.
"""

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
from .mt import _rouge_scorer  # 复用 mt 的中文 char-level rouge tokenizer

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
# 中英 mixed template 是有意为之：FakeJudgeLM 的 Jaccard 规则按 "Reference answer: " /
# "Response: " 字面切割 prompt（与 metrics/judge_core.DEFAULT_POINTWISE_TEMPLATE 同 anchor），
# 真 LLM judge 看中文部分即可正常打分。两路径 anchor 都齐.

DATA_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "qa_open" / "gold.jsonl"


@register_task("qa_open")
class QAOpen(Task):
    """开放式中文 QA。judge_lm 可选——None 时退回 lexical baseline."""

    name: ClassVar[str] = "qa_open"
    output_type: ClassVar[str] = "generate_until"

    def __init__(
        self,
        judge_lm: LM | None = None,
        *,
        judge_template: str = QA_OPEN_JUDGE_TEMPLATE,
        judge_n_samples: int = 1,
    ) -> None:
        """`judge_n_samples > 1` 时自动套 self_consistency 多采样取众数 wrapper."""
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
        metrics: dict[str, float] = {"em": float(pred == target)}
        if self._judge_pointwise_fn is not None:
            metrics["judge_pointwise"] = float(self._judge_pointwise_fn(doc, response))
        return SampleResult(doc_id=doc.id, prediction=pred, target=target, metrics=metrics)

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        agg: dict[str, Callable[[list[SampleResult]], float]] = {
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
        """DECISIONS §7.3：从 judge closure 的 _recorder 拉 LM 调用记录."""
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


def _judge_pointwise_mean(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    vals = [s.metrics["judge_pointwise"] for s in srs if "judge_pointwise" in s.metrics]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
