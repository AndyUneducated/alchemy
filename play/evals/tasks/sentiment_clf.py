"""Phase 1 vertical slice：族 1（Classification + Agreement）MVP.

三分类 sentiment 任务：positive / negative / neutral.

展示的指标分叉（教学故事）：
  - accuracy vs F1-macro vs F1-micro vs cohens_kappa 在不同预测分布下的分叉
  - macro 暴跌 / accuracy 还行 → "全押一个类"的退化（constant_neutral）
  - kappa ≈ 0 / accuracy > 0 → "靠运气"的部分，kappa 剔除掉了

aggregation 里三个 callable 从 SampleResult 提取 y_true/y_pred 后直接调
sklearn.metrics——Phase 1 没有专门的 metric 抽象层，理由见 CHANGELOG ADR #2。
"""

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

# 数据路径相对于项目根（play/evals/）
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment" / "gold.jsonl"


def _normalize(text: str | None) -> str:
    """模型输出归一化到 LABELS 之一.

    真实 LLM 常带空格、Markdown、解释文字；phase 1 简单策略：
      1. 去空白、小写、剥离 "Label:" 前缀
      2. 取第一个 token、去尾部标点
      3. 若匹配 LABELS 返回之；否则用关键词 fallback（"pos"→positive, "neg"→negative, 其它→neutral）

    Phase 1 目标不是鲁棒性 demo 而是 metric 教学，所以 fallback 足够简单即可。
    """
    if text is None:
        return "neutral"
    s = text.strip().lower()
    if s.startswith("label:"):
        s = s[len("label:") :].strip()
    first = s.split()[0] if s.split() else ""
    first = first.rstrip(".,;:!?'\"")
    if first in LABELS:
        return first
    # fallback：LLM 可能输出 "pos" / "negative." / "it's positive"
    if first.startswith("pos"):
        return "positive"
    if first.startswith("neg"):
        return "negative"
    return "neutral"


@register_task("sentiment_clf")
class SentimentClf(Task):
    """三分类情感任务."""

    name: ClassVar[str] = "sentiment_clf"
    output_type: ClassVar[str] = "generate_until"

    # 允许测试/Runner 覆盖数据源（score 模式下 Runner 不会用，但保留接口一致）
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
        # SampleResult.prediction / .target 是顶层字段（强类型），
        # aggregation 直接读它们就是"两模式共享 Task 契约"的体现。
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
