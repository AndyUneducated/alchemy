"""Phase 7 vertical slice：横切 Safety task.

任务目标：提供一个最小可复现实验载体，演示 refusal / jailbreak / over-refusal 的
5 份 stub predictions 矩阵。cross-cutting 主指标由 runner 注入：

  - sample.metrics["safety"] 子组（inject_per_sample_safety）
  - aggregated["safety"] 子组（safety_aggregated）

本 task 只做两件事：
  1) 读 `data/safety/gold.jsonl` 产 Doc（metadata 含 safety_category 切片标签）
  2) 可选注入 judge_safety_score（复用 judge_core.judge_pointwise，不重复造 closure）

按 DECISIONS §7.C 去重决策：不在 metrics/safety.py 再写 judge factory；只暴露模板常量，
task 端复用 phase 3 的 judge_pointwise + self_consistency。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.judge_core import judge_pointwise as _judge_pointwise_factory
from ..metrics.judge_core import self_consistency as _self_consistency
from ..metrics.safety import DEFAULT_SAFETY_JUDGE_TEMPLATE
from ..models.base import LM
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "safety" / "gold.jsonl"


@register_task("safety")
class Safety(Task):
    """Safety task：harmful / jailbreak / benign 三类提示的拒答行为审计."""

    name: ClassVar[str] = "safety"
    output_type: ClassVar[str] = "generate_until"

    def __init__(
        self,
        judge_lm: LM | None = None,
        *,
        judge_template: str = DEFAULT_SAFETY_JUDGE_TEMPLATE,
        judge_n_samples: int = 1,
    ) -> None:
        self.data_path = DATA_PATH
        self._judge_lm = judge_lm
        if judge_lm is not None:
            base = _judge_pointwise_factory(
                judge_lm,
                prompt_template=judge_template,
                scale=(1, 5),
            )
            if judge_n_samples > 1:
                base = _self_consistency(base, n_samples=judge_n_samples)
            self._judge_safety_fn: Callable[[Doc, Response], float] | None = base
        else:
            self._judge_safety_fn = None

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
                    metadata={"safety_category": row["category"]},
                )

    def doc_to_text(self, doc: Doc) -> str:
        # safety task 不额外包 system prompt：让 LM 原生安全策略直接暴露。
        return doc.input

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = (response.text or "").strip()
        target = doc.target or ""
        metrics: dict[str, float] = {}
        artifacts: dict[str, str] = {}

        # category 放 artifacts（非标量），供 safety_aggregated 按切片消费。
        cat = doc.metadata.get("safety_category")
        if isinstance(cat, str):
            artifacts["safety_category"] = cat

        # 可选 judge 通路（1-5，越高越安全），由 task 端注入，aggregation 端 None-skipping mean。
        if self._judge_safety_fn is not None:
            metrics["judge_safety_score"] = float(self._judge_safety_fn(doc, response))

        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        # Phase 7 主指标由 runner 注入 aggregated["safety"]，task 自身无顶层 metric。
        return {}

    def higher_is_better(self) -> dict[str, bool]:
        return {}
