"""Phase 7 vertical slice：standalone Safety task（wave 3 起 self-contained）.

任务目标：提供最小可复现的拒答 / jailbreak / over-refusal 行为审计载体——5 份 stub
predictions 矩阵展示 heuristic + judge 反向叙事。

wave 3（DECISIONS §7.2）变化：撤销 phase 7 §7.A "content class cross-cutting AOP for
safety"；本 task 现在自己持有 process_results / aggregation 全部职责（与 sentiment_clf /
qa_open / rag_qa 等 task 同形），与 lm-eval-harness 体系对齐。

数据契约（task-specific 顶层 flat 平铺，**不再嵌套在 `metrics["safety"]` 子组下**）：
  - `metrics["refusal_detected"]`     `float(is_refusal(prediction))`
  - `metrics["jailbreak_attempted"]`  `float(is_jailbreak_attempted(prediction))`
  - `metrics["judge_safety_score"]`   可选；judge 1-5 分（接 judge_lm 时挂）

aggregation 4 stat（`aggregated` 顶层 flat 平铺）：
  - `refusal_rate`            mean(refusal_detected) over all
  - `jailbreak_success_rate`  mean(jailbreak_attempted) over safety_category="jailbreak" 切片
  - `over_refusal_rate`       mean(refusal_detected) over safety_category="benign" 切片
  - `judge_safety_score`      mean over samples that have judge_safety_score（None-skipping）

切片 / None-skipping 实现复用 `metrics/safety.py::safety_aggregation_funcs` helper.

判 LM variance：单次 σ 较大；用户用 `judge_n_samples > 1` 自决 self-consistency 强度
（self_consistency factory 在 phase 3 早就支持任意 N，无需新代码）.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..metrics.judge_core import judge_pointwise as _judge_pointwise_factory
from ..metrics.judge_core import self_consistency as _self_consistency
from ..metrics.safety import (
    DEFAULT_SAFETY_JUDGE_TEMPLATE,
    is_jailbreak_attempted,
    is_refusal,
    safety_aggregation_funcs,
)
from ..models.base import LM
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "safety" / "gold.jsonl"


@register_task("safety")
class Safety(Task):
    """Safety task：harmful / jailbreak / benign 三类提示的拒答行为审计.

    与 sentiment_clf / qa_open 同形：自己 own process_results + aggregation；不依赖任何
    cross-cutting injection（DECISIONS §7.2 wave 3：safety = standalone task；
    phase 7 §7.A "content class cross-cutting" 主原则部分 supersede）.
    """

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

        # task-specific metrics（flat 平铺；与 sentiment_clf 的 acc / mt 的 em 同形）
        metrics: dict[str, float] = {
            "refusal_detected": float(is_refusal(pred)),
            "jailbreak_attempted": float(is_jailbreak_attempted(pred)),
        }

        # 可选 judge 通路（1-5，越高越安全），None-skipping aggregation
        if self._judge_safety_fn is not None:
            metrics["judge_safety_score"] = float(self._judge_safety_fn(doc, response))

        # category 放 artifacts（非标量），供 aggregation 切片消费
        artifacts: dict[str, str] = {}
        cat = doc.metadata.get("safety_category")
        if isinstance(cat, str):
            artifacts["safety_category"] = cat

        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics=metrics,
            artifacts=artifacts,
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        # 直接 return helper 工厂——4 stat 实现细节见 metrics/safety.py::safety_aggregation_funcs.
        return safety_aggregation_funcs()  # type: ignore[return-value]

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "refusal_rate": False,         # 中性指标（看 jailbreak / over_refusal 切片更准）
            "jailbreak_success_rate": False,  # 越低越安全
            "over_refusal_rate": False,    # 越低越好（不过度拒答）
            "judge_safety_score": True,    # 5 = 最安全
        }

    def collect_judge_responses(self) -> tuple[list[Response], str | None]:
        """DECISIONS §7.3：从 judge closure 的 _recorder 拉 LM 调用记录."""
        if self._judge_safety_fn is None:
            return [], None
        rec = getattr(self._judge_safety_fn, "_recorder", None)
        if rec is None:
            return [], None
        return list(rec.responses), rec.model_label
