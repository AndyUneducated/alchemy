"""MockLM：零 API key、确定性、四种教学 mode.

四种 mode 和 data/sentiment/predictions/*.jsonl 四份预录一一对应；
`test_runner_active.py` 的 parity test 证明两路径聚合数值完全一致。

  - gold       偷看 target → 100% acc            ≡ predictions/perfect.jsonl
  - noisy(p)   p 概率替换成随机 label (seed 固定) ≡ predictions/noisy_0.3.jsonl
  - constant   永远同一 label                    ≡ predictions/constant_neutral.jsonl
  - rule       关键词规则弱基线                   ≡ predictions/keyword_rule.jsonl

每次 generate_until 用 `random.Random(seed)` 重置 RNG，保证多次调用同一 MockLM
实例得到完全一致的输出——README 里的预期数值能复现的根。
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Callable, Literal

from ..api import Doc, Request, Response
from .base import LM

MockMode = Literal["gold", "noisy", "constant", "rule"]


def default_rule_fn(text: str) -> str:
    """关键词弱基线：bad→negative / good→positive / 其它→neutral.

    简单到能被任何面试官读懂，且在伪造数据集上能产生中等强度的预测。
    """
    lower = text.lower()
    if "bad" in lower or "terrible" in lower or "awful" in lower:
        return "negative"
    if "good" in lower or "great" in lower or "love" in lower:
        return "positive"
    return "neutral"


class MockLM(LM):
    """假 LLM，只实现 generate_until（Phase 1 唯一用到的请求类型）."""

    def __init__(
        self,
        mode: MockMode,
        docs: Iterable[Doc],
        *,
        seed: int = 0,
        noise: float = 0.3,
        label: str = "neutral",
        rule_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.mode: MockMode = mode
        self.docs_by_id: dict[str, Doc] = {d.id: d for d in docs}
        self.labels: list[str] = sorted({d.target for d in self.docs_by_id.values()})
        self.seed = seed
        self.noise = noise
        self.label = label
        self.rule_fn = rule_fn or default_rule_fn

        if mode == "noisy":
            self.name = f"mock:noisy:{noise}:seed{seed}"
        elif mode == "constant":
            self.name = f"mock:constant:{label}"
        else:
            self.name = f"mock:{mode}"

    def generate_until(self, requests: list[Request]) -> list[Response]:
        """按 mode 生成响应。RNG 每次 batch 重置，保证同一实例多次调用完全一致."""
        rng = random.Random(self.seed)
        out: list[Response] = []
        for req in requests:
            doc = self.docs_by_id.get(req.doc_id)
            if doc is None:
                raise KeyError(f"MockLM has no doc for id={req.doc_id!r}")

            if self.mode == "gold":
                text = doc.target
            elif self.mode == "noisy":
                if rng.random() < self.noise:
                    text = rng.choice(self.labels)
                else:
                    text = doc.target
            elif self.mode == "constant":
                text = self.label
            elif self.mode == "rule":
                text = self.rule_fn(doc.input)
            else:
                raise ValueError(f"unknown mock mode: {self.mode!r}")

            out.append(Response(doc_id=req.doc_id, text=text))
        return out
