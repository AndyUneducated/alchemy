"""LM 适配器 ABC.

对齐 lm-evaluation-harness 的三种请求：
  - generate_until          自由生成直到 stop seq，90% task 用这个
  - loglikelihood           给 prompt + continuation 算 logp，MCQ + calibration 基础
  - loglikelihood_rolling   整段 rolling perplexity，Phase 7 calibration 用

只有 run 模式用到；score 模式完全绕过这一层。

为什么不做 chat(messages) API：
  lm-eval 的哲学是 task 完全拥有 prompt 的字面字符串，保证 paper 报的 prompt
  和实际发给模型的一字不差。chat API 会被 provider 的 system prompt / role 模板
  隐式改写，破坏可复现性。适配层想要 chat 时自己负责包装。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..api import Request, Response


class LM(ABC):
    """所有 LM 后端（mock / OpenAI / Anthropic / Ollama / prerecorded）实现这个接口."""

    name: str  # 人类可读的 model 标签，落 EvalResult.model 字段

    @abstractmethod
    def generate_until(self, requests: list[Request]) -> list[Response]:
        """自由生成。Phase 1 实现一律 batch in → batch out，保持 Runner 简单."""
        ...

    def loglikelihood(self, requests: list[Request]) -> list[Response]:
        """给 (prompt, continuation) 算 logp. Phase 1 未启用，Phase 4+ MCQ 打开."""
        raise NotImplementedError("loglikelihood not implemented in phase 1")

    def loglikelihood_rolling(self, requests: list[Request]) -> list[Response]:
        """整段 rolling perplexity. Phase 7 calibration 用."""
        raise NotImplementedError("loglikelihood_rolling not implemented in phase 1")
