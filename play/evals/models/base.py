"""LM adapter ABC.

Three requests to align lm-evaluation-harness:
  - generate_until generates freely until stop seq, 90% of tasks use this
  - loglikelihood gives prompt + continuation to calculate logp, MCQ + calibration basics
  - loglikelihood_rolling the entire rolling perplexity, used for Phase 9 calibration

Only run mode is used; score mode bypasses this layer completely.

Why not do chat(messages) API:
  The philosophy of lm-eval is that the task completely owns the literal string of prompt, ensuring that the prompt reported by the paper is
  It is exactly the same as what was actually sent to the model. The chat API will be used by the provider's system prompt / role template
  Implicit rewriting destroys reproducibility. The adaptation layer is responsible for packaging when it wants to chat."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..api import Request, Response


class LM(ABC):
    """All LM backends (mock/OpenAI/Anthropic/Ollama/prerecorded) implement this interface."""

    name: str  # Human-readable model tag, dropped into the EvalResult.model field

    @abstractmethod
    def generate_until(self, requests: list[Request]) -> list[Response]:
        """Freely generated. Phase 1 implements uniform batch in → batch out and keeps the Runner simple."""
        ...

    def loglikelihood(self, requests: list[Request]) -> list[Response]:
        """Calculates logp for (prompt, continuation). Phase 1 is not enabled, Phase 4+ MCQ is on."""
        raise NotImplementedError("loglikelihood not implemented in phase 1")

    def loglikelihood_rolling(self, requests: list[Request]) -> list[Response]:
        """The whole rolling perplexity. Used for Phase 9 calibration."""
        raise NotImplementedError("loglikelihood_rolling not implemented in phase 1")
