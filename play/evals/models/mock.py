"""MockLM: zero API key, deterministic, four teaching modes.

There is a one-to-one correspondence between the four modes and the four pre-recorded data/sentiment/predictions/*.jsonl;
The parity test of `test_runner_run.py` proves that the aggregation values of the two paths are completely consistent.

  - gold peek target → 100% acc ≡ predictions/perfect.jsonl
  - noisy(p) p probability is replaced by random label (seed fixed) ≡ predictions/noisy_0.3.jsonl
  - constant always the same label ≡ predictions/constant_neutral.jsonl
  - rule keyword rule weak baseline ≡ predictions/keyword_rule.jsonl

Use `random.Random(seed)` to reset RNG each time generate_until to ensure that the same MockLM is called multiple times
The example produces exactly the same output - the expected values in the README can be reproduced."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Callable, Literal

from ..api import Doc, Request, Response
from .base import LM

MockMode = Literal["gold", "noisy", "constant", "rule"]


def default_rule_fn(text: str) -> str:
    """Keywords weak baseline: bad→negative / good→positive / other→neutral.

    Simple enough to be read by any interviewer, and capable of producing moderately strong predictions on fake datasets."""
    lower = text.lower()
    if "bad" in lower or "terrible" in lower or "awful" in lower:
        return "negative"
    if "good" in lower or "great" in lower or "love" in lower:
        return "positive"
    return "neutral"


class MockLM(LM):
    """Fake LLM, only implements generate_until (the only request type used in Phase 1)."""

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
        """Generate a response by mode. RNG is reset every batch to ensure that multiple calls to the same instance are completely consistent."""
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
