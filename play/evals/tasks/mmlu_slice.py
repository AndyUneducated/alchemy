"""Phase 1 general-capability **regression guard** baseline: MMLU 6-subject slice (96 examples).

Data source: [`data/mmlu_slice/SOURCE.md`](../data/mmlu_slice/SOURCE.md) (pinned HF revision + fetch script).

Teaching role (agent_sft view):
  - in-dist: nudge_fire_rate / agent_traj measure "capabilities affected by SFT"
  - OOD-A: bfcl_slice measures "original function-calling did not drop"
  - **OOD-B here**: mmlu_slice measures "general knowledge did not drop" — SFT data is all agent transcripts,
    classic catastrophic forgetting risk lives here.

Metric functions **inlined** (accuracy is a few lines of if/else; separate module would be "extract for extraction's sake"):

|metric|meaning|
|---|---|
|`accuracy`|hit rate on all 96 items: first letter ∈ {A,B,C,D}|
|`accuracy_by_subject`|(subgroup dict inside aggregation) 6 accuracies split by subject|

Evaluation protocol is **generate_until + take first letter** (not loglikelihood-of-letter) — closer to real deployment
feel, no logprobs API; trade-off: scores run slightly below original MMLU paper (when model omits A/B/C/D letter,
counted wrong).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, ClassVar

from ..api import Doc, Response, SampleResult
from ..registry import register_task
from .base import Task

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mmlu_slice" / "gold.jsonl"

PROMPT_TEMPLATE = (
    "The following is a multiple-choice question. "
    "Read the question and choose the best answer.\n\n"
    "Question: {question}\n\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
    "Answer:"
)


@register_task("mmlu_slice")
class MmluSlice(Task):
    """MMLU 6-subject 96-example slice, generate_until + take the first letter."""

    name: ClassVar[str] = "mmlu_slice"
    output_type: ClassVar[str] = "generate_until"

    def __init__(self) -> None:
        self.data_path = DATA_PATH

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
                    choices=tuple(row.get("choices", ())),
                    metadata=row.get("metadata", {}),
                )

    def doc_to_text(self, doc: Doc) -> str:
        choices = doc.metadata.get("raw_choices") or list(doc.choices or [])
        if len(choices) != 4:
            raise ValueError(f"mmlu_slice doc {doc.id!r} expects 4 choices, got {len(choices)}")
        a, b, c, d = choices
        return PROMPT_TEMPLATE.format(question=doc.input, a=a, b=b, c=c, d=d)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target or ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred_letter = parse_mcq_letter(response.text or "")
        target = (doc.target or "").upper()
        is_hit = 1.0 if pred_letter == target else 0.0
        # Disqualified predictions (no letters can be extracted) are also counted as 0; use artifact to distinguish "the model gave irrelevant characters" vs "given the wrong letters"
        return SampleResult(
            doc_id=doc.id,
            prediction=pred_letter or "",
            target=target,
            metrics={"accuracy": is_hit},
            artifacts={
                "subject": doc.metadata.get("subject", "unknown"),
                "raw_text": (response.text or "").strip(),
                "pred_letter": pred_letter,  # None means no letters are extracted
            },
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float | dict | None]]:
        return {
            "accuracy": _overall_accuracy,
            "accuracy_by_subject": _accuracy_by_subject,
        }

    def higher_is_better(self) -> dict[str, bool]:
        # Same convention as nudge_fire_rate: nested dict subgroups (accuracy_by_subject)
        # No advance higher_is_better - only scalar advance, dict is expanded key by key by CLI rendering layer
        return {"accuracy": True}


# ---- Inline measurement function (plan §2: mmlu accuracy only a few lines; metrics/ is not extracted) ----


_VALID_LETTERS = {"A", "B", "C", "D"}


def parse_mcq_letter(text: str) -> str | None:
    """Extract one of \"A/B/C/D\" from model output. Returns None if not found.

    Lenient parsing (try common LLM output pollution, in order):
      1. First non-empty line → strip markdown / punctuation
      2. If first char is letter, take it
      3. Else look for \"Answer: X\" echo
      4. Else scan for first isolated A/B/C/D (non-letter before/after)

    Step 4 \"isolated letter\" avoids \"according to A...\" false positives (isolated A still
    counts as echo; steps 1/2 filter letter-only outputs first).
    """
    if not text:
        return None
    s = text.strip()

    # The first line is not empty
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break

    s_clean = s.lstrip("*` ").rstrip(".,!?:;`*) ").strip()
    if not s_clean:
        return None

    # First character letter only / letter + punctuation
    if s_clean[0].upper() in _VALID_LETTERS:
        # Single character / followed by non-letter → Accept
        if len(s_clean) == 1 or not s_clean[1].isalpha():
            return s_clean[0].upper()

    # \"Answer: X\" / \"The answer is X\" etc. echo
    upper = s_clean.upper()
    for marker in ("ANSWER:", "ANSWER IS", "CORRECT ANSWER IS"):
        if marker in upper:
            after = upper.split(marker, 1)[1].lstrip(" *`(\"'")
            if after and after[0] in _VALID_LETTERS:
                return after[0]

    # Find isolated letters in the full text
    import re
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", upper)
    if m:
        return m.group(1)

    return None


def _overall_accuracy(srs: list[SampleResult]) -> float | None:
    if not srs:
        return None
    return sum(s.metrics.get("accuracy", 0.0) or 0.0 for s in srs) / len(srs)


def _accuracy_by_subject(srs: list[SampleResult]) -> dict[str, float] | None:
    """Nested dict: accuracy for each subject (identical to the aggregated crosscut subgroup schema)."""
    if not srs:
        return None
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in srs:
        subject = s.artifacts.get("subject", "unknown")
        bucket[subject].append(s.metrics.get("accuracy", 0.0) or 0.0)
    return {subj: sum(vals) / len(vals) for subj, vals in sorted(bucket.items())}
