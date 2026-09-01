"""Family 3 LLM-as-judge core paradigm (pointwise / pairwise / g_eval / self_consistency).

Trigger new creation according to README guideline #3: cross-task reuse (qa_open / future summarization / writing will be used)
+ There is no mature library to adjust (RAGAS is exclusive to RAG, deepeval is incompatible with task-decoupled of this project).

Phase 4 will be split into `judge_core.py` (this file) + `judge_rag.py` (RAG ground dimension),
Reason: The core paradigm is "scoring methodology", and the RAG dimension is "scoring object". The two layers are orthogonal to avoid single file expansion.

The main stage allocation of the four judges (see plan §6 for details):
  - judge_pointwise task layer main stage (lexical vs judge difference narrative on the task)
  - judge_pairwise Main stage of this file (position offset/swap debiasing)
  - g_eval Main stage of this file (multi-dimensional form-filling / multi-sampling instead of logprob)
  - self_consistency Main stage of this document (majority vote + tiebreak)

Design highlights:
  - **closure factory pattern**: `judge_pointwise(lm, ...)` returns `(doc, resp) -> float` closure,
    It is convenient for wrappers like self_consistency to be applied, and it is also convenient for task.process_results to reuse the same callable.
  - **Does not depend on logprob**: Ollama /api/chat does not return logprobs; G-Eval uses n-sample to sample multiple times
    Estimates a discrete distribution, equivalent to the non-logprob path of "logprob weighted mean". OpenAI adaptation can be added when it comes online
    `g_eval_logprob` Second level implementation, do not change the default.
  - **swap debiasing**: pairwise defaults to `swap=True` - A/B and B/A double running, inconsistency will be counted as tie,
    Treat "position offset" as noise rather than signal."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Literal, Sequence

from ..api import Doc, Request, Response
from ..models.base import LM

PairwiseVerdict = Literal["a", "b", "tie"]


# ---------- closure-internal LM calls recorder (DECISIONS §7.3 wave 3) ----------

class _JudgeRecorder:
    """Internal use of closure: block lm.generate_until call, append copy to responses list.

    DECISIONS §7.3: Let the judge closure self-report the number, and the runner passes task.collect_judge_responses
    Pull, hang aggregated["efficiency"]["judge"]——evaluation tool call class (hang on both paths).

    all judge factory (judge_pointwise / judge_pairwise / g_eval +
    5 RAG factories of metrics/judge_rag.py) internally directly `judge_lm.generate_until(...)`
    The call is changed to recorder, and the runner side pulls the response list from closure._recorder."""

    def __init__(self, lm: LM):
        self.lm = lm
        self.model_label = lm.name
        self.responses: list[Response] = []

    def call(self, requests: list[Request]) -> list[Response]:
        out = self.lm.generate_until(requests)
        self.responses.extend(out)
        return out

    def reset(self) -> None:
        """testing helper: Manually clear the state when reusing a closure."""
        self.responses = []


# ----------Default prompt template ----------

DEFAULT_POINTWISE_TEMPLATE = (
    "Rate the response on a scale of 1-5 (1=poor, 5=excellent).\n"
    "Question: {input}\n"
    "Reference answer: {reference}\n"
    "Response: {response}\n"
    "Score (1-5):"
)

DEFAULT_PAIRWISE_TEMPLATE = (
    "Compare two responses to the question. Choose A, B, or tie.\n"
    "Question: {input}\n"
    "Reference: {reference}\n"
    "Response A: {response_a}\n"
    "Response B: {response_b}\n"
    "Better response (A/B/tie):"
)

DEFAULT_G_EVAL_TEMPLATE = (
    "Rate the response on the dimension '{dimension}' from 1-5.\n"
    "Question: {input}\n"
    "Reference: {reference}\n"
    "Response: {response}\n"
    "Score (1-5):"
)


# ---------- parse helpers ----------

def parse_pointwise_score(text: str, *, scale: tuple[int, int] = (1, 5)) -> int:
    """Extract int score from judge output text.

    Parsing strategy (robust first):
      ① Find all integers (including negative numbers) in text
      ② If any int falls within the range [lo, hi] → return the first one
      ③ Otherwise, clamp the first int to [lo, hi] and return
      ④ There is not an int → ValueError

    Example:
      "Score: 4/5" → found [4, 5], 4 in [1,5] → 4
      "Score: 7/5" → found [7, 5], 5 in [1,5] → 5 (priority in-range)
      "0" → found [0], no in-range → clamp(0)=1
      "999" → found [999], no in-range → clamp(999)=5
      "totally not a score" → None int → ValueError"""
    lo, hi = scale
    ints = [int(m) for m in re.findall(r"-?\d+", text or "")]
    if not ints:
        raise ValueError(f"could not parse score from {text!r}")
    in_range = [n for n in ints if lo <= n <= hi]
    if in_range:
        return in_range[0]
    return max(lo, min(hi, ints[0]))


def parse_pairwise_verdict(text: str) -> PairwiseVerdict:
    """Extract A/B/tie verdict from judge output (case-insensitive).

    Priority:
      ① explicit "tie" / "equal" / "draw" / "neither" → tie
      ② prefer \\b[Aa]\\b → "a", \\b[Bb]\\b → "b"
      ③ if both 'A' and 'B' appear, take first occurrence
      ④ no signal at all → tie (conservative)
    """
    s = (text or "").strip().lower()
    if not s:
        return "tie"
    # tie / equal / draw / neither
    if re.search(r"\b(tie|equal|draw|neither|both|same)\b", s):
        return "tie"
    # find standalone A or B token
    m = re.search(r"\b([ab])\b", s)
    if m:
        return m.group(1)  # type: ignore[return-value]
    # fallback: any A/B mention, first one wins
    for ch in s:
        if ch == "a":
            return "a"
        if ch == "b":
            return "b"
    return "tie"


# ---------- judge_pointwise（closure factory）----------

def judge_pointwise(
    judge_lm: LM,
    *,
    prompt_template: str = DEFAULT_POINTWISE_TEMPLATE,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float | None]:
    """Generates a closure of (doc, response) -> float score | None.

    Template fields: `{input}` / `{reference}` / `{response}`. Missing fields will be ignored by .format
    (If the template is not quoted), so the test can use the minimalist template `"rate: {response}"`.

    DECISIONS §7.3: closure holds `_recorder` attribute for task.collect_judge_responses to pull
    judge LM calls the record, and the runner hangs `aggregated["efficiency"]["judge"]`.

    DECISIONS §X (wave 4): parser throws ValueError (LM output no int to parse) → closure returns None;
    Consistent with the "None vs 0 semantic separation" principle established by phase 7 wave 2 P2 - 1-5 scale 0 is out of bounds,
    None explicit table "not measured", the downstream aggregator filters the empty set →None, the same shape as safety.judge_safety_score."""
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float | None:
        prompt = prompt_template.format(
            input=doc.input,
            reference=doc.target,
            response=response.text or "",
        )
        req = Request(
            doc_id=doc.id, prompt=prompt,
            request_type="generate_until", max_tokens=max_tokens,
        )
        [resp] = rec.call([req])
        try:
            return float(parse_pointwise_score(resp.text or "", scale=scale))
        except ValueError:
            return None

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_pairwise + pairwise_winrate ----------

def judge_pairwise(
    judge_lm: LM,
    *,
    prompt_template: str = DEFAULT_PAIRWISE_TEMPLATE,
    swap: bool = True,
    max_tokens: int = 16,
) -> Callable[[Doc, Response, Response], PairwiseVerdict]:
    """Returns a closure of (doc, resp_a, resp_b) -> "a"/"b"/"tie".

    `swap=True` (default): Double run A/B and B/A, and the winner will be calculated after the two results are consistent after the translation back to the original sequence.
    Otherwise count tie - this is the standard practice to remove "positional bias" (Zheng et al. 2023, MT-Bench).

    DECISIONS §7.3: The closure holds the `_recorder` attribute for task.collect_judge_responses to pull."""
    rec = _JudgeRecorder(judge_lm)

    def _ask(doc: Doc, a: Response, b: Response) -> PairwiseVerdict:
        prompt = prompt_template.format(
            input=doc.input,
            reference=doc.target,
            response_a=a.text or "",
            response_b=b.text or "",
        )
        req = Request(
            doc_id=doc.id, prompt=prompt,
            request_type="generate_until", max_tokens=max_tokens,
        )
        [r] = rec.call([req])
        return parse_pairwise_verdict(r.text or "")

    def _verdict(doc: Doc, resp_a: Response, resp_b: Response) -> PairwiseVerdict:
        v1 = _ask(doc, resp_a, resp_b)
        if not swap:
            return v1
        v2_raw = _ask(doc, resp_b, resp_a)
        # translate v2 back to original ordering: in swapped call,
        # "a" means resp_b wins → original "b"; "b" means resp_a wins → original "a"
        v2 = {"a": "b", "b": "a", "tie": "tie"}[v2_raw]
        if v1 == v2:
            return v1
        return "tie"

    _verdict._recorder = rec  # type: ignore[attr-defined]
    return _verdict


def pairwise_winrate(
    judge_lm: LM,
    pairs: Sequence[tuple[Doc, Response, Response]],
    *,
    prompt_template: str = DEFAULT_PAIRWISE_TEMPLATE,
    swap: bool = True,
) -> dict[str, float]:
    """Aggregate the pairwise verdict → {a, b, tie} proportions of multiple pairs of samples.

    cross-task utility: score-pairwise CLI (phase 3.5) will adjust directly."""
    verdict_fn = judge_pairwise(judge_lm, prompt_template=prompt_template, swap=swap)
    counts = {"a": 0, "b": 0, "tie": 0}
    for doc, ra, rb in pairs:
        counts[verdict_fn(doc, ra, rb)] += 1
    n = sum(counts.values())
    if n == 0:
        return {"a": 0.0, "b": 0.0, "tie": 0.0}
    return {k: v / n for k, v in counts.items()}


# ---------- g_eval ----------

def g_eval(
    judge_lm: LM,
    *,
    dimensions: Sequence[str],
    prompt_template: str = DEFAULT_G_EVAL_TEMPLATE,
    n_samples: int = 5,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], dict[str, float | None]]:
    """Returns a closure of (doc, response) -> {dim: score | None}.

    `n_samples` samples per dimension + mean - alternative to logprob for weighted discrete distribution estimates (OpenAI does not have logprobs
    local ollama/compatible path).

    Template fields: `{input}` / `{reference}` / `{response}` / `{dimension}`.

    DECISIONS §7.3: The closure holds the `_recorder` attribute for task.collect_judge_responses to pull.

    DECISIONS §X (wave 4): A single sample parser failed → skip the sample; the valid sample in this dimension is all empty
    → Dimension returns None ("not measured" placeholder, the same shape as phase 7 P2); partial failure is reported as valid mean."""
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for dim in dimensions:
            scores: list[int] = []
            for _ in range(n_samples):
                prompt = prompt_template.format(
                    input=doc.input,
                    reference=doc.target,
                    response=response.text or "",
                    dimension=dim,
                )
                req = Request(
                    doc_id=doc.id, prompt=prompt,
                    request_type="generate_until", max_tokens=max_tokens,
                )
                [resp] = rec.call([req])
                try:
                    scores.append(parse_pointwise_score(resp.text or "", scale=scale))
                except ValueError:
                    continue
            out[dim] = sum(scores) / len(scores) if scores else None
        return out

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- self_consistency wrapper ----------

def self_consistency(
    base_judge: Callable[..., Any],
    *,
    n_samples: int = 5,
) -> Callable[..., Any]:
    """Wrap any base_judge into a "sample N times and take the majority" version.

    Applies to:
      - judge_pointwise closure (mode of int score)
      - any callable that returns a hashable (pairwise verdict / category label / ...)

    Tie break takes the first-seen tiebreak - deterministic, avoiding dictionary order/randomness.

    DECISIONS §7.3: Transparently transmit the `_recorder` attribute of base - multiple calls inside base will be accessed by the same recorder
    To collect and wrapper, you don’t need to open a new recorder yourself."""

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        results = [base_judge(*args, **kwargs) for _ in range(n_samples)]
        counts = Counter(results)
        top = max(counts.values())
        for r in results:
            if counts[r] == top:
                return r
        return results[0]  # unreachable; mypy/pylint friendly

    if hasattr(base_judge, "_recorder"):
        _wrapped._recorder = base_judge._recorder  # type: ignore[attr-defined]
    return _wrapped
