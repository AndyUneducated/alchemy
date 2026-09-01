"""Family 4 RAG ground dimension (5 judge_xxx closure factories + 2 RAG-specific parsers).

Why build `judge_rag.py` separately instead of stuffing it into `judge_core.py`:
  - judge_core is the "scoring methodology" (pointwise / pairwise / g_eval / self_consistency)
  - judge_rag is "scoring object = each link of RAG pipeline" (faithfulness / answer_correctness / ...)
  Two levels of orthogonality: The extension of the judge LM paradigm to the 5th does not drag down the evolution of judge_rag (DECISIONS §4 decision).

5 dimensions aligned to the RAGAS main framework (faithfulness / answer_correctness / context_precision /
context_recall / answer_relevancy), but without importing ragas directly:
  - Dependency expansion (langchain/openai/data science family bucket)
  - We already have LM ABC and closure factory patterns, and it only takes ~150 lines to run the same path.
  - Lock the prompt literal string (lm-eval invariant), the prompt inside ragas is a black box

Data contract (path B+C, see DECISIONS §4):
  - `response.text` loads LM-side output (answer string)
  - `doc.metadata['contexts']` installs retrieval products (list[str], injected by rag_qa.process_docs)
  - `doc.target` installs gold answers (for answer_correctness / context_recall)

Each judge_xxx is a closure factory: returns `Callable[[Doc, Response], float]`,
Identical to the judge_core.judge_pointwise protocol - self_consistency can also be used."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from ..api import Doc, Request, Response
from ..models.base import LM


# ----------Default prompt template-------------------------------------------------

DEFAULT_CLAIM_EXTRACT_TEMPLATE = (
    "Decompose the following text into atomic factual statements. "
    "Output each statement on a new line, prefixed with '- '.\n\n"
    "Text: {text}\n\n"
    "Statements:"
)

DEFAULT_NLI_TEMPLATE = (
    "Given the following context, can the statement be inferred? "
    "Answer 'yes' or 'no'.\n\n"
    "Context:\n{context}\n\n"
    "Statement: {statement}\n\n"
    "Answer:"
)

DEFAULT_TP_FP_FN_TEMPLATE = (
    "Compare the response to the reference answer. Count three quantities:\n"
    "  TP = facts in response that ALSO appear in reference\n"
    "  FP = facts in response that DO NOT appear in reference\n"
    "  FN = facts in reference that DO NOT appear in response\n\n"
    "Reference: {reference}\n"
    "Response: {response}\n\n"
    "Output exactly three integers in the form 'TP=<int> FP=<int> FN=<int>'.\n"
    "Counts:"
)

DEFAULT_CONTEXT_RELEVANCE_TEMPLATE = (
    "Given the question, is the following context useful for answering it? "
    "Answer 'yes' or 'no'.\n\n"
    "Question: {input}\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

DEFAULT_ANSWER_RELEVANCE_TEMPLATE = (
    "Rate how relevant the response is to the question on a scale of 1-5 "
    "(5=fully on-topic, 1=completely off-topic; ignore factual correctness).\n\n"
    "Question: {input}\n"
    "Response: {response}\n\n"
    "Score (1-5):"
)


# ---------- RAG-specific parsers --------------------------------------------------

def parse_statement_list(text: str) -> list[str]:
    """Extract bulleted/numbered list items from LLM output → list[str].

    Supported prefixes: `- foo` / `* foo` / `1. foo` / `1) foo` / Pure newline lines are also accepted.
    Trim blank, filter empty lines. This is faithfulness/context_recall/answer_correctness
    Decomposition entry shared by three metrics."""
    lines = (text or "").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Strip common bullet prefixes
        s = re.sub(r"^[\-\*•]\s+", "", s)
        s = re.sub(r"^\d+[\.\)]\s+", "", s)
        s = s.strip()
        if s:
            out.append(s)
    return out


def parse_tp_fp_fn(text: str) -> tuple[int, int, int]:
    """Draw (TP, FP, FN) three integers from the judge output.

    Parsing strategy (robust first):
      ① First find the named token with explicit form like 'TP=3' / 'FP: 1' / 'FN 2'
      ② If none of the three keys match, fall back to the first three non-negative integers.

    Any key is missing → ValueError, forcing the caller to know that the judge output is disqualified."""
    s = text or ""
    found: dict[str, int] = {}
    for key in ("TP", "FP", "FN"):
        m = re.search(rf"\b{key}\s*[:=]?\s*(\d+)", s, re.IGNORECASE)
        if m:
            found[key] = int(m.group(1))
    if all(k in found for k in ("TP", "FP", "FN")):
        return found["TP"], found["FP"], found["FN"]

    # fallback: first 3 non-negative ints
    ints = [int(m) for m in re.findall(r"\d+", s)]
    if len(ints) >= 3:
        return ints[0], ints[1], ints[2]

    raise ValueError(f"could not parse TP/FP/FN from {text!r}")


def _parse_yes_no(text: str) -> int:
    """LLM 'yes/no' output → 1/0; conservative strategy: ambiguity returns 0 (default is not-supported on NLI failure)."""
    s = (text or "").strip().lower()
    if not s:
        return 0
    # Give priority to whether the first word is 'yes'/'no'
    first = re.split(r"\W+", s, maxsplit=1)[0]
    if first in {"yes", "y", "true", "supported", "entailed", "1"}:
        return 1
    if first in {"no", "n", "false", "unsupported", "0"}:
        return 0
    # Full text search
    if re.search(r"\byes\b", s):
        return 1
    return 0


def _ask(lm_or_rec, prompt: str, *, doc_id: str = "", max_tokens: int = 64) -> str:
    """Single-round LM call assistant - compress the batch in/out of LM ABC into a single prompt → text form.

    DECISIONS §7.3: Accepts `LM` or `_JudgeRecorder` (duck-typing: has `.call`);
    When the recorder is used inside the factory, the LM call records will be collected for consumption by the efficiency.judge.* subgroup."""
    req = Request(
        doc_id=doc_id, prompt=prompt,
        request_type="generate_until", max_tokens=max_tokens,
    )
    if hasattr(lm_or_rec, "call"):
        # _JudgeRecorder
        [resp] = lm_or_rec.call([req])
    else:
        [resp] = lm_or_rec.generate_until([req])
    return resp.text or ""


# ---------- judge_faithfulness ----------------------------------------------

def judge_faithfulness(
    judge_lm: LM,
    *,
    claim_template: str = DEFAULT_CLAIM_EXTRACT_TEMPLATE,
    nli_template: str = DEFAULT_NLI_TEMPLATE,
    max_tokens_extract: int = 256,
    max_tokens_nli: int = 16,
) -> Callable[[Doc, Response], float]:
    """Two-step judge:
      ① Let judge_lm split `response.text` into atomic claims (parse_statement_list parsing)
      ② Let judge_lm pair `doc.metadata['contexts']` claim by claim (spelled into single context block)
         Check yes/no (NLI style). Return #supported / #total.

    `contexts` comes from `doc.metadata`, following path B+C: retriever products live on the doc side.
    No contexts / no resolvable claim → 0.0 (conservative).

    DECISIONS §7.3: The closure holds the `_recorder` property for task.collect_judge_responses to pull."""
    from .judge_core import _JudgeRecorder
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float:
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not contexts:
            return 0.0
        text = (response.text or "").strip()
        if not text:
            return 0.0

        # Step 1: extract claims
        extract_prompt = claim_template.format(text=text)
        raw = _ask(rec, extract_prompt, doc_id=doc.id, max_tokens=max_tokens_extract)
        claims = parse_statement_list(raw)
        if not claims:
            return 0.0

        # Step 2: NLI per claim against the joined contexts
        joined_ctx = "\n---\n".join(contexts)
        supported = 0
        for claim in claims:
            nli_prompt = nli_template.format(context=joined_ctx, statement=claim)
            verdict = _ask(rec, nli_prompt, doc_id=doc.id, max_tokens=max_tokens_nli)
            supported += _parse_yes_no(verdict)
        return supported / len(claims)

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_answer_correctness ----------------------------------------

def judge_answer_correctness(
    judge_lm: LM,
    *,
    template: str = DEFAULT_TP_FP_FN_TEMPLATE,
    max_tokens: int = 64,
) -> Callable[[Doc, Response], float | None]:
    """Single-step judge: let judge_lm count TP/FP/FN (fact level), return F1 | None.

    Does not rely on retrieval contexts - only response and doc.target, in the grounding dimension
    The most direct "endpoint quality" indicator. The advantage of F1 is that precision (1-FP/(TP+FP)) and
    recall (1-FN/(TP+FN)) double constraint.

    No target / response empty / TP+FP+FN=0 → 0.0 (degenerate-input: empty input
    of F1=0 is the legal minimum); parse fails → None (DECISIONS §X wave 4, with phase 7 P2
    "Not measured" placeholder is the same shape).

    DECISIONS §7.3: closure holds the `_recorder` attribute."""
    from .judge_core import _JudgeRecorder
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float | None:
        target = doc.target or ""
        pred = (response.text or "").strip()
        if not target or not pred:
            return 0.0
        prompt = template.format(reference=target, response=pred)
        try:
            raw = _ask(rec, prompt, doc_id=doc.id, max_tokens=max_tokens)
            tp, fp, fn = parse_tp_fp_fn(raw)
        except ValueError:
            return None
        if tp + fp + fn == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_context_precision -----------------------------------------

def judge_context_precision(
    judge_lm: LM,
    *,
    template: str = DEFAULT_CONTEXT_RELEVANCE_TEMPLATE,
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float]:
    """Let judge_lm judge "whether it is useful to answer query yes/no" context by context (in rank order),
    Returns the proportion of the relevant context (binary precision of top-k).

    Difference from metrics/retrieval.precision_at_k: that one is based on doc_id and gold_ids
    Compared with set-membership; this is LLM directly judging "whether the semantics are relevant" - closer to the "context quality"
    Rather than "whether the recalled doc is in the gold collection".

    No contexts → 0.0; prediction is not involved (this is the relationship between query and context).

    DECISIONS §7.3: closure holds the `_recorder` attribute."""
    from .judge_core import _JudgeRecorder
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float:
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not contexts:
            return 0.0
        relevant = 0
        for ctx in contexts:
            prompt = template.format(input=doc.input, context=ctx)
            verdict = _ask(rec, prompt, doc_id=doc.id, max_tokens=max_tokens)
            relevant += _parse_yes_no(verdict)
        return relevant / len(contexts)

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_context_recall --------------------------------------------

def judge_context_recall(
    judge_lm: LM,
    *,
    claim_template: str = DEFAULT_CLAIM_EXTRACT_TEMPLATE,
    nli_template: str = DEFAULT_NLI_TEMPLATE,
    max_tokens_extract: int = 256,
    max_tokens_nli: int = 16,
) -> Callable[[Doc, Response], float]:
    """Two-step judge:
      ① Split `doc.target` (gold answer) into atomic claims
      ② Determine "whether it can be deduced from `doc.metadata['contexts']` claim by claim, and return the proportion.

    Dual to judge_faithfulness:
      - faithfulness: the claim of response is in contexts → "I can see your answer in the material"
      - context_recall: target's claim is in contexts → "How much is covered by the factual material in the standard answer"

    No target / no resolvable claim → 0.0.

    DECISIONS §7.3: closure holds the `_recorder` attribute."""
    from .judge_core import _JudgeRecorder
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float:
        target = doc.target or ""
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not target or not contexts:
            return 0.0

        extract_prompt = claim_template.format(text=target)
        raw = _ask(rec, extract_prompt, doc_id=doc.id, max_tokens=max_tokens_extract)
        claims = parse_statement_list(raw)
        if not claims:
            return 0.0

        joined_ctx = "\n---\n".join(contexts)
        attributable = 0
        for claim in claims:
            nli_prompt = nli_template.format(context=joined_ctx, statement=claim)
            verdict = _ask(rec, nli_prompt, doc_id=doc.id, max_tokens=max_tokens_nli)
            attributable += _parse_yes_no(verdict)
        return attributable / len(claims)

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_answer_relevancy ------------------------------------------

def judge_answer_relevancy(
    judge_lm: LM,
    *,
    template: str = DEFAULT_ANSWER_RELEVANCE_TEMPLATE,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float | None]:
    """Single-step judge: rate 1-5 "Answer whether there is a positive address question" (regardless of whether it is right or wrong).

    Complementary to judge_answer_correctness:
      - correctness: whether the facts are correct (see target)
      - relevancy: Whether you are answering this question (do not look at the target, look at the input vs response topic alignment)

    Parse reuse `judge_core.parse_pointwise_score` (same source 1-5 form).
    No response → 0.0 (relevancy=0 of empty pred is the legal minimum score); parse failed → None
    (DECISIONS §X wave 4, 1-5 scale 0 is out of bounds, None explicitly means "not measured", the same shape as phase 7 P2).

    DECISIONS §7.3: closure holds the `_recorder` attribute."""
    from .judge_core import _JudgeRecorder, parse_pointwise_score  # Avoid circular import
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float | None:
        pred = (response.text or "").strip()
        if not pred:
            return 0.0
        prompt = template.format(input=doc.input, response=pred)
        raw = _ask(rec, prompt, doc_id=doc.id, max_tokens=max_tokens)
        try:
            return float(parse_pointwise_score(raw, scale=scale))
        except ValueError:
            return None

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score
