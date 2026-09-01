"""metrics/judge_rag.py unit layer: 5 RAG judges + 2 RAG dedicated parsers.

Zero network. FakeJudgeLM reuses the stub of test_judge_core (advance by cursor/rule function dual strategy).
Focus on:
  - Robust parsing with 2 new parsers (parse_statement_list / parse_tp_fp_fn)
  - 5 closure shapes of judge_xxx + numerical boundaries + path B+C data contract (contexts in doc.metadata)"""

from __future__ import annotations

import pytest

from evals.api import Doc, Response
from evals.metrics.judge_rag import (
    judge_answer_correctness,
    judge_answer_relevancy,
    judge_context_precision,
    judge_context_recall,
    judge_faithfulness,
    parse_statement_list,
    parse_tp_fp_fn,
)
from evals.tests.test_judge_core import FakeJudgeLM


def _doc_with_ctx(*, target: str | None = "ref", contexts=("ctx1", "ctx2")) -> Doc:
    """Construct a Doc with retrieved contexts (path B+C: contexts and doc.metadata)."""
    return Doc(
        id="d0",
        input="q?",
        target=target,
        metadata={"contexts": tuple(contexts)},
    )


def _resp(text: str = "hyp") -> Response:
    return Response(doc_id="d0", text=text)


# ---------- parse_statement_list (4 items)---------------------------------------------

def test_parse_statement_list_handles_dash_bullets():
    text = "- 巴黎是法国首都。\n- 法国位于欧洲。"
    assert parse_statement_list(text) == ["巴黎是法国首都。", "法国位于欧洲。"]


def test_parse_statement_list_handles_numbered():
    text = "1. fact one\n2) fact two\n3. fact three"
    assert parse_statement_list(text) == ["fact one", "fact two", "fact three"]


def test_parse_statement_list_skips_empty_lines():
    text = "\n- a\n\n- b\n   \n- c\n"
    assert parse_statement_list(text) == ["a", "b", "c"]


def test_parse_statement_list_returns_empty_on_blank():
    assert parse_statement_list("") == []
    assert parse_statement_list("\n\n  \n") == []


# ---------- parse_tp_fp_fn (4 items)---------------------------------------------

def test_parse_tp_fp_fn_named_tokens():
    """Explicit 'TP=3 FP=1 FN=2' form."""
    assert parse_tp_fp_fn("TP=3 FP=1 FN=2") == (3, 1, 2)


def test_parse_tp_fp_fn_loose_punctuation():
    """With colon / mixed case / without equal sign also accepted."""
    assert parse_tp_fp_fn("Counts: TP: 5 fp 0 FN=4") == (5, 0, 4)


def test_parse_tp_fp_fn_fallback_first_three_ints():
    """If there is no named token, fallback to the first 3 integers."""
    assert parse_tp_fp_fn("answers: 2 1 3 (extra: 99)") == (2, 1, 3)


def test_parse_tp_fp_fn_invalid_raises():
    """Less than 3 integer → ValueError (forces caller to know that judge outputs disqualification)."""
    with pytest.raises(ValueError):
        parse_tp_fp_fn("just two: 1 2")


# ---------- judge_faithfulness (3 items) ------------------------------------------

def test_faithfulness_full_supported():
    """response splits 2 claims, both claims are connected by 'yes' → faithfulness=1.0.

    FakeJudgeLM cursor sequence: [extract result, nli yes, nli yes]."""
    fake = FakeJudgeLM(outputs=[
        "- claim A\n- claim B",  # extract → 2 claims
        "yes",                     # NLI claim A
        "yes",                     # NLI claim B
    ])
    f = judge_faithfulness(fake)
    assert f(_doc_with_ctx(), _resp("answer")) == 1.0


def test_faithfulness_partial_half():
    """3 claims in 2 supported / 1 unsupported → 2/3."""
    fake = FakeJudgeLM(outputs=[
        "- a\n- b\n- c",
        "yes", "no", "yes",
    ])
    f = judge_faithfulness(fake)
    val = f(_doc_with_ctx(), _resp("answer"))
    assert abs(val - 2 / 3) < 1e-9


def test_faithfulness_no_contexts_zero():
    """contexts is empty → directly 0.0 (does not burn judge call)."""
    fake = FakeJudgeLM(outputs=["should not be reached"])
    f = judge_faithfulness(fake)
    doc_no_ctx = Doc(id="d0", input="q?", target="ref", metadata={"contexts": ()})
    assert f(doc_no_ctx, _resp("answer")) == 0.0


# ---------- judge_answer_correctness (3 items)--------------------------------

def test_answer_correctness_perfect_f1():
    """TP=3 FP=0 FN=0 → P=1, R=1, F1=1.0."""
    fake = FakeJudgeLM(outputs=["TP=3 FP=0 FN=0"])
    ac = judge_answer_correctness(fake)
    assert ac(_doc_with_ctx(target="gold"), _resp("good")) == 1.0


def test_answer_correctness_balanced_50pct():
    """TP=1 FP=1 FN=1 → P=0.5, R=0.5, F1=0.5."""
    fake = FakeJudgeLM(outputs=["TP=1 FP=1 FN=1"])
    ac = judge_answer_correctness(fake)
    assert abs(ac(_doc_with_ctx(target="gold"), _resp("partial")) - 0.5) < 1e-9


def test_answer_correctness_zero_when_no_overlap():
    """TP=0 → F1=0.0 (precision+recall=0 short circuit)."""
    fake = FakeJudgeLM(outputs=["TP=0 FP=4 FN=2"])
    ac = judge_answer_correctness(fake)
    assert ac(_doc_with_ctx(target="gold"), _resp("wrong")) == 0.0


def test_answer_correctness_returns_none_on_parse_failure():
    """DECISIONS §X wave 4: judge outputs no TP/FP/FN three integers → returns None "not measured"
    (Instead of 0.0; distinguishes "the judge didn't count" vs "the judge counted to 0 TP", aligning with phase 7 P2 style).

    parse failed("No int triple") = Unmeasured → None;
    degenerate path (target/pred is empty / TP+FP+FN=0 / P+R=0) reserved 0.0 = legal minimum score."""
    fake = FakeJudgeLM(outputs=["totally not parseable"])
    ac = judge_answer_correctness(fake)
    assert ac(_doc_with_ctx(target="gold"), _resp("hyp")) is None


# ---------- judge_context_precision (2 items)--------------------------------

def test_context_precision_all_relevant():
    """2 contexts are yes → 1.0."""
    fake = FakeJudgeLM(outputs=["yes", "yes"])
    cp = judge_context_precision(fake)
    assert cp(_doc_with_ctx(contexts=("a", "b")), _resp()) == 1.0


def test_context_precision_half_relevant():
    """3 contexts: yes/no/yes → 2/3."""
    fake = FakeJudgeLM(outputs=["yes", "no", "yes"])
    cp = judge_context_precision(fake)
    val = cp(_doc_with_ctx(contexts=("a", "b", "c")), _resp())
    assert abs(val - 2 / 3) < 1e-9


# ---------- judge_context_recall (2 items) ----------------------------------

def test_context_recall_full():
    """target split 2 claims, both were yes → 1.0."""
    fake = FakeJudgeLM(outputs=["- gold A\n- gold B", "yes", "yes"])
    cr = judge_context_recall(fake)
    assert cr(_doc_with_ctx(target="gold answer"), _resp()) == 1.0


def test_context_recall_partial():
    """3 claim in 2 attributable → 2/3."""
    fake = FakeJudgeLM(outputs=["- a\n- b\n- c", "yes", "no", "yes"])
    cr = judge_context_recall(fake)
    val = cr(_doc_with_ctx(target="gold"), _resp())
    assert abs(val - 2 / 3) < 1e-9


# ---------- judge_answer_relevancy (2 items) ----------------------------------

def test_answer_relevancy_pointwise_5():
    """Single prompt 1-5 rating, judge output 5 → 5.0."""
    fake = FakeJudgeLM(outputs=["5"])
    ar = judge_answer_relevancy(fake)
    assert ar(_doc_with_ctx(), _resp("on-topic answer")) == 5.0


def test_answer_relevancy_zero_on_empty_response():
    """response empty → straight 0.0 (does not burn judge call)."""
    fake = FakeJudgeLM(outputs=["unused"])
    ar = judge_answer_relevancy(fake)
    assert ar(_doc_with_ctx(), Response(doc_id="d0", text="")) == 0.0


def test_answer_relevancy_returns_none_on_parse_failure():
    """DECISIONS §X wave 4: 1-5 scale parse failed → return None "Not measured"
    (Distinguish empty pred=0.0 legal minimum score vs parse failure=None;
    Identical to judge_pointwise / judge_safety_score None placeholder protocol)."""
    fake = FakeJudgeLM(outputs=["the response was generally"])
    ar = judge_answer_relevancy(fake)
    # pred is not empty → go to parse; parse fails → None (not empty pred 0.0)
    assert ar(_doc_with_ctx(), _resp("non-empty pred")) is None
