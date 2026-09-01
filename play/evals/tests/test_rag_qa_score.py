"""rag_qa task score path e2e (FakeJudgeLM zero network) + 4 copies of the stub diagnostic narrative.

Demonstrate the two core narratives of phase 4 RAG QA:
  - **paraphrase**: lexical (em / rouge_l) fails, but grounding (faithfulness /
    answer_correctness) can still be retained - judge comes to the rescue (core narrative)
  - **wrong_fact**: lexical looks okay (rouge_l high, few character replacements), grounding
    Indicators capture the wrong facts (reverse narrative)

Use FakeJudgeLM with rule-based outputs to judge LM (points are given based on the prompt content heuristic).
True LLM e2e in test_rag_live.py via vdb-probe gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.api import Doc, Request, Response
from evals.models.base import LM
from evals.runner import evaluate_score
from evals.tasks.rag_qa import RagQA
from evals.tests.test_judge_core import FakeJudgeLM

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "rag_qa" / "predictions"


def _score(pred_name: str, judge_lm: LM | None = None) -> dict[str, float]:
    task = RagQA(judge_lm=judge_lm)
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 8
    return r.aggregated


# ---------- Upper and lower bounds sanity (lexical only, no judge_lm) -----------------------

def test_perfect_lexical_high():
    """perfect = gold answer → em=1.0 / rouge_l~1.0."""
    agg = _score("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["rouge_l"] >= 0.99


def test_garbage_lexical_low():
    """garbage is completely irrelevant → em=0 / rouge_l is low."""
    agg = _score("garbage")
    assert agg["exact_match"] == 0.0
    assert agg["rouge_l"] <= 0.1


# ---------- Core narrative: paraphrase lexical fall / grounding rescue ----------

def _yes_judge_with_perfect_correctness() -> LM:
    """rule-based fake judge:
      - claim extract class prompt → returns 1 fixed claim (let faithfulness/recall go the ratio=1.0 path)
      - NLI yes/no class prompt → always 'yes' (contexts perfectly matched optimistic judge)
      - TP/FP/FN type prompt → TP=3 FP=0 FN=0（answer_correctness=1.0）
      - 1-5 rating category prompt → '5' (answer_relevancy full score)

    Simulates "optimistic judge ignoring literal changes": paraphrase on grounding should ≈ 1.0."""

    def rule(prompt: str) -> str:
        p_low = prompt.lower()
        if "tp" in p_low and "fp" in p_low and "fn" in p_low:
            return "TP=3 FP=0 FN=0"
        if "decompose" in p_low or "atomic" in p_low or "statement" in p_low and "context" not in p_low:
            return "- claim X"
        if "1-5" in p_low or "score (1-5)" in p_low:
            return "5"
        # Default yes/no NLI / context relevance / faithfulness
        return "yes"

    return FakeJudgeLM(outputs=rule)


def _strict_judge() -> LM:
    """rule-based fake judge: Use "prediction == 1 statement, NLI failure → 0 points" to simulate a strict judge."""

    def rule(prompt: str) -> str:
        p_low = prompt.lower()
        if "tp" in p_low and "fp" in p_low and "fn" in p_low:
            return "TP=0 FP=3 FN=2"  # Totally wrong → F1=0
        if "decompose" in p_low or "atomic" in p_low or "statement" in p_low and "context" not in p_low:
            return "- claim X\n- claim Y"
        if "1-5" in p_low or "score (1-5)" in p_low:
            return "1"
        return "no"  # Strict NLI

    return FakeJudgeLM(outputs=rule)


def test_paraphrase_lexical_drops_grounding_holds():
    """**Core narrative**: paraphrase em=0, rouge_l is medium, but the lenient judge gives full marks to grounding.

    The lenient judge will answer yes when seeing "Semantically Paired NLI" - proving that the grounding metric can still distinguish when lexical fails."""
    agg_lex = _score("paraphrase")  # Lexical only
    assert agg_lex["exact_match"] == 0.0
    assert agg_lex["rouge_l"] < 0.7  # All words are replaced with light, and the char-level rouge is low.
    # After adding lenient judge, the grounding dimension is full score
    agg_judge = _score("paraphrase", _yes_judge_with_perfect_correctness())
    assert agg_judge["faithfulness"] == 1.0
    assert agg_judge["answer_correctness"] == 1.0
    assert agg_judge["context_precision"] == 1.0
    assert agg_judge["context_recall"] == 1.0
    assert agg_judge["answer_relevancy"] == 5.0


# ---------- Reverse narrative: lexical misjudgment / grounding on wrong_fact -----------

def test_wrong_fact_lexical_high_grounding_low():
    """**Reverse narrative**: rouge_l is high on wrong_fact (a small amount of character replacement), but low grounding is given by strict judge.

    Key comparison: strict judge simulates "identifying factual errors" - this is a blind spot that lexical cannot catch.
    Face to face with the real LLM judge in the e2e live test."""
    agg_lex = _score("wrong_fact")  # Lexical only
    assert agg_lex["exact_match"] == 0.0
    assert agg_lex["rouge_l"] >= 0.7  # Only replace numbers → rouge is still high

    agg_judge = _score("wrong_fact", _strict_judge())
    assert agg_judge["faithfulness"] == 0.0  # NLI all 'no'
    assert agg_judge["answer_correctness"] == 0.0  # F1 = 0
    # Grounding is significantly lower than lexical, indicating that the judge caught the fact that lexical missed the judgment.
    assert agg_judge["faithfulness"] < agg_lex["rouge_l"] - 0.5


# ----------Framework invariants (relocking per task)-------------------------------------

def test_n_matches_gold():
    """n == number of rows in the data set."""
    task = RagQA()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 8


def test_score_missing_pred_raises(tmp_path):
    """Missing doc_id strict KeyError (same contract as sentiment/mt/qa_open)."""
    task = RagQA()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"qNONE","prediction":"x","contexts":["c"],"retrieved_ids":["a"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_artifacts_carry_pred_and_gold_ids():
    """artifacts.pred_ids / gold_ids required (rag_qa also supports retrieval-side metric aggregation)."""
    task = RagQA()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert "pred_ids" in s.artifacts
        assert "gold_ids" in s.artifacts


def test_load_prediction_injects_contexts_into_doc_metadata():
    """Unit test: rag_qa.load_prediction puts row['contexts']/['retrieved_ids'] into doc.metadata."""
    task = RagQA()
    doc = Doc(id="x", input="q", target="t", metadata={"gold_doc_ids": ("a.txt",)})
    row = {
        "id": "x",
        "prediction": "the answer",
        "contexts": ["ctx1", "ctx2"],
        "retrieved_ids": ["a.txt", "b.txt"],
    }
    enriched, response = task.load_prediction(doc, row)
    assert enriched.metadata["contexts"] == ("ctx1", "ctx2")
    assert enriched.metadata["retrieved_ids"] == ("a.txt", "b.txt")
    assert enriched.metadata["gold_doc_ids"] == ("a.txt",)
    assert response.text == "the answer"
