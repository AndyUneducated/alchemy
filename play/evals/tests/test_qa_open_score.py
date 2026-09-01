"""qa_open task score path e2e (FakeJudgeLM zero network).

Demonstrate the core narrative of Phase 3 instruction:
  - exact_match=0 on paraphrase, judge_pointwise is still significantly higher than garbage (**core narrative**)
  - wrong_fact has rouge_l~0.9 (lexical misjudgment) but the judge gives a low score (**reverse narrative**)

FakeJudgeLM uses two rules to simulate judges:
  ① char-Jaccard: effective for perfect/paraphrase/garbage distinction
  ② const(score): used for wrong_fact - char-Jaccard cannot catch "wrong fact" ("1368" → "1378" almost all characters
     Same, jaccard ~0.94), which is the advantage space of the true LLM judge. e2e live is tested in test_qa_open_live.py
     Run with real ollama.

According to plan §6: Each new task relocks the runner invariant (n_matches / missing_pred_raises),
Even though sentiment / mt already has a test with the same name."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.models.base import LM
from evals.runner import evaluate_score
from evals.tasks.qa_open import QAOpen
from evals.tests.test_judge_core import FakeJudgeLM

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def _jaccard_fake() -> LM:
    """Press 'Reference answer: ' / 'Response: ' anchor to cut prompt, char-set Jaccard to give points."""

    def rule(prompt: str) -> str:
        ref = prompt.split("Reference answer: ")[1].split("\n")[0]
        resp = prompt.split("Response: ")[1].split("\n")[0]
        a, b = set(ref), set(resp)
        if not a or not b:
            return "1"
        j = len(a & b) / len(a | b)
        if j > 0.7:
            return "5"
        if j > 0.4:
            return "4"
        if j > 0.15:
            return "3"
        if j > 0.05:
            return "2"
        return "1"

    return FakeJudgeLM(outputs=rule)


def _const_fake(score: int) -> LM:
    """always-`score` fake: oracle stand-in for char-Jaccard failure scenarios such as wrong_fact."""
    return FakeJudgeLM(outputs=lambda _p: str(score))


def _score(pred_name: str, judge_lm: LM | None = None) -> dict[str, float]:
    task = QAOpen(judge_lm=judge_lm)
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 10
    return r.aggregated


# ---------- Upper and lower bounds sanity ----------

def test_perfect_judge_high_lexical_high():
    """perfect == gold → high scores for all members."""
    agg = _score("perfect", _jaccard_fake())
    assert agg["exact_match"] == 1.0
    assert agg["rouge_l"] >= 0.99
    # Jaccard=1 → 5 points ×10
    assert agg["judge_pointwise"] >= 4.5


def test_garbage_judge_low_lexical_low():
    """garbage is completely unrelated → lexical and judge are both low."""
    agg = _score("garbage", _jaccard_fake())
    assert agg["exact_match"] == 0.0
    assert agg["rouge_l"] < 0.30
    assert agg["judge_pointwise"] <= 2.0


# ---------- Core narrative: paraphrase paradox ----------

def test_paraphrase_judge_saves_meaning():
    """**Core narrative**: Paraphrase on exact_match=0, judge is still significantly higher than garbage.

    Comparison point: char-Jaccard fake judge gave paraphrase ~3.9 points, garbage only ~1.3 points;
    The judge can still correctly distinguish between "semantic pairs, literal changes" and "completely irrelevant" when lexical fails."""
    paraphrase_agg = _score("paraphrase", _jaccard_fake())
    assert paraphrase_agg["exact_match"] == 0.0
    assert paraphrase_agg["judge_pointwise"] >= 3.5

    garbage_agg = _score("garbage", _jaccard_fake())
    assert paraphrase_agg["judge_pointwise"] - garbage_agg["judge_pointwise"] > 1.5


# ---------- Reverse narrative: wrong_fact on lexical misjudgment / judge catch ----------

def test_wrong_fact_judge_low_lexical_could_be_high():
    """**Reverse narrative**: wrong_fact has almost all the same characters as gold ("1368"→"1378"), lexical gives moderate false positives.

    real LLM judge faces this story in the e2e live test; here, the const(1) oracle substitute is used to lock the "ideal"
    "judge should give low score" contract - char-Jaccard on wrong_fact j~0.94 will also give 5 points (similar to lexical
    Synchronous blindness), so Jaccard cannot be used to "act" the wrong_fact story."""
    agg = _score("wrong_fact", _const_fake(1))
    assert agg["rouge_l"] >= 0.5
    assert agg["exact_match"] == 0.0
    assert agg["judge_pointwise"] <= 2.0


# ---------- Framework invariants (re-locking for each task, plan § 6) ----------

def test_score_qa_open_n_matches_gold():
    """n == the number of rows in the data set (to prevent new task codepath from returning in advance / leaking samples)."""
    task = QAOpen()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 10


def test_qa_open_score_missing_pred_raises(tmp_path):
    """Missing doc_id Strict KeyError (same contract as sentiment/mt)."""
    task = QAOpen()
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"id":"qNONE","prediction":"x"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


# ---------- DECISIONS §X wave 4: judge_pointwise all sample None → aggregator None -----

def test_judge_pointwise_aggregator_returns_none_when_all_unparseable():
    """When the judge fails to parse all samples (each sample returns None) → the aggregator of the task
    Returns None "not measured", no longer silently giving 0.0 to pull the value down.

    The same shape as the judge_safety_score slice of phase 7 P2 safety task is empty → None;
    Dropping result.json as JSON null, the CLI renders `<n/a>` (phase 7 wave 2 is in place).

    Use the minimalist template `"rate: {response}"` so that all LM outputs seen by closure are
    "totally not parseable" - 10 samples × 1 fake LM × all None paths."""
    fake = FakeJudgeLM(outputs=lambda _p: "totally not parseable")
    task = QAOpen(judge_lm=fake, judge_template="rate: {response}")
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 10
    # task-specific lexical metrics do not move (orthogonal to judge)
    assert r.aggregated["exact_match"] == 1.0
    # judge_pointwise all None → aggregator None
    assert r.aggregated["judge_pointwise"] is None
