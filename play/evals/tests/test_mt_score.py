"""mt task score mode: 6 metrics core assertions on 4 story-based predictions.

Compared with sentiment_clf's test_runner_score.py, there is one more story point -
**paraphrase** This prediction: BLEU plummets but BERTScore comes to the rescue, it is embedding tier
(vs lexical tier) executable proof. `test_paraphrase_bertscore_saves_meaning`
This assertion is green = the README is not bragging.

Numerical tolerance: BERTScore uses rescale_with_baseline=False, identical to give exactly 1.0;
But mBERT/bert-base-chinese cross-platform values have 1e-3 magnitude jitter - all BERTScore
Assertions use loose bands instead of strict thresholds. METEOR occasionally gives 0.998 on identical (NLTK
fragmentation penalty), also use >= 0.99 as a guarantee.

The first run triggers ~400MB bert-base-chinese download + ~3-5s model loading; cached thereafter."""

from __future__ import annotations

from pathlib import Path

from evals.runner import evaluate_score
from evals.tasks.mt import MT

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "mt" / "predictions"


def _score(name: str) -> dict[str, float]:
    task = MT()
    r = evaluate_score(task, PRED_DIR / f"{name}.jsonl")
    assert r.mode == "score"
    assert r.n == 30
    return r.aggregated


# ---------- perfect: upper bound sanity ----------

def test_perfect_all_metrics_top():
    """perfect.jsonl == gold target → all indicators close to 1.0."""
    agg = _score("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["bleu"] >= 0.99
    assert agg["chrf"] >= 0.99
    assert agg["rouge_l"] >= 0.99
    assert agg["meteor"] >= 0.99  # NLTK fragmentation penalty occasionally drops to 0.998
    assert agg["bertscore_f1"] >= 0.99


# ---------- garbage: lower bound sanity ----------

def test_garbage_lexical_low():
    """garbage.jsonl is irrelevant text → lexical indicators are close to 0."""
    agg = _score("garbage")
    assert agg["exact_match"] == 0.0
    assert agg["bleu"] < 0.05
    assert agg["chrf"] < 0.10
    assert agg["meteor"] < 0.15
    # rouge_l is slightly higher due to co-occurrence of common Chinese chars (是/的/在/了) but still <0.25
    assert agg["rouge_l"] < 0.25


def test_garbage_bertscore_above_baseline():
    """BERTScore F1 of garbage.jsonl ~0.5-0.65 (mBERT similarity floor for any zh text), not 0."""
    agg = _score("garbage")
    assert 0.40 <= agg["bertscore_f1"] <= 0.70, (
        f"BERTScore on unrelated zh text 应在 mBERT baseline 区间，got {agg['bertscore_f1']}"
    )


# ---------- paraphrase: embedding tier core story ----------

def test_paraphrase_lexical_drops():
    """paraphrase synonymous rewriting (word replacement) → BLEU/chrF plunge."""
    agg = _score("paraphrase")
    assert agg["exact_match"] == 0.0
    assert agg["bleu"] < 0.30
    assert agg["chrf"] < 0.30


def test_paraphrase_bertscore_saves_meaning():
    """**embedding tier core story**: BLEU plummets on paraphrase but BERTScore remains high.

    This green = "BERTScore can capture semantics that lexical cannot capture" is true on our demo data."""
    agg = _score("paraphrase")
    assert agg["bleu"] < 0.30, "前置：BLEU 必须真的暴跌才有故事"
    assert agg["bertscore_f1"] > 0.65, "BERTScore 应保住语义相似度"
    assert agg["bertscore_f1"] - agg["bleu"] > 0.40, (
        f"BERTScore-BLEU 差值应 > 0.40 才算 embedding tier 真正救场，"
        f"got bertscore={agg['bertscore_f1']:.3f} bleu={agg['bleu']:.3f}"
    )


# ---------- literal: medium baseline, loose to keep the bottom ----------

def test_literal_middle_ground():
    """literal.jsonl literal translation, idioms are overturned → all staff are at an average level."""
    agg = _score("literal")
    # exact_match is very low (only a few sentences with almost the same sentence structure), BLEU/chrF is between garbage and perfect
    assert agg["exact_match"] < 0.20
    assert 0.10 < agg["bleu"] < 0.50
    assert 0.10 < agg["chrf"] < 0.50
    # BERTScore is higher than garbage (the semantics are more or less the same) but less than perfect
    assert agg["bertscore_f1"] > 0.65
    assert agg["bertscore_f1"] < 0.99
