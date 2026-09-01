"""mmlu_slice unit + e2e score tests.

Two layers:
  ① **Unit**: parse_mcq_letter on handcrafted inputs (letter only / 'Answer: X' echo /
     markdown wrapper / distractor sentences)
  ② **E2E**: MmluSlice + evaluate_score on 3 stub fixtures (perfect / all_wrong /
     half_correct); assert accuracy and by-subject breakdown direction.

Per plan §6 "re-lock runner invariants per new task": n_matches_gold + missing_pred_raises both covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.mmlu_slice import MmluSlice, parse_mcq_letter

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "mmlu_slice" / "predictions"


# ============================================================
# parse_mcq_letter unit
# ============================================================

def test_parse_letter_only():
    """The model outputs only one letter - ideal situation."""
    assert parse_mcq_letter("A") == "A"
    assert parse_mcq_letter("B") == "B"
    assert parse_mcq_letter("C") == "C"
    assert parse_mcq_letter("D") == "D"


def test_parse_letter_with_punctuation():
    """Decorations such as periods/parentheses/backticks after letters - peel them off."""
    assert parse_mcq_letter("A.") == "A"
    assert parse_mcq_letter("B)") == "B"
    assert parse_mcq_letter("`C`") == "C"
    assert parse_mcq_letter("**D**") == "D"


def test_parse_lowercase_normalized():
    """Case-insensitive — \"a\" → \"A\"."""
    assert parse_mcq_letter("a") == "A"
    assert parse_mcq_letter("b.") == "B"


def test_parse_first_line_only():
    """Multi-line output: first non-empty line — ignore trailing explanation paragraph."""
    assert parse_mcq_letter("A\nbecause that's the answer") == "A"
    assert parse_mcq_letter("\n\nC\nrationale...") == "C"


def test_parse_answer_echo():
    """Echo templates like \"Answer: X\" / \"The answer is X\"."""
    assert parse_mcq_letter("Answer: A") == "A"
    assert parse_mcq_letter("answer: B") == "B"
    assert parse_mcq_letter("The answer is C.") == "C"
    assert parse_mcq_letter("The correct answer is D") == "D"


def test_parse_letter_inline_search():
    """When the letter head/echo is not found in the full text, fallback: Search for the first isolated A/B/C/D (non-letters before and after)."""
    # \"option (A)\" — A is preceded and followed by non-letters, which can be found
    assert parse_mcq_letter("the best option is (A)") == "A"


def test_parse_isolated_letter_protected_from_word_match():
    """Isolated letter detection must not match A inside \"Anatomy\"."""
    # There is no isolated A/B/C/D here: the A in Anatomy is followed by the letter
    assert parse_mcq_letter("Anatomy is the study of structures") is None


def test_parse_returns_none_on_empty_or_no_letter():
    """Empty string / no letters → None."""
    assert parse_mcq_letter("") is None
    assert parse_mcq_letter("   ") is None
    assert parse_mcq_letter("I don't know") is None
    assert parse_mcq_letter("123") is None


def test_parse_only_accepts_abcd():
    """E/F/Z etc. should not be recognized."""
    assert parse_mcq_letter("E") is None
    assert parse_mcq_letter("Z.") is None


# ============================================================
# evaluate_score e2e on 3 stub fixtures
# ============================================================

def _agg(pred_name: str) -> dict:
    task = MmluSlice()
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 96
    return r.aggregated


def test_perfect_e2e_accuracy_one():
    """perfect predictions = gold target → accuracy = 1.0; by_subject is also all 1."""
    agg = _agg("perfect")
    assert agg["accuracy"] == 1.0
    by_subj = agg["accuracy_by_subject"]
    assert isinstance(by_subj, dict)
    assert len(by_subj) == 6  # 6 subjects
    assert all(v == 1.0 for v in by_subj.values())


def test_all_wrong_e2e_accuracy_zero():
    """all predictions are next-letter-cycled → accuracy = 0.0."""
    agg = _agg("all_wrong")
    assert agg["accuracy"] == 0.0
    assert all(v == 0.0 for v in agg["accuracy_by_subject"].values())


def test_half_correct_e2e_around_half():
    """Even idx is correct / odd idx is wrong → total accuracy is close to 0.5 (96 rows of even numbers are exactly half correct)."""
    agg = _agg("half_correct")
    # Row 96: Even index 48 True + Odd index 48 False → 48/96 = 0.5
    assert agg["accuracy"] == 0.5
    # 16 rows per subject are also 8 each for even/odd → 0.5 for each subject
    by_subj = agg["accuracy_by_subject"]
    assert all(v == 0.5 for v in by_subj.values())


def test_by_subject_lists_all_six_subjects():
    """All 6 subjects must appear in by_subject - stable schema so that cross-run report headers do not drift."""
    agg = _agg("perfect")
    expected = {
        "abstract_algebra",
        "college_computer_science",
        "clinical_knowledge",
        "high_school_world_history",
        "philosophy",
        "econometrics",
    }
    assert set(agg["accuracy_by_subject"].keys()) == expected


def test_higher_is_better_only_scalar_listed():
    """nested dict subgroup (accuracy_by_subject) does not enter higher_is_better - same convention as nudge_fire_rate."""
    hib = MmluSlice().higher_is_better()
    assert hib == {"accuracy": True}


# ============================================================
# Framework invariants (plan §6: relocking per task)
# ============================================================

def test_score_n_matches_gold():
    """n == the number of rows in the data set (to prevent the task's own codepath from returning early/leaking samples)."""
    task = MmluSlice()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 96


def test_score_missing_pred_raises(tmp_path):
    """Missing doc_id strict KeyError."""
    task = MmluSlice()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"abstract_algebra_NONE","prediction":"A"}\n', encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_task_registered_under_correct_name():
    """`@register_task(\"mmlu_slice\")` side effect: CLI `--task mmlu_slice` resolves to this class."""
    from evals.registry import get_task
    assert isinstance(get_task("mmlu_slice"), MmluSlice)


def test_doc_to_text_renders_four_choices():
    """The prompt template expands all options A/B/C/D + all question stems enter prompt."""
    task = MmluSlice()
    docs = list(task.docs())
    text = task.doc_to_text(docs[0])
    assert "A. " in text and "B. " in text and "C. " in text and "D. " in text
    assert docs[0].input in text
    assert text.rstrip().endswith("Answer:")
