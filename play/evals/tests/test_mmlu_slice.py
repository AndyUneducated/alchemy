"""mmlu_slice 单元 + e2e score 测试.

两层：
  ① **单元**：parse_mcq_letter 在 handcrafted 输入（letter only / 'Answer: X' echo /
     markdown 包装 / 干扰句）上的合约
  ② **e2e**：MmluSlice + evaluate_score 跑 3 个 stub fixture（perfect / all_wrong /
     half_correct），断言 accuracy 与 by-subject breakdown 的方向.

按 plan §六 \"每个新 task 重锁 runner 不变量\"：n_matches_gold + missing_pred_raises 都补上.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import evaluate_score
from evals.tasks.mmlu_slice import MmluSlice, parse_mcq_letter

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "mmlu_slice" / "predictions"


# ============================================================
# parse_mcq_letter 单元
# ============================================================

def test_parse_letter_only():
    """模型只输出一个字母——理想情况."""
    assert parse_mcq_letter("A") == "A"
    assert parse_mcq_letter("B") == "B"
    assert parse_mcq_letter("C") == "C"
    assert parse_mcq_letter("D") == "D"


def test_parse_letter_with_punctuation():
    """字母后带句点 / 括号 / 反引号 等装饰——剥掉."""
    assert parse_mcq_letter("A.") == "A"
    assert parse_mcq_letter("B)") == "B"
    assert parse_mcq_letter("`C`") == "C"
    assert parse_mcq_letter("**D**") == "D"


def test_parse_lowercase_normalized():
    """大小写不敏感——\"a\" → \"A\"."""
    assert parse_mcq_letter("a") == "A"
    assert parse_mcq_letter("b.") == "B"


def test_parse_first_line_only():
    """多行输出取第一行非空——后续\"explanation\"段忽略."""
    assert parse_mcq_letter("A\nbecause that's the answer") == "A"
    assert parse_mcq_letter("\n\nC\nrationale...") == "C"


def test_parse_answer_echo():
    """\"Answer: X\" / \"The answer is X\" 等 echo 模板."""
    assert parse_mcq_letter("Answer: A") == "A"
    assert parse_mcq_letter("answer: B") == "B"
    assert parse_mcq_letter("The answer is C.") == "C"
    assert parse_mcq_letter("The correct answer is D") == "D"


def test_parse_letter_inline_search():
    """全文未找到 letter 头/echo 时 fallback：搜首个孤立 A/B/C/D（前后非字母）."""
    # \"option (A)\" — A 前后是非字母，能找到
    assert parse_mcq_letter("the best option is (A)") == "A"


def test_parse_isolated_letter_protected_from_word_match():
    """孤立 letter 检测要避免 \"Anatomy\" 里的 A 被误匹配."""
    # 这里没有任何孤立 A/B/C/D：Anatomy 里的 A 后面跟字母
    assert parse_mcq_letter("Anatomy is the study of structures") is None


def test_parse_returns_none_on_empty_or_no_letter():
    """空字符串 / 无字母 → None."""
    assert parse_mcq_letter("") is None
    assert parse_mcq_letter("   ") is None
    assert parse_mcq_letter("I don't know") is None
    assert parse_mcq_letter("123") is None


def test_parse_only_accepts_abcd():
    """E/F/Z 等不应被识别."""
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
    """perfect predictions = gold target → accuracy = 1.0；by_subject 也全 1."""
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
    """偶数 idx 对 / 奇数 idx 错 → 总 accuracy 接近 0.5（96 行偶数刚好对一半）."""
    agg = _agg("half_correct")
    # 96 行：偶数索引 48 对 + 奇数索引 48 错 → 48/96 = 0.5
    assert agg["accuracy"] == 0.5
    # 每 subject 16 行也是偶/奇各 8 → 各 subject 0.5
    by_subj = agg["accuracy_by_subject"]
    assert all(v == 0.5 for v in by_subj.values())


def test_by_subject_lists_all_six_subjects():
    """6 个 subject 都必须在 by_subject 里出现——稳定 schema 让跨 run 报告表头不漂移."""
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
    """nested dict 子组（accuracy_by_subject）不进 higher_is_better——与 nudge_fire_rate 同 convention."""
    hib = MmluSlice().higher_is_better()
    assert hib == {"accuracy": True}


# ============================================================
# 框架不变量（plan §六：每 task 重锁）
# ============================================================

def test_score_n_matches_gold():
    """n == 数据集行数（防 task 自身 codepath 提前 return / 漏样本）."""
    task = MmluSlice()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 96


def test_score_missing_pred_raises(tmp_path):
    """缺 doc_id 严格 KeyError."""
    task = MmluSlice()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"abstract_algebra_NONE","prediction":"A"}\n', encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_task_registered_under_correct_name():
    """`@register_task(\"mmlu_slice\")` 副作用：CLI `--task mmlu_slice` 能拿到本类."""
    from evals.registry import get_task
    assert isinstance(get_task("mmlu_slice"), MmluSlice)


def test_doc_to_text_renders_four_choices():
    """prompt 模板把 A/B/C/D 选项都展开 + 题干都进入 prompt."""
    task = MmluSlice()
    docs = list(task.docs())
    text = task.doc_to_text(docs[0])
    assert "A. " in text and "B. " in text and "C. " in text and "D. " in text
    assert docs[0].input in text
    assert text.rstrip().endswith("Answer:")
