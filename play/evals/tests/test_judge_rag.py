"""metrics/judge_rag.py 单元层：5 个 RAG judge + 2 个 RAG 专用 parser.

零网络。FakeJudgeLM 复用 test_judge_core 的 stub（按 cursor 推进 / 规则函数双策略）。
专注于：
  - 2 个新 parser（parse_statement_list / parse_tp_fp_fn）的鲁棒解析
  - 5 个 judge_xxx 的 closure 形状 + 数值边界 + path B+C 数据契约（contexts 在 doc.metadata）
"""

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
    """构造带 retrieved contexts 的 Doc（path B+C：contexts 住 doc.metadata）."""
    return Doc(
        id="d0",
        input="q?",
        target=target,
        metadata={"contexts": tuple(contexts)},
    )


def _resp(text: str = "hyp") -> Response:
    return Response(doc_id="d0", text=text)


# ---------- parse_statement_list（4 条）------------------------------------

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


# ---------- parse_tp_fp_fn（4 条）------------------------------------------

def test_parse_tp_fp_fn_named_tokens():
    """显式 'TP=3 FP=1 FN=2' 形态."""
    assert parse_tp_fp_fn("TP=3 FP=1 FN=2") == (3, 1, 2)


def test_parse_tp_fp_fn_loose_punctuation():
    """带冒号 / 大小写混用 / 无等号也接."""
    assert parse_tp_fp_fn("Counts: TP: 5 fp 0 FN=4") == (5, 0, 4)


def test_parse_tp_fp_fn_fallback_first_three_ints():
    """没有命名 token 时回退取前 3 个整数."""
    assert parse_tp_fp_fn("answers: 2 1 3 (extra: 99)") == (2, 1, 3)


def test_parse_tp_fp_fn_invalid_raises():
    """不足 3 整数 → ValueError（强制 caller 知道 judge 输出失格）."""
    with pytest.raises(ValueError):
        parse_tp_fp_fn("just two: 1 2")


# ---------- judge_faithfulness（3 条）--------------------------------------

def test_faithfulness_full_supported():
    """response 拆 2 claim，两 claim 都被 'yes' 接 → faithfulness=1.0.

    FakeJudgeLM cursor 顺序：[extract result, nli yes, nli yes].
    """
    fake = FakeJudgeLM(outputs=[
        "- claim A\n- claim B",  # extract → 2 claims
        "yes",                     # NLI claim A
        "yes",                     # NLI claim B
    ])
    f = judge_faithfulness(fake)
    assert f(_doc_with_ctx(), _resp("answer")) == 1.0


def test_faithfulness_partial_half():
    """3 claim 里 2 supported / 1 unsupported → 2/3."""
    fake = FakeJudgeLM(outputs=[
        "- a\n- b\n- c",
        "yes", "no", "yes",
    ])
    f = judge_faithfulness(fake)
    val = f(_doc_with_ctx(), _resp("answer"))
    assert abs(val - 2 / 3) < 1e-9


def test_faithfulness_no_contexts_zero():
    """contexts 为空 → 直接 0.0（不烧 judge 调用）."""
    fake = FakeJudgeLM(outputs=["should not be reached"])
    f = judge_faithfulness(fake)
    doc_no_ctx = Doc(id="d0", input="q?", target="ref", metadata={"contexts": ()})
    assert f(doc_no_ctx, _resp("answer")) == 0.0


# ---------- judge_answer_correctness（3 条）--------------------------------

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
    """TP=0 → F1=0.0（precision+recall=0 短路）."""
    fake = FakeJudgeLM(outputs=["TP=0 FP=4 FN=2"])
    ac = judge_answer_correctness(fake)
    assert ac(_doc_with_ctx(target="gold"), _resp("wrong")) == 0.0


# ---------- judge_context_precision（2 条）--------------------------------

def test_context_precision_all_relevant():
    """2 contexts 都 yes → 1.0."""
    fake = FakeJudgeLM(outputs=["yes", "yes"])
    cp = judge_context_precision(fake)
    assert cp(_doc_with_ctx(contexts=("a", "b")), _resp()) == 1.0


def test_context_precision_half_relevant():
    """3 contexts: yes/no/yes → 2/3."""
    fake = FakeJudgeLM(outputs=["yes", "no", "yes"])
    cp = judge_context_precision(fake)
    val = cp(_doc_with_ctx(contexts=("a", "b", "c")), _resp())
    assert abs(val - 2 / 3) < 1e-9


# ---------- judge_context_recall（2 条）-----------------------------------

def test_context_recall_full():
    """target 拆 2 claim，两个都被 yes → 1.0."""
    fake = FakeJudgeLM(outputs=["- gold A\n- gold B", "yes", "yes"])
    cr = judge_context_recall(fake)
    assert cr(_doc_with_ctx(target="gold answer"), _resp()) == 1.0


def test_context_recall_partial():
    """3 claim 里 2 attributable → 2/3."""
    fake = FakeJudgeLM(outputs=["- a\n- b\n- c", "yes", "no", "yes"])
    cr = judge_context_recall(fake)
    val = cr(_doc_with_ctx(target="gold"), _resp())
    assert abs(val - 2 / 3) < 1e-9


# ---------- judge_answer_relevancy（2 条）---------------------------------

def test_answer_relevancy_pointwise_5():
    """单 prompt 1-5 评分，judge 输出 5 → 5.0."""
    fake = FakeJudgeLM(outputs=["5"])
    ar = judge_answer_relevancy(fake)
    assert ar(_doc_with_ctx(), _resp("on-topic answer")) == 5.0


def test_answer_relevancy_zero_on_empty_response():
    """response 空 → 直接 0.0（不烧 judge 调用）."""
    fake = FakeJudgeLM(outputs=["unused"])
    ar = judge_answer_relevancy(fake)
    assert ar(_doc_with_ctx(), Response(doc_id="d0", text="")) == 0.0
