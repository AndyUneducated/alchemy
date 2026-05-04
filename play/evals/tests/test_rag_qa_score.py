"""rag_qa task score 路径 e2e（FakeJudgeLM 零网络）+ 4 份 stub 诊断叙事.

证明 phase 4 RAG QA 的两条核心叙事：
  - **paraphrase**：lexical (em / rouge_l) 失效，但 grounding (faithfulness /
    answer_correctness) 仍能保留——judge 救场（核心叙事）
  - **wrong_fact**：lexical 看起来还行（rouge_l 高，少量字符替换），grounding
    指标抓事实错（反向叙事）

判 LM 用 FakeJudgeLM with rule-based outputs（按 prompt 内容启发式给分）.
真 LLM e2e 在 test_rag_live.py via vdb-probe gate.
"""

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


# ---------- 上下界 sanity（lexical only，无 judge_lm）----------------------

def test_perfect_lexical_high():
    """perfect = gold answer → em=1.0 / rouge_l~1.0."""
    agg = _score("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["rouge_l"] >= 0.99


def test_garbage_lexical_low():
    """garbage 完全无关 → em=0 / rouge_l 低."""
    agg = _score("garbage")
    assert agg["exact_match"] == 0.0
    assert agg["rouge_l"] <= 0.1


# ---------- 核心叙事：paraphrase 上 lexical 跌 / grounding 救场 ------------

def _yes_judge_with_perfect_correctness() -> LM:
    """rule-based fake judge：
      - claim extract 类 prompt → 返回 1 个固定 claim（让 faithfulness/recall 走 ratio=1.0 通路）
      - NLI yes/no 类 prompt → 总是 'yes'（contexts 完美匹配的乐观 judge）
      - TP/FP/FN 类 prompt → TP=3 FP=0 FN=0（answer_correctness=1.0）
      - 1-5 评分类 prompt → '5'（answer_relevancy 满分）

    模拟"乐观 judge 无视字面变化"：paraphrase 上 grounding 应≈1.0.
    """

    def rule(prompt: str) -> str:
        p_low = prompt.lower()
        if "tp" in p_low and "fp" in p_low and "fn" in p_low:
            return "TP=3 FP=0 FN=0"
        if "decompose" in p_low or "atomic" in p_low or "statement" in p_low and "context" not in p_low:
            return "- claim X"
        if "1-5" in p_low or "score (1-5)" in p_low:
            return "5"
        # 默认 yes/no NLI / context relevance / faithfulness
        return "yes"

    return FakeJudgeLM(outputs=rule)


def _strict_judge() -> LM:
    """rule-based fake judge：用"prediction == 1 个 statement, NLI 失败 → 0 分"模拟严格 judge."""

    def rule(prompt: str) -> str:
        p_low = prompt.lower()
        if "tp" in p_low and "fp" in p_low and "fn" in p_low:
            return "TP=0 FP=3 FN=2"  # 完全不对 → F1=0
        if "decompose" in p_low or "atomic" in p_low or "statement" in p_low and "context" not in p_low:
            return "- claim X\n- claim Y"
        if "1-5" in p_low or "score (1-5)" in p_low:
            return "1"
        return "no"  # 严格 NLI

    return FakeJudgeLM(outputs=rule)


def test_paraphrase_lexical_drops_grounding_holds():
    """**核心叙事**：paraphrase 上 em=0、rouge_l 中等，但 lenient judge 给 grounding 满分.

    lenient judge 看到"语义对的 NLI"会答 yes——证明 grounding metric 在 lexical 失效时仍区分得开.
    """
    agg_lex = _score("paraphrase")  # 仅 lexical
    assert agg_lex["exact_match"] == 0.0
    assert agg_lex["rouge_l"] < 0.7  # 词全换光，char-level rouge 偏低
    # 加 lenient judge 后 grounding 维度满分
    agg_judge = _score("paraphrase", _yes_judge_with_perfect_correctness())
    assert agg_judge["faithfulness"] == 1.0
    assert agg_judge["answer_correctness"] == 1.0
    assert agg_judge["context_precision"] == 1.0
    assert agg_judge["context_recall"] == 1.0
    assert agg_judge["answer_relevancy"] == 5.0


# ---------- 反向叙事：wrong_fact 上 lexical 误判 / grounding 抓住 -----------

def test_wrong_fact_lexical_high_grounding_low():
    """**反向叙事**：wrong_fact 上 rouge_l 偏高（少量字符替换），但 strict judge 给低 grounding.

    关键对照：strict judge 模拟"识别出事实错"——这是 lexical 抓不到的盲区，
    real LLM judge 在 e2e live 测试里直面.
    """
    agg_lex = _score("wrong_fact")  # 仅 lexical
    assert agg_lex["exact_match"] == 0.0
    assert agg_lex["rouge_l"] >= 0.7  # 仅替换数字 → rouge 还高

    agg_judge = _score("wrong_fact", _strict_judge())
    assert agg_judge["faithfulness"] == 0.0  # NLI 全 'no'
    assert agg_judge["answer_correctness"] == 0.0  # F1 = 0
    # grounding 大幅低于 lexical，说明 judge 抓到了 lexical 漏判的事实错
    assert agg_judge["faithfulness"] < agg_lex["rouge_l"] - 0.5


# ---------- 框架不变量（每 task 重锁）-------------------------------------

def test_n_matches_gold():
    """n == 数据集行数."""
    task = RagQA()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 8


def test_score_missing_pred_raises(tmp_path):
    """缺 doc_id 严格 KeyError（与 sentiment / mt / qa_open 同 contract）."""
    task = RagQA()
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        '{"id":"qNONE","prediction":"x","contexts":["c"],"retrieved_ids":["a"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


def test_artifacts_carry_pred_and_gold_ids():
    """artifacts.pred_ids / gold_ids 必填（rag_qa 也支持 retrieval-side 指标聚合）."""
    task = RagQA()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    for s in r.per_sample:
        assert "pred_ids" in s.artifacts
        assert "gold_ids" in s.artifacts


def test_load_prediction_injects_contexts_into_doc_metadata():
    """单元测试：rag_qa.load_prediction 把 row['contexts']/['retrieved_ids'] 进 doc.metadata."""
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
