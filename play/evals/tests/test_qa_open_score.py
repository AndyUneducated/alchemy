"""qa_open task score 路径 e2e（FakeJudgeLM 零网络）.

证明 Phase 3 教学核心叙事：
  - paraphrase 上 exact_match=0，judge_pointwise 仍显著高于 garbage（**核心叙事**）
  - wrong_fact 上 rouge_l~0.9（lexical 误判）但 judge 给低分（**反向叙事**）

FakeJudgeLM 用两种规则模拟 judge：
  ① char-Jaccard：对 perfect / paraphrase / garbage 区分有效
  ② const(score)：用于 wrong_fact——char-Jaccard 抓不住"事实错"（"1368"→"1378"几乎全字符
     相同，jaccard ~0.94），这是真 LLM judge 的优势空间。e2e live 测在 test_qa_open_live.py
     用真 ollama 跑过.

按 plan §六：每个新 task 重锁 runner 不变量（n_matches / missing_pred_raises），
即便 sentiment / mt 已有同名测试.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.models.base import LM
from evals.runner import evaluate_score
from evals.tasks.qa_open import QAOpen
from evals.tests.test_judge_core import FakeJudgeLM

PRED_DIR = Path(__file__).resolve().parent.parent / "data" / "qa_open" / "predictions"


def _jaccard_fake() -> LM:
    """按 'Reference answer: ' / 'Response: ' anchor 切 prompt，char-set Jaccard 给分."""

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
    """always-`score` fake：用于 wrong_fact 等 char-Jaccard 失效场景的 oracle 替身."""
    return FakeJudgeLM(outputs=lambda _p: str(score))


def _score(pred_name: str, judge_lm: LM | None = None) -> dict[str, float]:
    task = QAOpen(judge_lm=judge_lm)
    r = evaluate_score(task, PRED_DIR / f"{pred_name}.jsonl")
    assert r.mode == "score"
    assert r.n == 10
    return r.aggregated


# ---------- 上下界 sanity ----------

def test_perfect_judge_high_lexical_high():
    """perfect == gold → 全员高分."""
    agg = _score("perfect", _jaccard_fake())
    assert agg["exact_match"] == 1.0
    assert agg["rouge_l"] >= 0.99
    # Jaccard=1 → 5 分 ×10
    assert agg["judge_pointwise"] >= 4.5


def test_garbage_judge_low_lexical_low():
    """garbage 完全无关 → lexical 与 judge 都低."""
    agg = _score("garbage", _jaccard_fake())
    assert agg["exact_match"] == 0.0
    assert agg["rouge_l"] < 0.30
    assert agg["judge_pointwise"] <= 2.0


# ---------- 核心叙事：paraphrase paradox ----------

def test_paraphrase_judge_saves_meaning():
    """**核心叙事**：paraphrase 上 exact_match=0，judge 仍显著高于 garbage.

    对照点：char-Jaccard fake judge 把 paraphrase 给到 ~3.9 分，garbage 只到 ~1.3 分；
    judge 在 lexical 失效时仍能正确区分"语义对、字面变"与"完全无关".
    """
    paraphrase_agg = _score("paraphrase", _jaccard_fake())
    assert paraphrase_agg["exact_match"] == 0.0
    assert paraphrase_agg["judge_pointwise"] >= 3.5

    garbage_agg = _score("garbage", _jaccard_fake())
    assert paraphrase_agg["judge_pointwise"] - garbage_agg["judge_pointwise"] > 1.5


# ---------- 反向叙事：wrong_fact 上 lexical 误判 / judge 抓住 ----------

def test_wrong_fact_judge_low_lexical_could_be_high():
    """**反向叙事**：wrong_fact 几乎全字符与 gold 相同（"1368"→"1378"），lexical 给中等假阳性.

    real LLM judge 在 e2e live 测试里直面这个故事；这里用 const(1) oracle 替身锁住"理想
    judge 应给低分"的契约——char-Jaccard 在 wrong_fact 上 j~0.94 也会给 5 分（与 lexical
    同步失明），所以不能用 Jaccard 来"演"wrong_fact 的故事.
    """
    agg = _score("wrong_fact", _const_fake(1))
    assert agg["rouge_l"] >= 0.5
    assert agg["exact_match"] == 0.0
    assert agg["judge_pointwise"] <= 2.0


# ---------- 框架不变量（每 task 重锁，plan §六）----------

def test_score_qa_open_n_matches_gold():
    """n == 数据集行数（防新 task 自身 codepath 提前 return / 漏样本）."""
    task = QAOpen()
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 10


def test_qa_open_score_missing_pred_raises(tmp_path):
    """缺 doc_id 严格 KeyError（与 sentiment / mt 同 contract）."""
    task = QAOpen()
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"id":"qNONE","prediction":"x"}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        evaluate_score(task, partial)


# ---------- DECISIONS §X wave 4：judge_pointwise 全 sample None → aggregator None -----

def test_judge_pointwise_aggregator_returns_none_when_all_unparseable():
    """判官全条 parse 失败时（每条 sample 都返 None）→ task 的 aggregator
    返 None"未测得"，不再 silently 给 0.0 把数值往下拉.

    与 phase 7 P2 安全 task 的 judge_safety_score 切片为空 → None 同形；
    落 result.json 为 JSON null，CLI 渲染 `<n/a>`（phase 7 wave 2 已就位）.

    采用极简模板 `"rate: {response}"` 让 closure 看到的所有 LM 输出都是
    "totally not parseable"——10 条 sample × 1 个 fake LM × 全 None 路径.
    """
    fake = FakeJudgeLM(outputs=lambda _p: "totally not parseable")
    task = QAOpen(judge_lm=fake, judge_template="rate: {response}")
    r = evaluate_score(task, PRED_DIR / "perfect.jsonl")
    assert r.n == 10
    # task-specific lexical metrics 不动（与 judge 正交）
    assert r.aggregated["exact_match"] == 1.0
    # judge_pointwise 全条 None → aggregator None
    assert r.aggregated["judge_pointwise"] is None
