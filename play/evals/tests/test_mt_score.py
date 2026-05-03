"""mt task score 模式：6 指标在 4 份故事化 predictions 上的核心断言.

对比 sentiment_clf 的 test_runner_score.py，这里多了一个故事点——
**paraphrase** 这份 prediction：BLEU 暴跌但 BERTScore 救场，是 embedding tier
（vs lexical tier）的可执行证明。`test_paraphrase_bertscore_saves_meaning`
这条断言绿 = README 没在吹牛。

数值容差：BERTScore 用 rescale_with_baseline=False，identical 给精确 1.0；
但 mBERT/bert-base-chinese 跨 platform 数值有 1e-3 量级抖动——所有 BERTScore
断言都用宽松 band 而非严格阈。METEOR 在 identical 上偶尔给 0.998（NLTK
fragmentation penalty 的微小 quirk），同样用 >= 0.99 兜底。

首次跑触发 ~400MB bert-base-chinese 下载 + ~3-5s 模型加载；之后缓存。
"""

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


# ---------- perfect：上界 sanity ----------

def test_perfect_all_metrics_top():
    """perfect.jsonl == gold target → 所有指标接近 1.0."""
    agg = _score("perfect")
    assert agg["exact_match"] == 1.0
    assert agg["bleu"] >= 0.99
    assert agg["chrf"] >= 0.99
    assert agg["rouge_l"] >= 0.99
    assert agg["meteor"] >= 0.99  # NLTK fragmentation penalty 偶尔扣到 0.998
    assert agg["bertscore_f1"] >= 0.99


# ---------- garbage：下界 sanity ----------

def test_garbage_lexical_low():
    """garbage.jsonl 是无关文本 → lexical 指标都接近 0."""
    agg = _score("garbage")
    assert agg["exact_match"] == 0.0
    assert agg["bleu"] < 0.05
    assert agg["chrf"] < 0.10
    assert agg["meteor"] < 0.15
    # rouge_l 因常见汉字（是/的/在/了）共现略高但仍 <0.25
    assert agg["rouge_l"] < 0.25


def test_garbage_bertscore_above_baseline():
    """garbage.jsonl 的 BERTScore F1 ~0.5-0.65（mBERT 任意 zh 文本的相似度地板），不是 0."""
    agg = _score("garbage")
    assert 0.40 <= agg["bertscore_f1"] <= 0.70, (
        f"BERTScore on unrelated zh text 应在 mBERT baseline 区间，got {agg['bertscore_f1']}"
    )


# ---------- paraphrase：embedding tier 核心故事 ----------

def test_paraphrase_lexical_drops():
    """paraphrase 同义改写（词换光）→ BLEU/chrF 暴跌."""
    agg = _score("paraphrase")
    assert agg["exact_match"] == 0.0
    assert agg["bleu"] < 0.30
    assert agg["chrf"] < 0.30


def test_paraphrase_bertscore_saves_meaning():
    """**embedding tier 核心故事**：paraphrase 上 BLEU 暴跌但 BERTScore 仍高.

    这条绿 = "BERTScore 能捕捉到 lexical 抓不住的语义" 在我们的 demo 数据上成立.
    """
    agg = _score("paraphrase")
    assert agg["bleu"] < 0.30, "前置：BLEU 必须真的暴跌才有故事"
    assert agg["bertscore_f1"] > 0.65, "BERTScore 应保住语义相似度"
    assert agg["bertscore_f1"] - agg["bleu"] > 0.40, (
        f"BERTScore-BLEU 差值应 > 0.40 才算 embedding tier 真正救场，"
        f"got bertscore={agg['bertscore_f1']:.3f} bleu={agg['bleu']:.3f}"
    )


# ---------- literal：中等 baseline，loose 守底 ----------

def test_literal_middle_ground():
    """literal.jsonl 逐字直译，成语处翻车 → 全员中等水平."""
    agg = _score("literal")
    # exact_match 极低（仅句式几乎相同的几条），BLEU/chrF 在 garbage 与 perfect 之间
    assert agg["exact_match"] < 0.20
    assert 0.10 < agg["bleu"] < 0.50
    assert 0.10 < agg["chrf"] < 0.50
    # BERTScore 比 garbage 高（语义多少对得上）但不到 perfect
    assert agg["bertscore_f1"] > 0.65
    assert agg["bertscore_f1"] < 0.99
