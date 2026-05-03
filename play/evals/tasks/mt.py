"""Phase 2 vertical slice：族 2（Generation）EN→中 翻译 task.

6 个指标，覆盖 lexical + embedding 两个 tier（learned tier deferred）：
  - exact_match    完全字符串相等              （手算）
  - bleu           sacrebleu corpus-BLEU       （tokenize='zh'）
  - chrf           sacrebleu corpus-chrF       （字符 n-gram，跨语言鲁棒）
  - rouge_l        rouge_score F-LCS           （字符级 tokenizer，否则中文被剥光）
  - meteor         nltk meteor_score           （字符级，需 wordnet 但中文无 synset）
  - bertscore_f1   bert_score F1               （embedding tier 单独代表）

教学故事（4 份 predictions 的设计）：
  - perfect      = gold target，全员 ≈ 1.0（BERTScore 浮点不精确等于 1）
  - literal      逐字直译（成语处刻意"翻车"），BLEU/chrF 中等
  - paraphrase   意思保留、词换光，**BLEU 暴跌但 BERTScore 救场** ← embedding tier 核心故事
  - garbage      完全无关文本，全员低分（BERTScore 仍有 ~0.4 mBERT baseline）

bertscore 重依赖处理（lazy + lru_cache）：
  `bert_score` 在 `_bertscore_scorer()` 内 import，避免 list-tasks 等命令也付 ~700MB
  下载 + ~3-5s torch 启动；scorer 实例 module-level 缓存避免重复加载模型。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Callable, ClassVar

from sacrebleu import corpus_bleu, corpus_chrf

from ..api import Doc, Response, SampleResult
from ..registry import register_task
from .base import Task

PROMPT_TEMPLATE = (
    "Translate the following English sentence into Chinese.\n"
    "English: {input}\n"
    "Chinese:"
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mt" / "gold.jsonl"


def _zh_chars(text: str) -> list[str]:
    """中文字符级分词：去空白后按字拆。

    BLEU 和 chrF 由 sacrebleu 内置 zh-tokenizer 处理；这里给 ROUGE/METEOR 用。
    """
    return [c for c in text if not c.isspace()]


class _ZhCharTokenizer:
    """rouge_score 默认 tokenizer 会 strip 非 ASCII 字符——必须提供自定义."""

    def tokenize(self, text: str) -> list[str]:
        return _zh_chars(text)


@lru_cache(maxsize=1)
def _rouge_scorer():
    """rouge_score 实例化有非零成本，整个 process 缓存一次."""
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_ZhCharTokenizer())


@lru_cache(maxsize=1)
def _bertscore_scorer():
    """BERTScorer 加载约 ~700MB 模型 + ~3-5s. 整个 process 缓存一次.

    `lang="zh"` 让 bert-score 选择 `bert-base-chinese` (~400MB)；
    `rescale_with_baseline=False` 避免依赖 baseline 文件，identical strings 给 ~1.0 但非精确等。
    """
    from bert_score import BERTScorer

    return BERTScorer(lang="zh", rescale_with_baseline=False)


def _ensure_nltk_wordnet() -> None:
    """METEOR 强依赖 wordnet 资源（即便中文无 synset 也要这个文件）。

    用 SSL fix 兜底：macOS 系统 Python 常缺根证书。
    """
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
        return
    except LookupError:
        pass
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        # 走 certifi 兜底再试一次
        import ssl

        import certifi

        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


@register_task("mt")
class MT(Task):
    """EN→中 翻译任务，6 指标 demo."""

    name: ClassVar[str] = "mt"
    output_type: ClassVar[str] = "generate_until"

    data_path: Path = DATA_PATH

    def docs(self) -> Iterable[Doc]:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Doc(id=row["id"], input=row["input"], target=row["target"])

    def doc_to_text(self, doc: Doc) -> str:
        return PROMPT_TEMPLATE.format(input=doc.input)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.target

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        pred = (response.text or "").strip()
        target = doc.target
        return SampleResult(
            doc_id=doc.id,
            prediction=pred,
            target=target,
            metrics={"em": float(pred == target)},
        )

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        return {
            "exact_match": _exact_match,
            "bleu": _bleu,
            "chrf": _chrf,
            "rouge_l": _rouge_l,
            "meteor": _meteor,
            "bertscore_f1": _bertscore_f1,
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            "exact_match": True,
            "bleu": True,
            "chrf": True,
            "rouge_l": True,
            "meteor": True,
            "bertscore_f1": True,
        }


def _exact_match(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    return sum(s.metrics["em"] for s in srs) / len(srs)


def _bleu(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    hyps = [s.prediction for s in srs]
    refs = [s.target for s in srs]
    return corpus_bleu(hyps, [refs], tokenize="zh").score / 100.0


def _chrf(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    hyps = [s.prediction for s in srs]
    refs = [s.target for s in srs]
    return corpus_chrf(hyps, [refs]).score / 100.0


def _rouge_l(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    scorer = _rouge_scorer()
    scores = [scorer.score(s.target, s.prediction)["rougeL"].fmeasure for s in srs]
    return sum(scores) / len(scores)


def _meteor(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    _ensure_nltk_wordnet()
    from nltk.translate.meteor_score import meteor_score

    scores = [meteor_score([_zh_chars(s.target)], _zh_chars(s.prediction)) for s in srs]
    return sum(scores) / len(scores)


def _bertscore_f1(srs: list[SampleResult]) -> float:
    if not srs:
        return 0.0
    scorer = _bertscore_scorer()
    cands = [s.prediction for s in srs]
    refs = [s.target for s in srs]
    _, _, F = scorer.score(cands, refs)
    return float(F.mean().item())
