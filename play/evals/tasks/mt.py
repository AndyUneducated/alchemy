"""Phase 2 vertical slice: Family 2 (Generation) EN → Medium Translation task.

6 indicators, covering two tiers of lexical + embedding (learned tier deferred):
  - exact_match exact string equality (hand calculation)
  - bleu sacrebleu corpus-BLEU (tokenize='zh')
  - chrf sacrebleu corpus-chrF (character n-gram, cross-language robust)
  - rouge_l rouge_score F-LCS (character-level tokenizer, otherwise Chinese will be stripped)
  - meteor nltk meteor_score (character level, requires wordnet but no synset in Chinese)
  - bertscore_f1 bert_score F1 (embedding tier represents alone)

Teaching story (design of 4 predictions):
  - perfect = gold target, all members ≈ 1.0 (BERTScore floating point imprecision is equal to 1)
  - literal literal translation (deliberately "overturning" the idiom), BLEU/chrF medium
  - Paraphrase retains meaning, replaces words, **BLEU plummets but BERTScore comes to the rescue** ← embedding tier core story
  - Garbage has nothing to do with text, and everyone has low scores (BERTScore still has ~0.4 mBERT baseline)

bertscore heavy dependency processing (lazy + lru_cache):
  `bert_score` is imported in `_bertscore_scorer()` to avoid commands such as list-tasks also paying ~700MB
  Download + ~3-5s torch startup; scorer instance module-level cache avoids repeated loading of models."""

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
    """Chinese character-level word segmentation: remove blanks and split by characters.

    BLEU and chrF are handled by sacrebleu's built-in zh-tokenizer; here for ROUGE/METEOR."""
    return [c for c in text if not c.isspace()]


class _ZhCharTokenizer:
    """rouge_score The default tokenizer strips non-ASCII characters - customization must be provided."""

    def tokenize(self, text: str) -> list[str]:
        return _zh_chars(text)


@lru_cache(maxsize=1)
def _rouge_scorer():
    """rouge_score instantiation has a non-zero cost and is cached once for the entire process."""
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_ZhCharTokenizer())


@lru_cache(maxsize=1)
def _bertscore_scorer():
    """BERTScorer loads ~700MB model + ~3-5s. The entire process is cached once.

    `lang="zh"` makes bert-score select `bert-base-chinese` (~400MB);
    `rescale_with_baseline=False` avoids relying on baseline files, gives identical strings to ~1.0 but not exact, etc.

    `TRANSFORMERS_VERBOSITY=error` + `disable_progress_bar()` suppress transformers loading
    `BertModel LOAD REPORT` UNEXPECTED warning (logging) + `Loading hit stderr
    weights:` tqdm progress bar (progress bar) two types of noise; the former is controlled by env var (HuggingFace
    Officially recommended log level control method), the latter is an independent mechanism (progress bar does not use logging).
    `setdefault` allows users to export explicitly without being overwritten; `disable_progress_bar` is the same as import
    Single point side effects. See DECISIONS §7.1.4 for details."""
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from bert_score import BERTScorer
    from transformers.utils import logging as _hf_logging

    _hf_logging.disable_progress_bar()
    return BERTScorer(lang="zh", rescale_with_baseline=False)


def _ensure_nltk_wordnet() -> None:
    """METEOR strongly relies on wordnet resources (even if Chinese does not have synset, this file is required).

    Use SSL fix to find out: Python on macOS often lacks root certificates."""
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
        # Go through certifi and try again
        import ssl

        import certifi

        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


@register_task("mt")
class MT(Task):
    """EN→ZH translation task, 6-metric demo."""

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
