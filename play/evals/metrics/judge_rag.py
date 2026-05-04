"""族 4 RAG 接地维度（5 个 judge_xxx 闭包工厂 + 2 个 RAG 专用 parser）.

为什么单独建 `judge_rag.py` 而不是塞进 `judge_core.py`：
  - judge_core 是"评分方法学"（pointwise / pairwise / g_eval / self_consistency）
  - judge_rag  是"评分对象 = RAG pipeline 各环节"（faithfulness / answer_correctness / ...）
  两层正交：判 LM 范式扩展到第 5 个不会拖累 judge_rag 的演化（CHANGELOG §4 决策）.

5 个维度对齐 RAGAS 主框架（faithfulness / answer_correctness / context_precision /
context_recall / answer_relevancy），但**不直接 import ragas**：
  - 依赖膨胀（langchain / openai / 数据科学全家桶）
  - 我们已有 LM ABC 与 closure 工厂模式，自实现 ~150 行就把同一通路跑通
  - 锁住 prompt 字面字符串（lm-eval 不变量），ragas 内部 prompt 是黑盒

数据契约（path B+C，见 CHANGELOG §4）：
  - `response.text` 装 LM-side 输出（answer 字符串）
  - `doc.metadata['contexts']` 装 retrieval 产物（list[str]，by rag_qa.process_docs 注入）
  - `doc.target` 装 gold 答案（answer_correctness / context_recall 用）

每个 judge_xxx 都是 closure 工厂：返回 `Callable[[Doc, Response], float]`，
与 judge_core.judge_pointwise 协议同形——self_consistency 也能照样套.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from ..api import Doc, Request, Response
from ..models.base import LM


# ---------- 默认 prompt 模板 -------------------------------------------------

DEFAULT_CLAIM_EXTRACT_TEMPLATE = (
    "Decompose the following text into atomic factual statements. "
    "Output each statement on a new line, prefixed with '- '.\n\n"
    "Text: {text}\n\n"
    "Statements:"
)

DEFAULT_NLI_TEMPLATE = (
    "Given the following context, can the statement be inferred? "
    "Answer 'yes' or 'no'.\n\n"
    "Context:\n{context}\n\n"
    "Statement: {statement}\n\n"
    "Answer:"
)

DEFAULT_TP_FP_FN_TEMPLATE = (
    "Compare the response to the reference answer. Count three quantities:\n"
    "  TP = facts in response that ALSO appear in reference\n"
    "  FP = facts in response that DO NOT appear in reference\n"
    "  FN = facts in reference that DO NOT appear in response\n\n"
    "Reference: {reference}\n"
    "Response: {response}\n\n"
    "Output exactly three integers in the form 'TP=<int> FP=<int> FN=<int>'.\n"
    "Counts:"
)

DEFAULT_CONTEXT_RELEVANCE_TEMPLATE = (
    "Given the question, is the following context useful for answering it? "
    "Answer 'yes' or 'no'.\n\n"
    "Question: {input}\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

DEFAULT_ANSWER_RELEVANCE_TEMPLATE = (
    "Rate how relevant the response is to the question on a scale of 1-5 "
    "(5=fully on-topic, 1=completely off-topic; ignore factual correctness).\n\n"
    "Question: {input}\n"
    "Response: {response}\n\n"
    "Score (1-5):"
)


# ---------- RAG 专用 parsers -------------------------------------------------

def parse_statement_list(text: str) -> list[str]:
    """从 LLM 输出抽 bulleted/numbered 列表项 → list[str].

    支持的前缀：`- foo` / `* foo` / `1. foo` / `1) foo` / 纯换行行也接.
    Trim 空白，过滤空行。这是 faithfulness / context_recall / answer_correctness
    三个 metric 共用的 decomposition 入口.
    """
    lines = (text or "").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # 剥常见 bullet 前缀
        s = re.sub(r"^[\-\*•]\s+", "", s)
        s = re.sub(r"^\d+[\.\)]\s+", "", s)
        s = s.strip()
        if s:
            out.append(s)
    return out


def parse_tp_fp_fn(text: str) -> tuple[int, int, int]:
    """从 judge 输出抽 (TP, FP, FN) 三整数.

    解析策略（鲁棒优先）：
      ① 先找显式形如 'TP=3' / 'FP: 1' / 'FN 2' 的命名 token
      ② 若三键都没匹配，回退取前 3 个非负整数

    缺任一键 → ValueError，强制 caller 知道 judge 输出失格.
    """
    s = text or ""
    found: dict[str, int] = {}
    for key in ("TP", "FP", "FN"):
        m = re.search(rf"\b{key}\s*[:=]?\s*(\d+)", s, re.IGNORECASE)
        if m:
            found[key] = int(m.group(1))
    if all(k in found for k in ("TP", "FP", "FN")):
        return found["TP"], found["FP"], found["FN"]

    # fallback: 前 3 个非负 int
    ints = [int(m) for m in re.findall(r"\d+", s)]
    if len(ints) >= 3:
        return ints[0], ints[1], ints[2]

    raise ValueError(f"could not parse TP/FP/FN from {text!r}")


def _parse_yes_no(text: str) -> int:
    """LLM 'yes/no' 输出 → 1/0；保守策略：歧义返回 0（NLI 失败默认 not-supported）."""
    s = (text or "").strip().lower()
    if not s:
        return 0
    # 优先看首词是否 'yes'/'no'
    first = re.split(r"\W+", s, maxsplit=1)[0]
    if first in {"yes", "y", "true", "supported", "entailed", "1"}:
        return 1
    if first in {"no", "n", "false", "unsupported", "0"}:
        return 0
    # 全文搜兜底
    if re.search(r"\byes\b", s):
        return 1
    return 0


def _ask(lm: LM, prompt: str, *, doc_id: str = "", max_tokens: int = 64) -> str:
    """单轮 LM 调用助手——把 LM ABC 的 batch in/out 压缩到单 prompt → text 形态."""
    req = Request(
        doc_id=doc_id, prompt=prompt,
        request_type="generate_until", max_tokens=max_tokens,
    )
    [resp] = lm.generate_until([req])
    return resp.text or ""


# ---------- judge_faithfulness ----------------------------------------------

def judge_faithfulness(
    judge_lm: LM,
    *,
    claim_template: str = DEFAULT_CLAIM_EXTRACT_TEMPLATE,
    nli_template: str = DEFAULT_NLI_TEMPLATE,
    max_tokens_extract: int = 256,
    max_tokens_nli: int = 16,
) -> Callable[[Doc, Response], float]:
    """两步 judge：
      ① 让 judge_lm 把 `response.text` 拆 atomic claims（parse_statement_list 解析）
      ② 逐 claim 让 judge_lm 对 `doc.metadata['contexts']`（拼成 single context block）
         判 yes/no（NLI 风格）。返回 #supported / #total.

    `contexts` 来自 `doc.metadata`，遵循 path B+C：检索器产物住在 doc 一侧.
    无 contexts / 无可解析 claim → 0.0（保守）.
    """

    def _score(doc: Doc, response: Response) -> float:
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not contexts:
            return 0.0
        text = (response.text or "").strip()
        if not text:
            return 0.0

        # Step 1: extract claims
        extract_prompt = claim_template.format(text=text)
        raw = _ask(judge_lm, extract_prompt, doc_id=doc.id, max_tokens=max_tokens_extract)
        claims = parse_statement_list(raw)
        if not claims:
            return 0.0

        # Step 2: NLI per claim against the joined contexts
        joined_ctx = "\n---\n".join(contexts)
        supported = 0
        for claim in claims:
            nli_prompt = nli_template.format(context=joined_ctx, statement=claim)
            verdict = _ask(judge_lm, nli_prompt, doc_id=doc.id, max_tokens=max_tokens_nli)
            supported += _parse_yes_no(verdict)
        return supported / len(claims)

    return _score


# ---------- judge_answer_correctness ----------------------------------------

def judge_answer_correctness(
    judge_lm: LM,
    *,
    template: str = DEFAULT_TP_FP_FN_TEMPLATE,
    max_tokens: int = 64,
) -> Callable[[Doc, Response], float]:
    """单步 judge：让 judge_lm 数 TP/FP/FN（事实级），返回 F1.

    不依赖 retrieval contexts——只比 response 与 doc.target，是 grounding 维度里
    最直接的 "endpoint quality" 指标。F1 的好处是 precision (1-FP/(TP+FP)) 与
    recall (1-FN/(TP+FN)) 双约束.

    无 target / response 全空 / TP+FP+FN=0 → 0.0（保守）.
    """

    def _score(doc: Doc, response: Response) -> float:
        target = doc.target or ""
        pred = (response.text or "").strip()
        if not target or not pred:
            return 0.0
        prompt = template.format(reference=target, response=pred)
        try:
            raw = _ask(judge_lm, prompt, doc_id=doc.id, max_tokens=max_tokens)
            tp, fp, fn = parse_tp_fp_fn(raw)
        except ValueError:
            return 0.0
        if tp + fp + fn == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return _score


# ---------- judge_context_precision -----------------------------------------

def judge_context_precision(
    judge_lm: LM,
    *,
    template: str = DEFAULT_CONTEXT_RELEVANCE_TEMPLATE,
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float]:
    """逐 context（按 rank 顺序）让 judge_lm 判"对回答 query 是否有用 yes/no"，
    返回相关 context 的比例（top-k 的 binary precision）.

    与 metrics/retrieval.precision_at_k 的差别：那个是按 doc_id 与 gold_ids 的
    set-membership 比；这个是 LLM 直接判"语义是否相关"——更贴近"上下文质量"
    而非"召回的 doc 在不在 gold 集合里".

    无 contexts → 0.0；prediction 不参与（这是 query 与 context 的关系）.
    """

    def _score(doc: Doc, response: Response) -> float:
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not contexts:
            return 0.0
        relevant = 0
        for ctx in contexts:
            prompt = template.format(input=doc.input, context=ctx)
            verdict = _ask(judge_lm, prompt, doc_id=doc.id, max_tokens=max_tokens)
            relevant += _parse_yes_no(verdict)
        return relevant / len(contexts)

    return _score


# ---------- judge_context_recall --------------------------------------------

def judge_context_recall(
    judge_lm: LM,
    *,
    claim_template: str = DEFAULT_CLAIM_EXTRACT_TEMPLATE,
    nli_template: str = DEFAULT_NLI_TEMPLATE,
    max_tokens_extract: int = 256,
    max_tokens_nli: int = 16,
) -> Callable[[Doc, Response], float]:
    """两步 judge：
      ① 把 `doc.target`（gold 答案）拆 atomic claims
      ② 逐 claim 判"能否从 `doc.metadata['contexts']` 推出"，返回比例.

    与 judge_faithfulness 对偶：
      - faithfulness：response 的 claim 在 contexts 里 →  "你答的我能在材料里看到"
      - context_recall：target  的 claim 在 contexts 里 → "标准答案里的事实材料覆盖了多少"

    无 target / 无可解析 claim → 0.0.
    """

    def _score(doc: Doc, response: Response) -> float:
        target = doc.target or ""
        contexts: Sequence[str] = doc.metadata.get("contexts", ())
        if not target or not contexts:
            return 0.0

        extract_prompt = claim_template.format(text=target)
        raw = _ask(judge_lm, extract_prompt, doc_id=doc.id, max_tokens=max_tokens_extract)
        claims = parse_statement_list(raw)
        if not claims:
            return 0.0

        joined_ctx = "\n---\n".join(contexts)
        attributable = 0
        for claim in claims:
            nli_prompt = nli_template.format(context=joined_ctx, statement=claim)
            verdict = _ask(judge_lm, nli_prompt, doc_id=doc.id, max_tokens=max_tokens_nli)
            attributable += _parse_yes_no(verdict)
        return attributable / len(claims)

    return _score


# ---------- judge_answer_relevancy ------------------------------------------

def judge_answer_relevancy(
    judge_lm: LM,
    *,
    template: str = DEFAULT_ANSWER_RELEVANCE_TEMPLATE,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float]:
    """单步 judge：rate 1-5 "回答有没有正面 address 问题"（不管对错）.

    与 judge_answer_correctness 互补：
      - correctness：事实层面对不对（看 target）
      - relevancy  ：是否在答这个问题（不看 target，看 input vs response 主题对齐）

    解析复用 `judge_core.parse_pointwise_score`（同源 1-5 形态）.
    无 response → 0.0.
    """
    from .judge_core import parse_pointwise_score  # 避免循环 import

    def _score(doc: Doc, response: Response) -> float:
        pred = (response.text or "").strip()
        if not pred:
            return 0.0
        prompt = template.format(input=doc.input, response=pred)
        raw = _ask(judge_lm, prompt, doc_id=doc.id, max_tokens=max_tokens)
        try:
            return float(parse_pointwise_score(raw, scale=scale))
        except ValueError:
            return 0.0

    return _score
