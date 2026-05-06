"""族 3 LLM-as-judge 核心范式（pointwise / pairwise / g_eval / self_consistency）.

按 README 指导原则 #3 触发新建：跨 task 复用（qa_open / 未来 summarization / writing 都会用）
+ 无成熟库可调（RAGAS 是 RAG 专用、deepeval 与本项目 task-decoupled 不兼容）.

phase 4 起拆为 `judge_core.py`（本文件）+ `judge_rag.py`（RAG 接地维度），
理由：核心范式是"评分方法学"，RAG 维度是"评分对象"，两层正交，避免单文件膨胀.

四个 judge 的主舞台分配（详见 plan §六）：
  - judge_pointwise   task 层主舞台（任务上 lexical vs judge 分歧叙事）
  - judge_pairwise    本文件主舞台（位置偏置 / swap 去偏）
  - g_eval            本文件主舞台（多维 form-filling / 多采样替代 logprob）
  - self_consistency  本文件主舞台（majority vote + tiebreak）

设计 highlights：
  - **closure 工厂模式**：`judge_pointwise(lm, ...)` 返回 `(doc, resp) -> float` 闭包，
    便于 self_consistency 这种 wrapper 套用，也便于 task.process_results 复用同一份 callable.
  - **不依赖 logprob**：Ollama /api/chat 不返回 logprobs；G-Eval 用 n-sample 多次采样
    估计离散分布，等价于"logprob 加权 mean"的非 logprob 通路。OpenAI 适配上线时可加
    `g_eval_logprob` 二级实现，不改默认.
  - **swap 去偏**：pairwise 默认 `swap=True`——A/B 与 B/A 双跑，不一致计 tie，
    把"位置偏置"当作 noise 而非信号.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Literal, Sequence

from ..api import Doc, Request, Response
from ..models.base import LM

PairwiseVerdict = Literal["a", "b", "tie"]


# ---------- closure-internal LM 调用 recorder（DECISIONS §7.3 wave 3）----------

class _JudgeRecorder:
    """closure 内部用：拦 lm.generate_until 调用，副本 append 到 responses 列表.

    DECISIONS §7.3：让 judge closure 自报数，runner 通过 task.collect_judge_responses
    拉取，挂 aggregated["efficiency"]["judge"]——评估工具 call class（双路径都挂）.

    所有 judge factory（judge_pointwise / judge_pairwise / g_eval +
    metrics/judge_rag.py 的 5 个 RAG factory）内部把原来直接 `judge_lm.generate_until(...)`
    的调用改走 recorder，runner 端从 closure._recorder 拉响应列表.
    """

    def __init__(self, lm: LM):
        self.lm = lm
        self.model_label = lm.name
        self.responses: list[Response] = []

    def call(self, requests: list[Request]) -> list[Response]:
        out = self.lm.generate_until(requests)
        self.responses.extend(out)
        return out

    def reset(self) -> None:
        """testing helper：复用 closure 时手动清状态."""
        self.responses = []


# ---------- 默认 prompt 模板 ----------

DEFAULT_POINTWISE_TEMPLATE = (
    "Rate the response on a scale of 1-5 (1=poor, 5=excellent).\n"
    "Question: {input}\n"
    "Reference answer: {reference}\n"
    "Response: {response}\n"
    "Score (1-5):"
)

DEFAULT_PAIRWISE_TEMPLATE = (
    "Compare two responses to the question. Choose A, B, or tie.\n"
    "Question: {input}\n"
    "Reference: {reference}\n"
    "Response A: {response_a}\n"
    "Response B: {response_b}\n"
    "Better response (A/B/tie):"
)

DEFAULT_G_EVAL_TEMPLATE = (
    "Rate the response on the dimension '{dimension}' from 1-5.\n"
    "Question: {input}\n"
    "Reference: {reference}\n"
    "Response: {response}\n"
    "Score (1-5):"
)


# ---------- parse helpers ----------

def parse_pointwise_score(text: str, *, scale: tuple[int, int] = (1, 5)) -> int:
    """从 judge 输出文本提取 int score.

    解析策略（鲁棒优先）：
      ① 找 text 中所有整数（含负数）
      ② 若有任意 int 落在 [lo, hi] 范围内 → 返回首个
      ③ 否则把第一个 int clamp 到 [lo, hi] 返回
      ④ 一个 int 都没有 → ValueError

    示例：
      "Score: 4/5" → 找到 [4, 5]，4 在 [1,5] → 4
      "Score: 7/5" → 找到 [7, 5]，5 在 [1,5] → 5（优先 in-range）
      "0"          → 找到 [0]，无 in-range → clamp(0)=1
      "999"        → 找到 [999]，无 in-range → clamp(999)=5
      "totally not a score" → 无 int → ValueError
    """
    lo, hi = scale
    ints = [int(m) for m in re.findall(r"-?\d+", text or "")]
    if not ints:
        raise ValueError(f"could not parse score from {text!r}")
    in_range = [n for n in ints if lo <= n <= hi]
    if in_range:
        return in_range[0]
    return max(lo, min(hi, ints[0]))


def parse_pairwise_verdict(text: str) -> PairwiseVerdict:
    """从 judge 输出提取 A/B/tie verdict（大小写不敏感）.

    优先级：
      ① 显式 "tie" / "equal" / "draw" / "neither" → tie
      ② 优先匹配 \\b[Aa]\\b → "a"，\\b[Bb]\\b → "b"
      ③ 若 'A' 与 'B' 都出现，取首个出现的
      ④ 完全无信号 → tie（保守）
    """
    s = (text or "").strip().lower()
    if not s:
        return "tie"
    # tie / equal / draw / neither
    if re.search(r"\b(tie|equal|draw|neither|both|same)\b", s):
        return "tie"
    # find standalone A or B token
    m = re.search(r"\b([ab])\b", s)
    if m:
        return m.group(1)  # type: ignore[return-value]
    # fallback: any A/B mention, first one wins
    for ch in s:
        if ch == "a":
            return "a"
        if ch == "b":
            return "b"
    return "tie"


# ---------- judge_pointwise（closure factory）----------

def judge_pointwise(
    judge_lm: LM,
    *,
    prompt_template: str = DEFAULT_POINTWISE_TEMPLATE,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], float | None]:
    """生成一个 (doc, response) -> float score | None 的闭包.

    模板字段：`{input}` / `{reference}` / `{response}`. 缺失字段会被 .format 忽略
    （如果模板没引用），所以测试可以用极简模板 `"rate: {response}"`.

    DECISIONS §7.3：closure 持有 `_recorder` 属性供 task.collect_judge_responses 拉取
    judge LM 调用记录，runner 挂 `aggregated["efficiency"]["judge"]`.

    DECISIONS §X (wave 4)：parser 抛 ValueError（LM 输出无 int 可解析）→ 闭包返 None；
    与 phase 7 wave 2 P2 立的"None vs 0 语义分离"原则一致——1-5 scale 0 越界，
    None 显式表"未测得"，下游 aggregator 过滤后空集→None，与 safety.judge_safety_score 同形.
    """
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> float | None:
        prompt = prompt_template.format(
            input=doc.input,
            reference=doc.target,
            response=response.text or "",
        )
        req = Request(
            doc_id=doc.id, prompt=prompt,
            request_type="generate_until", max_tokens=max_tokens,
        )
        [resp] = rec.call([req])
        try:
            return float(parse_pointwise_score(resp.text or "", scale=scale))
        except ValueError:
            return None

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- judge_pairwise + pairwise_winrate ----------

def judge_pairwise(
    judge_lm: LM,
    *,
    prompt_template: str = DEFAULT_PAIRWISE_TEMPLATE,
    swap: bool = True,
    max_tokens: int = 16,
) -> Callable[[Doc, Response, Response], PairwiseVerdict]:
    """返回一个 (doc, resp_a, resp_b) -> "a"/"b"/"tie" 的闭包.

    `swap=True`（默认）：双跑 A/B 与 B/A，把翻译回原序后两次结果一致才计胜负，
    否则计 tie——这是去除"位置偏置"的标准做法（Zheng et al. 2023, MT-Bench）.

    DECISIONS §7.3：closure 持有 `_recorder` 属性供 task.collect_judge_responses 拉取.
    """
    rec = _JudgeRecorder(judge_lm)

    def _ask(doc: Doc, a: Response, b: Response) -> PairwiseVerdict:
        prompt = prompt_template.format(
            input=doc.input,
            reference=doc.target,
            response_a=a.text or "",
            response_b=b.text or "",
        )
        req = Request(
            doc_id=doc.id, prompt=prompt,
            request_type="generate_until", max_tokens=max_tokens,
        )
        [r] = rec.call([req])
        return parse_pairwise_verdict(r.text or "")

    def _verdict(doc: Doc, resp_a: Response, resp_b: Response) -> PairwiseVerdict:
        v1 = _ask(doc, resp_a, resp_b)
        if not swap:
            return v1
        v2_raw = _ask(doc, resp_b, resp_a)
        # translate v2 back to original ordering: in swapped call,
        # "a" means resp_b wins → original "b"; "b" means resp_a wins → original "a"
        v2 = {"a": "b", "b": "a", "tie": "tie"}[v2_raw]
        if v1 == v2:
            return v1
        return "tie"

    _verdict._recorder = rec  # type: ignore[attr-defined]
    return _verdict


def pairwise_winrate(
    judge_lm: LM,
    pairs: Sequence[tuple[Doc, Response, Response]],
    *,
    prompt_template: str = DEFAULT_PAIRWISE_TEMPLATE,
    swap: bool = True,
) -> dict[str, float]:
    """聚合多对样本的 pairwise verdict → {a, b, tie} 比例.

    cross-task utility：score-pairwise CLI（phase 3.5）会直接调.
    """
    verdict_fn = judge_pairwise(judge_lm, prompt_template=prompt_template, swap=swap)
    counts = {"a": 0, "b": 0, "tie": 0}
    for doc, ra, rb in pairs:
        counts[verdict_fn(doc, ra, rb)] += 1
    n = sum(counts.values())
    if n == 0:
        return {"a": 0.0, "b": 0.0, "tie": 0.0}
    return {k: v / n for k, v in counts.items()}


# ---------- g_eval ----------

def g_eval(
    judge_lm: LM,
    *,
    dimensions: Sequence[str],
    prompt_template: str = DEFAULT_G_EVAL_TEMPLATE,
    n_samples: int = 5,
    scale: tuple[int, int] = (1, 5),
    max_tokens: int = 16,
) -> Callable[[Doc, Response], dict[str, float | None]]:
    """返回一个 (doc, response) -> {dim: score | None} 的闭包.

    每维 `n_samples` 次采样 + mean——替代 logprob 加权的离散分布估计（OpenAI 没 logprobs
    的本地 ollama / 兼容路径）.

    模板字段：`{input}` / `{reference}` / `{response}` / `{dimension}`.

    DECISIONS §7.3：closure 持有 `_recorder` 属性供 task.collect_judge_responses 拉取.

    DECISIONS §X (wave 4)：单次 sample parser 失败 → 跳过该 sample；该维 valid sample 全空
    → 维返 None（"未测得"占位，与 phase 7 P2 同形）；部分失败按 valid mean.
    """
    rec = _JudgeRecorder(judge_lm)

    def _score(doc: Doc, response: Response) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for dim in dimensions:
            scores: list[int] = []
            for _ in range(n_samples):
                prompt = prompt_template.format(
                    input=doc.input,
                    reference=doc.target,
                    response=response.text or "",
                    dimension=dim,
                )
                req = Request(
                    doc_id=doc.id, prompt=prompt,
                    request_type="generate_until", max_tokens=max_tokens,
                )
                [resp] = rec.call([req])
                try:
                    scores.append(parse_pointwise_score(resp.text or "", scale=scale))
                except ValueError:
                    continue
            out[dim] = sum(scores) / len(scores) if scores else None
        return out

    _score._recorder = rec  # type: ignore[attr-defined]
    return _score


# ---------- self_consistency wrapper ----------

def self_consistency(
    base_judge: Callable[..., Any],
    *,
    n_samples: int = 5,
) -> Callable[..., Any]:
    """把任意 base_judge 包成"采样 N 次取众数"的版本.

    适用于：
      - judge_pointwise 闭包（int score 的众数）
      - 任意返回 hashable 的 callable（pairwise verdict / 类别 label / ...）

    平票取首个出现的众数（first-seen tiebreak）——deterministic，避免字典序 / 随机.

    DECISIONS §7.3：透传 base 的 `_recorder` 属性——base 内部多次调用都会被同一 recorder
    收集，wrapper 不需自己另开 recorder.
    """

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        results = [base_judge(*args, **kwargs) for _ in range(n_samples)]
        counts = Counter(results)
        top = max(counts.values())
        for r in results:
            if counts[r] == top:
                return r
        return results[0]  # unreachable; mypy/pylint friendly

    if hasattr(base_judge, "_recorder"):
        _wrapped._recorder = base_judge._recorder  # type: ignore[attr-defined]
    return _wrapped
