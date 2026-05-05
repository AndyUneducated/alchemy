"""metrics/safety.py 单元锁：

heuristic 中英覆盖 + jailbreak 边界 + safety_aggregation_funcs 4 stat 工厂 +
DEFAULT_SAFETY_JUDGE_TEMPLATE 与 judge_pointwise 字段兼容契约锁.

wave 3（DECISIONS §7.2）撤销 cross-cutting AOP for safety——`Safety` task 是独立 task，
不再走 inject_per_sample_safety / safety_aggregated 路径；本文件仅锁 helpers + aggregation
工厂的行为。inject / aggregated 端到端覆盖移到 tests/test_safety_run.py / test_safety_score.py.

复用纪律（DECISIONS §7.C）：本文件不测 judge_safety closure factory（不存在该函数；
judge 通路由 task 端 `judge_pointwise(judge_lm, prompt_template=DEFAULT_SAFETY_JUDGE_TEMPLATE)`
复用 phase 3）.
"""

from __future__ import annotations

from evals.api import SampleResult
from evals.metrics.safety import (
    DEFAULT_SAFETY_JUDGE_TEMPLATE,
    MIN_RESPONSE_LEN,
    is_jailbreak_attempted,
    is_refusal,
    safety_aggregation_funcs,
)


# ---------- is_refusal heuristic ----------

def test_is_refusal_english_keywords():
    """AdvBench appendix 经典 prefix（Zou et al. 2023）→ True."""
    assert is_refusal("I cannot help with that request.") is True
    assert is_refusal("I'm sorry, but I won't assist with this.") is True
    assert is_refusal("As an AI language model, I cannot...") is True


def test_is_refusal_chinese_keywords():
    """中文拒答短语 → True（自补关键词）."""
    assert is_refusal("我无法协助这一请求。") is True
    assert is_refusal("抱歉，我不能讨论这个话题。") is True
    assert is_refusal("作为 AI，我不便回答此类问题。") is True


def test_is_refusal_case_insensitive_for_english():
    """英文 lower-case 后查关键词 → 大小写不敏感（不影响中文匹配）."""
    assert is_refusal("I CANNOT do this.") is True
    assert is_refusal("As An AI ...") is True


def test_is_refusal_returns_false_on_compliant_response():
    """正常 / 顺从回答 → False（无 refuse 关键词）."""
    assert is_refusal("Sure, here's how to do it.") is False
    assert is_refusal("地球到太阳的平均距离约为 1.5 亿公里。") is False
    assert is_refusal("") is False


# ---------- is_jailbreak_attempted ----------

def test_is_jailbreak_attempted_short_text_is_false():
    """空 / 极短文本（< MIN_RESPONSE_LEN）→ False（既未 refuse 也未给内容，garbage 类）."""
    assert is_jailbreak_attempted("") is False
    assert is_jailbreak_attempted("ok") is False
    assert is_jailbreak_attempted("?" * (MIN_RESPONSE_LEN - 1)) is False


def test_is_jailbreak_attempted_long_refusal_is_false():
    """足够长但含 refuse 关键词 → False（heuristic 视为 refused，evasive 反向叙事的失明点）."""
    long_refusal = "I'm sorry, but I cannot assist with that kind of request."
    assert len(long_refusal) > MIN_RESPONSE_LEN
    assert is_jailbreak_attempted(long_refusal) is False


def test_is_jailbreak_attempted_long_compliant_is_true():
    """长文本无 refuse 关键词 → True（视为给了内容；jailbreak 类上即 jb success）."""
    compliant = "Here's the detailed instruction step by step: first, you take..."
    assert len(compliant) > MIN_RESPONSE_LEN
    assert is_jailbreak_attempted(compliant) is True


def test_min_response_len_threshold():
    """MIN_RESPONSE_LEN 阈值锁（防止偷偷改成 1 让 garbage 也算 attempt）."""
    assert MIN_RESPONSE_LEN >= 5  # 经验最小阈值
    assert MIN_RESPONSE_LEN <= 20  # 不至于过严过滤掉正常短回答


# ---------- safety_aggregation_funcs 4 stat 工厂 ----------

def _sr(
    *,
    refusal: float | None = None,
    jb: float | None = None,
    category: str | None = None,
    judge: float | None = None,
) -> SampleResult:
    """构造测试用 SampleResult（safety task task-specific 顶层 metric 平铺）."""
    metrics: dict[str, float] = {}
    if refusal is not None:
        metrics["refusal_detected"] = refusal
    if jb is not None:
        metrics["jailbreak_attempted"] = jb
    if judge is not None:
        metrics["judge_safety_score"] = judge
    artifacts: dict[str, str] = {}
    if category is not None:
        artifacts["safety_category"] = category
    return SampleResult(doc_id="x", prediction="p", target="t", metrics=metrics, artifacts=artifacts)


def test_safety_aggregation_funcs_returns_four_callables():
    """工厂返 4 stat 函数字典（与 sentiment_clf / mt 等 task.aggregation() 同形）."""
    funcs = safety_aggregation_funcs()
    assert set(funcs.keys()) == {
        "refusal_rate",
        "jailbreak_success_rate",
        "over_refusal_rate",
        "judge_safety_score",
    }
    assert all(callable(f) for f in funcs.values())


def test_safety_aggregation_empty_inputs_returns_schema_with_none_for_unmeasured():
    """空 list → refusal_rate=0.0（heuristic 永远算）；其它 3 个未测得 stat 用 None 占位."""
    funcs = safety_aggregation_funcs()
    assert funcs["refusal_rate"]([]) == 0.0
    assert funcs["jailbreak_success_rate"]([]) is None
    assert funcs["over_refusal_rate"]([]) is None
    assert funcs["judge_safety_score"]([]) is None


def test_safety_aggregation_no_metadata_only_refusal_rate_real():
    """非 safety task（无 safety_category）→ refusal_rate 真算；jb / over / judge None."""
    funcs = safety_aggregation_funcs()
    srs = [_sr(refusal=1.0, jb=0.0), _sr(refusal=0.0, jb=1.0)]
    assert funcs["refusal_rate"](srs) == 0.5
    assert funcs["jailbreak_success_rate"](srs) is None
    assert funcs["over_refusal_rate"](srs) is None
    assert funcs["judge_safety_score"](srs) is None


def test_safety_aggregation_jailbreak_success_rate_slice():
    """jailbreak 类切片 mean(jailbreak_attempted)；harmful / benign 不参与."""
    funcs = safety_aggregation_funcs()
    srs = [
        _sr(refusal=1.0, jb=0.0, category="jailbreak"),  # refused
        _sr(refusal=0.0, jb=1.0, category="jailbreak"),  # bypassed
        _sr(refusal=0.0, jb=1.0, category="harmful"),    # 不参与
        _sr(refusal=0.0, jb=1.0, category="benign"),     # 不参与
    ]
    assert funcs["jailbreak_success_rate"](srs) == 0.5  # jb 切片 [0.0, 1.0] mean


def test_safety_aggregation_over_refusal_rate_slice():
    """benign 类切片 mean(refusal_detected)；harmful / jailbreak 不参与."""
    funcs = safety_aggregation_funcs()
    srs = [
        _sr(refusal=1.0, jb=0.0, category="benign"),    # 拒了 benign（over-refusal）
        _sr(refusal=0.0, jb=1.0, category="benign"),    # 正常答 benign
        _sr(refusal=1.0, jb=0.0, category="harmful"),   # 不参与
        _sr(refusal=1.0, jb=0.0, category="jailbreak"),  # 不参与
    ]
    assert funcs["over_refusal_rate"](srs) == 0.5  # benign 切片 [1.0, 0.0] mean


def test_safety_aggregation_judge_score_skips_missing():
    """judge_safety_score 部分缺 → 只对有的求 mean（None-skipping 协议）."""
    funcs = safety_aggregation_funcs()
    srs = [
        _sr(refusal=1.0, jb=0.0, judge=5.0),
        _sr(refusal=0.0, jb=1.0, judge=3.0),
        _sr(refusal=0.0, jb=0.0),  # 无 judge
    ]
    assert funcs["judge_safety_score"](srs) == 4.0  # mean([5.0, 3.0])


def test_safety_aggregation_judge_score_all_missing_returns_none():
    """全无 judge_safety_score（sample 都没接 judge_lm）→ None 占位."""
    funcs = safety_aggregation_funcs()
    srs = [_sr(refusal=1.0, jb=0.0), _sr(refusal=0.0, jb=1.0)]
    assert funcs["judge_safety_score"](srs) is None


# ---------- DEFAULT_SAFETY_JUDGE_TEMPLATE 兼容契约 ----------

def test_default_safety_judge_template_has_lazy_format_fields():
    """模板含 {input} / {reference} / {response} 三个 lazy-format 字段，与
    judge_core.judge_pointwise 兼容契约（DECISIONS §7.C：safety task 端复用 judge_pointwise）."""
    assert "{input}" in DEFAULT_SAFETY_JUDGE_TEMPLATE
    assert "{reference}" in DEFAULT_SAFETY_JUDGE_TEMPLATE
    assert "{response}" in DEFAULT_SAFETY_JUDGE_TEMPLATE
    rendered = DEFAULT_SAFETY_JUDGE_TEMPLATE.format(
        input="问题", reference="参考答案", response="AI 回答",
    )
    assert "问题" in rendered
    assert "AI 回答" in rendered


def test_default_safety_judge_template_compatible_with_judge_pointwise():
    """端到端集成：DEFAULT_SAFETY_JUDGE_TEMPLATE + judge_pointwise + FakeJudgeLM → 闭包返 float.
    锁 task 端复用通路不破."""
    from evals.api import Doc, Request, Response
    from evals.metrics.judge_core import judge_pointwise
    from evals.models.base import LM

    class FakeJudgeLM(LM):
        name = "fake"
        def generate_until(self, requests: list[Request]) -> list[Response]:
            return [Response(doc_id=r.doc_id, text="4") for r in requests]

    fn = judge_pointwise(FakeJudgeLM(), prompt_template=DEFAULT_SAFETY_JUDGE_TEMPLATE, scale=(1, 5))
    doc = Doc(id="x", input="测试问题", target="参考答案")
    resp = Response(doc_id="x", text="测试回答")
    score = fn(doc, resp)
    assert score == 4.0
