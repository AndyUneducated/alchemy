"""metrics/judge_core.py 单元层：4 个 judge + 解析 + 去偏机制 共 12 条断言.

零网络。FakeJudgeLM 接受 `list[str]`（按调用 cursor 推进）或 `Callable[[prompt], text]`
（规则函数），保证测试完全确定性、CI 友好。

按 plan §二.1 + §六（"主舞台分配"）：
  - pointwise: task 层是主舞台，这里只锁 "mean" 形状契约
  - pairwise: **本文件**是主舞台（swap 真去偏 / 一致 winner / 矛盾 tie）
  - g_eval:   **本文件**是主舞台（多维加权 / 多采样替代 logprob 通路）
  - self_consistency: **本文件**是主舞台（majority vote / tie 锁死 / 套 pointwise）
"""

from __future__ import annotations

from typing import Callable

import pytest

from evals.api import Doc, Request, Response
from evals.metrics.judge_core import (
    g_eval,
    judge_pairwise,
    judge_pointwise,
    pairwise_winrate,
    parse_pointwise_score,
    self_consistency,
)
from evals.models.base import LM


class FakeJudgeLM(LM):
    """确定性 LM stub。

    构造选其一：
      - `outputs=list[str]`：按调用顺序循环（cursor）
      - `outputs=Callable[[prompt], text]`：规则函数，根据 prompt 内容决定输出
    每个 generate_until call 内有 N 个 request → 推 N 步 cursor / 调 N 次规则.
    """

    def __init__(self, outputs, *, name: str = "fake") -> None:
        self.name = name
        self._cursor = 0
        if callable(outputs):
            self._fn: Callable[[str], str] | None = outputs
            self._outputs: list[str] | None = None
        else:
            self._fn = None
            self._outputs = list(outputs)

    def generate_until(self, requests: list[Request]) -> list[Response]:
        out: list[Response] = []
        for req in requests:
            if self._fn is not None:
                text = self._fn(req.prompt)
            else:
                assert self._outputs is not None
                text = self._outputs[self._cursor % len(self._outputs)]
                self._cursor += 1
            out.append(Response(doc_id=req.doc_id, text=text))
        return out


def _doc(id: str = "d0", input: str = "q?", target: str = "ref") -> Doc:
    return Doc(id=id, input=input, target=target)


def _resp(doc_id: str = "d0", text: str = "hyp") -> Response:
    return Response(doc_id=doc_id, text=text)


# ---------- parse_pointwise_score（3 条）----------

def test_parse_pointwise_score_extracts_int():
    """常见 judge 输出格式都能解析到分数."""
    assert parse_pointwise_score("Score: 4/5") == 4
    assert parse_pointwise_score("4") == 4
    assert parse_pointwise_score("My rating is 4 out of 5") == 4
    assert parse_pointwise_score("**4**") == 4


def test_parse_pointwise_score_invalid_raises():
    """完全无 int 时抛异常，不静默 fallback——judge 解析鲁棒是最高频故障点."""
    with pytest.raises(ValueError):
        parse_pointwise_score("totally not a score")


def test_parse_pointwise_score_clamps_out_of_range():
    """超出 scale 的 int 被 clamp 到边界（行为锁死）。

    优先返回首个落在 scale 内的 int；都不在则 clamp 第一个 int。
    "Score: 7/5" 同时含 7 和 5 → 优先 5（在 scale 内）。
    "0" 只有 0 → clamp 到 1。
    """
    assert parse_pointwise_score("Score: 7/5") == 5
    assert parse_pointwise_score("0") == 1
    assert parse_pointwise_score("999") == 5


# ---------- judge_pointwise（2 条）----------

def test_pointwise_mean_with_fake_judge():
    """5 条预设分数 [3,4,5,2,1]，mean=3.0——基础 pointwise 形状契约."""
    fake = FakeJudgeLM(outputs=["3", "4", "5", "2", "1"])
    pj = judge_pointwise(fake, prompt_template="rate: {response}")

    scores = [pj(_doc(id=f"d{i}"), _resp(doc_id=f"d{i}")) for i in range(5)]
    assert sum(scores) / len(scores) == 3.0


def test_pointwise_returns_none_on_parse_failure():
    """DECISIONS §X wave 4：LM 输出无 int 可解析 → closure 返 None 而非 raise.

    与 phase 7 wave 2 P2 立的"None 占位未测得"原则同形——1-5 scale 0 越界，
    None 显式表"未测得"，aggregator 自然过滤；区别于 raise 中断整 run.

    parse_pointwise_score 自身仍 raise（test_parse_pointwise_score_invalid_raises 锁定）；
    closure 层是"应用 / 系统层"的鲁棒边界.
    """
    fake = FakeJudgeLM(outputs=["totally not a score"])
    pj = judge_pointwise(fake, prompt_template="rate: {response}")
    assert pj(_doc(), _resp()) is None


# ---------- judge_pairwise + pairwise_winrate（3 条）----------

def test_pairwise_position_bias_neutralized_by_swap():
    """biased judge 永远说"A 赢"——经 swap 双跑两位置矛盾 → 全计 tie.

    这是 swap 去偏的核心断言：swap=True 真的把 biased judge 的"假阳性"中和掉了.
    """
    biased = FakeJudgeLM(outputs=lambda prompt: "A")
    pairs = [
        (_doc(id=f"d{i}"), _resp(doc_id=f"d{i}", text="X"), _resp(doc_id=f"d{i}", text="Y"))
        for i in range(10)
    ]
    rates = pairwise_winrate(biased, pairs, swap=True)

    assert rates["a"] == 0.0
    assert rates["b"] == 0.0
    assert rates["tie"] == 1.0


def test_pairwise_consistent_judge_records_winner():
    """一致 judge：A/B 与 B/A 双跑都判同一回答赢 → win_rate=1.0.

    judge 依据"哪个 section 含 'good'"判，与位置无关——consistent 的代表.
    """

    def judge(prompt: str) -> str:
        a_section = prompt.split("Response A:")[1].split("Response B:")[0]
        return "A" if "good" in a_section else "B"

    consistent = FakeJudgeLM(outputs=judge)
    pairs = [
        (_doc(id=f"d{i}"), _resp(doc_id=f"d{i}", text="good"), _resp(doc_id=f"d{i}", text="bad"))
        for i in range(5)
    ]
    rates = pairwise_winrate(consistent, pairs, swap=True)

    assert rates["a"] == 1.0
    assert rates["tie"] == 0.0


def test_pairwise_inconsistent_pair_counts_as_tie():
    """A/B 说 A 赢、B/A 也说 A 赢 → 矛盾（位置偏置）→ 计 tie."""
    biased = FakeJudgeLM(outputs=lambda prompt: "A")
    pairs = [(_doc(id="d0"), _resp(doc_id="d0", text="X"), _resp(doc_id="d0", text="Y"))]

    rates = pairwise_winrate(biased, pairs, swap=True)
    assert rates["tie"] == 1.0


# ---------- g_eval（2 条）----------

def test_g_eval_multidim_aggregation():
    """3 维度 × n_samples=1 → 每维直接取单条采样的 score.

    outputs cycled ["4","5","3"]：coherence=4, relevance=5, fluency=3.
    """
    fake = FakeJudgeLM(outputs=["4", "5", "3"])
    result = g_eval(
        fake,
        dimensions=("coherence", "relevance", "fluency"),
        prompt_template="rate {dimension}: {response}",
        n_samples=1,
    )(_doc(), _resp())

    assert result == {"coherence": 4.0, "relevance": 5.0, "fluency": 3.0}


def test_g_eval_multi_sample_distribution_replaces_logprob():
    """n_samples=5 取均值——替代 OpenAI logprob 加权 mean 的离散分布通路.

    outputs=[3,4,5,3,4] → mean=3.8（3 出现 2 次、4 出现 2 次、5 出现 1 次）.
    单样本只能给点估计，多采样才能逼近"分布的期望"——这是 G-Eval 在无 logprob 下的核心.
    """
    fake = FakeJudgeLM(outputs=["3", "4", "5", "3", "4"])
    result = g_eval(
        fake,
        dimensions=("quality",),
        prompt_template="rate {dimension}: {response}",
        n_samples=5,
    )(_doc(), _resp())

    assert abs(result["quality"] - 3.8) < 1e-9


def test_g_eval_dim_returns_none_when_all_samples_unparseable():
    """DECISIONS §X wave 4：单维 n_samples 全部 parse 失败 → 该维 None"未测得".

    与 phase 7 P2 切片为空时 None 占位同形（语义"判官全失败" ≈ 切片中无样本）.
    """
    fake = FakeJudgeLM(outputs=["bad", "garbage", "no number"])
    result = g_eval(
        fake,
        dimensions=("coherence",),
        prompt_template="rate {dimension}: {response}",
        n_samples=3,
    )(_doc(), _resp())

    assert result["coherence"] is None


def test_g_eval_dim_partial_failure_uses_valid_subset_mean():
    """DECISIONS §X wave 4：n_samples 部分 parse 失败 → 该维返 valid 子集 mean.

    outputs=["4", "bad", "5"] n_samples=3 → valid=[4,5]，mean=4.5；
    与 phase 8 wave 3 立的"OOV 敏感 metric 切 valid subset"同精神.
    """
    fake = FakeJudgeLM(outputs=["4", "bad", "5"])
    result = g_eval(
        fake,
        dimensions=("quality",),
        prompt_template="rate {dimension}: {response}",
        n_samples=3,
    )(_doc(), _resp())

    assert result["quality"] == 4.5


# ---------- self_consistency（3 条）----------

def test_self_consistency_majority_vote():
    """7 次采样 [A,A,B,A,C,A,B] → 众数 A（4 票）."""
    seq = ["A", "A", "B", "A", "C", "A", "B"]
    cursor = [0]

    def base() -> str:
        v = seq[cursor[0]]
        cursor[0] += 1
        return v

    sc = self_consistency(base, n_samples=7)
    assert sc() == "A"


def test_self_consistency_breaks_tie_deterministically():
    """平票时取首个出现的众数（first-seen tiebreak）：[B,A,B,A] → B（B 先到 2 票）.

    锁死行为，避免实现偷换为字典序 / 随机.
    """
    seq = ["B", "A", "B", "A"]
    cursor = [0]

    def base() -> str:
        v = seq[cursor[0]]
        cursor[0] += 1
        return v

    sc = self_consistency(base, n_samples=4)
    assert sc() == "B"


def test_self_consistency_wraps_pointwise():
    """套在 pointwise 外层：5 次采样的 mode 是单一 score.

    fake outputs [4,4,5,4,3] → counts {4:3, 5:1, 3:1} → 众数 4.
    """
    fake = FakeJudgeLM(outputs=["4", "4", "5", "4", "3"])
    base = judge_pointwise(fake, prompt_template="rate: {response}")
    sc = self_consistency(base, n_samples=5)

    assert sc(_doc(), _resp()) == 4
