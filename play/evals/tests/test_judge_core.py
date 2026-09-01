"""metrics/judge_core.py unit layer: 4 judges + parsing + debiasing mechanism, a total of 12 assertions.

Zero network. FakeJudgeLM accepts `list[str]` (advanced by calling cursor) or `Callable[[prompt], text]`
(Rule function) to ensure that the test is completely deterministic and CI-friendly.

According to plan §2.1 + §6 ("Main Stage Assignment"):
  - pointwise: The task layer is the main stage, and only the "mean" shape contract is locked here.
  - pairwise: **This document** is the main stage (swap true debiasing / consistent winner / contradictory tie)
  - g_eval: **this file** is the main stage (multidimensional weighting/multisampling instead of logprob pass)
  - self_consistency: **This file** is the main stage (majority vote / tie lock / set pointwise)"""

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
    """Deterministic LM stub.

    Choose one of the structures:
      - `outputs=list[str]`: loop in calling order (cursor)
      - `outputs=Callable[[prompt], text]`: rule function, determines output based on prompt content
    There are N requests in each generate_until call → push the cursor N steps / adjust the rules N times."""

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


# ---------- parse_pointwise_score (3 items) ----------

def test_parse_pointwise_score_extracts_int():
    """Common judge output formats can be parsed into scores."""
    assert parse_pointwise_score("Score: 4/5") == 4
    assert parse_pointwise_score("4") == 4
    assert parse_pointwise_score("My rating is 4 out of 5") == 4
    assert parse_pointwise_score("**4**") == 4


def test_parse_pointwise_score_invalid_raises():
    """Exceptions are thrown when there is no int at all, and there is no silent fallback - the robustness of judge analysis is the most frequent failure point."""
    with pytest.raises(ValueError):
        parse_pointwise_score("totally not a score")


def test_parse_pointwise_score_clamps_out_of_range():
    """Ints that exceed scale are clamped to the boundary (the behavior is locked).

    The first int that falls within scale is returned first; if it is not present, clamp the first int.
    "Score: 7/5" contains both 7 and 5 → takes precedence over 5 (within scale).
    "0" only clamps 0 → to 1."""
    assert parse_pointwise_score("Score: 7/5") == 5
    assert parse_pointwise_score("0") == 1
    assert parse_pointwise_score("999") == 5


# ---------- judge_pointwise (2 items) ----------

def test_pointwise_mean_with_fake_judge():
    """5 preset scores [3,4,5,2,1], mean=3.0 - basic pointwise shape contract."""
    fake = FakeJudgeLM(outputs=["3", "4", "5", "2", "1"])
    pj = judge_pointwise(fake, prompt_template="rate: {response}")

    scores = [pj(_doc(id=f"d{i}"), _resp(doc_id=f"d{i}")) for i in range(5)]
    assert sum(scores) / len(scores) == 3.0


def test_pointwise_returns_none_on_parse_failure():
    """DECISIONS §X wave 4: LM output no int parsable → closure returns None instead of raise.

    Identical to the "None occupancy not measured" principle of phase 7 wave 2 P2 - 1-5 scale 0 is out of bounds,
    None explicitly indicates "not measured" and the aggregator is naturally filtered; it is different from raise which interrupts the entire run.

    parse_pointwise_score itself still raises (test_parse_pointwise_score_invalid_raises locked);
    The closure layer is the robust boundary of the "application/system layer"."""
    fake = FakeJudgeLM(outputs=["totally not a score"])
    pj = judge_pointwise(fake, prompt_template="rate: {response}")
    assert pj(_doc(), _resp()) is None


# ---------- judge_pairwise + pairwise_winrate (3 items) ----------

def test_pairwise_position_bias_neutralized_by_swap():
    """The biased judge always says "A wins" - after swap, the two positions are contradictory → total tie.

    This is the core assertion of swap debiasing: swap=True really neutralizes the "false positives" of the biased judge."""
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
    """Unanimous judge: Both A/B and B/A will judge the same answer to win → win_rate=1.0.

    Judge is based on "which section contains 'good'", regardless of position - representative of consistent."""

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
    """A/B says A wins, B/A also says A wins → contradiction (position offset) → tie."""
    biased = FakeJudgeLM(outputs=lambda prompt: "A")
    pairs = [(_doc(id="d0"), _resp(doc_id="d0", text="X"), _resp(doc_id="d0", text="Y"))]

    rates = pairwise_winrate(biased, pairs, swap=True)
    assert rates["tie"] == 1.0


# ---------- g_eval (2 items) ----------

def test_g_eval_multidim_aggregation():
    """3 dimensions × n_samples=1 → directly take the score of a single sample in each dimension.

    outputs cycled ["4","5","3"]: coherence=4, relevance=5, fluency=3."""
    fake = FakeJudgeLM(outputs=["4", "5", "3"])
    result = g_eval(
        fake,
        dimensions=("coherence", "relevance", "fluency"),
        prompt_template="rate {dimension}: {response}",
        n_samples=1,
    )(_doc(), _resp())

    assert result == {"coherence": 4.0, "relevance": 5.0, "fluency": 3.0}


def test_g_eval_multi_sample_distribution_replaces_logprob():
    """n_samples=5 mean - a discrete distribution pass that replaces OpenAI logprob's weighted mean.

    outputs=[3,4,5,3,4] → mean=3.8 (3 appears 2 times, 4 appears 2 times, 5 appears 1 time).
    A single sample can only give a point estimate, and multiple samples can approximate the "expectation of the distribution" - this is the core of G-Eval without logprob."""
    fake = FakeJudgeLM(outputs=["3", "4", "5", "3", "4"])
    result = g_eval(
        fake,
        dimensions=("quality",),
        prompt_template="rate {dimension}: {response}",
        n_samples=5,
    )(_doc(), _resp())

    assert abs(result["quality"] - 3.8) < 1e-9


def test_g_eval_dim_returns_none_when_all_samples_unparseable():
    """DECISIONS §X wave 4: Single dimension n_samples all parse failed → None for this dimension "Not Measured".

    The same shape as the None placeholder when phase 7 P2 slice is empty (semantic "all judges failed" ≈ no sample in the slice)."""
    fake = FakeJudgeLM(outputs=["bad", "garbage", "no number"])
    result = g_eval(
        fake,
        dimensions=("coherence",),
        prompt_template="rate {dimension}: {response}",
        n_samples=3,
    )(_doc(), _resp())

    assert result["coherence"] is None


def test_g_eval_dim_partial_failure_uses_valid_subset_mean():
    """DECISIONS §X wave 4: n_samples partial parse failed → this dimension returns valid subset mean.

    outputs=["4", "bad", "5"] n_samples=3 → valid=[4,5], mean=4.5;
    In the same spirit as phase 8 wave 3 independent "OOV sensitive metric cut valid subset"."""
    fake = FakeJudgeLM(outputs=["4", "bad", "5"])
    result = g_eval(
        fake,
        dimensions=("quality",),
        prompt_template="rate {dimension}: {response}",
        n_samples=3,
    )(_doc(), _resp())

    assert result["quality"] == 4.5


# ---------- self_consistency (3 items) ----------

def test_self_consistency_majority_vote():
    """7 samples [A,A,B,A,C,A,B] → mode A (4 votes)."""
    seq = ["A", "A", "B", "A", "C", "A", "B"]
    cursor = [0]

    def base() -> str:
        v = seq[cursor[0]]
        cursor[0] += 1
        return v

    sc = self_consistency(base, n_samples=7)
    assert sc() == "A"


def test_self_consistency_breaks_tie_deterministically():
    """In the event of a tie break, the mode that appears first (first-seen tiebreak) is taken: [B,A,B,A] → B (B comes first 2 votes).

    Locking behavior to avoid sneaking into dictionary order/random."""
    seq = ["B", "A", "B", "A"]
    cursor = [0]

    def base() -> str:
        v = seq[cursor[0]]
        cursor[0] += 1
        return v

    sc = self_consistency(base, n_samples=4)
    assert sc() == "B"


def test_self_consistency_wraps_pointwise():
    """Wrapped in a pointwise outer layer: the mode of 5 samples is a single score.

    fake outputs [4,4,5,4,3] → counts {4:3, 5:1, 3:1} → mode 4."""
    fake = FakeJudgeLM(outputs=["4", "4", "5", "4", "3"])
    base = judge_pointwise(fake, prompt_template="rate: {response}")
    sc = self_consistency(base, n_samples=5)

    assert sc(_doc(), _resp()) == 4
