"""_JudgeRecorder protocol unit lock for metrics/judge_core.py + judge_rag.py (DECISIONS §7.3).

Lock:
  1. _JudgeRecorder.call() is passed transparently to lm.generate_until and append responses
  2. The three factories of judge_pointwise / g_eval / judge_pairwise expose the `_recorder` attribute
  3. self_consistency wrapper transparently transmits base’s _recorder
  4. judge_rag.py 5 factories also expose `_recorder`
  5. recorder.responses is list[Response], including model_label
  6. recorder.reset() Clear"""

from __future__ import annotations

from evals.api import Doc, Request, Response, Usage
from evals.metrics.judge_core import (
    _JudgeRecorder,
    g_eval,
    judge_pairwise,
    judge_pointwise,
    self_consistency,
)
from evals.models.base import LM


class _CountingFakeLM(LM):
    """Latency/tokens are recorded for each call to facilitate testing of recorder collection behavior."""

    def __init__(self, score_text: str = "4", label: str = "fake:rec") -> None:
        self.name = label
        self._score_text = score_text
        self.calls = 0

    def generate_until(self, requests: list[Request]) -> list[Response]:
        out: list[Response] = []
        for i, r in enumerate(requests):
            self.calls += 1
            out.append(
                Response(
                    doc_id=r.doc_id,
                    text=self._score_text,
                    latency_ms=float(100 + i * 10),
                    usage=Usage(tokens_in=20, tokens_out=2),
                )
            )
        return out


# ---------- _JudgeRecorder's own behavior ----------

def test_recorder_call_proxies_to_lm_and_appends():
    lm = _CountingFakeLM()
    rec = _JudgeRecorder(lm)
    out = rec.call([Request(doc_id="d1", prompt="p", request_type="generate_until", max_tokens=8)])
    assert len(out) == 1
    assert lm.calls == 1
    assert len(rec.responses) == 1
    assert rec.responses[0].text == "4"
    assert rec.model_label == lm.name


def test_recorder_accumulates_across_calls():
    lm = _CountingFakeLM()
    rec = _JudgeRecorder(lm)
    rec.call([Request(doc_id="d1", prompt="a", request_type="generate_until", max_tokens=8)])
    rec.call([Request(doc_id="d2", prompt="b", request_type="generate_until", max_tokens=8)])
    rec.call([Request(doc_id="d3", prompt="c", request_type="generate_until", max_tokens=8)])
    assert len(rec.responses) == 3


def test_recorder_reset_clears_responses():
    lm = _CountingFakeLM()
    rec = _JudgeRecorder(lm)
    rec.call([Request(doc_id="d1", prompt="p", request_type="generate_until", max_tokens=8)])
    assert len(rec.responses) == 1
    rec.reset()
    assert rec.responses == []


# ---------- judge_pointwise / g_eval / judge_pairwise exposed _recorder ----------

def test_judge_pointwise_exposes_recorder():
    lm = _CountingFakeLM()
    fn = judge_pointwise(lm, prompt_template="rate: {response}")
    assert hasattr(fn, "_recorder")
    assert fn._recorder.model_label == lm.name
    # recorder.responses should grow after calling it once
    score = fn(Doc(id="d1", input="?", target="t"), Response(doc_id="d1", text="r"))
    assert score == 4.0
    assert len(fn._recorder.responses) == 1


def test_g_eval_exposes_recorder_and_records_all_dim_samples():
    lm = _CountingFakeLM()
    fn = g_eval(lm, dimensions=("dimA", "dimB"), n_samples=3, prompt_template="rate {dimension}: {response}")
    assert hasattr(fn, "_recorder")
    fn(Doc(id="d1", input="?", target="t"), Response(doc_id="d1", text="r"))
    # 2 dim × 3 samples = 6 calls
    assert len(fn._recorder.responses) == 6


def test_judge_pairwise_exposes_recorder():
    lm = _CountingFakeLM(score_text="A")
    fn = judge_pairwise(lm, swap=False, prompt_template="cmp: {response_a} vs {response_b}")
    assert hasattr(fn, "_recorder")
    fn(Doc(id="d1", input="?", target="t"),
       Response(doc_id="d1", text="ra"),
       Response(doc_id="d1", text="rb"))
    assert len(fn._recorder.responses) == 1


# ---------- self_consistency transparent transmission _recorder ----------

def test_self_consistency_propagates_recorder_from_base():
    lm = _CountingFakeLM()
    base = judge_pointwise(lm, prompt_template="rate: {response}")
    wrapped = self_consistency(base, n_samples=3)
    assert hasattr(wrapped, "_recorder")
    # The wrapper adjusts the base multiple times internally, and accumulates it through the same recorder each time.
    wrapped(Doc(id="d1", input="?", target="t"), Response(doc_id="d1", text="r"))
    assert len(wrapped._recorder.responses) == 3
    # base and wrapper share the same recorder
    assert wrapped._recorder is base._recorder


# ---------- judge_rag.py 5 factory are exposed _recorder ----------

def test_judge_rag_5_factories_each_expose_recorder():
    from evals.metrics.judge_rag import (
        judge_answer_correctness,
        judge_answer_relevancy,
        judge_context_precision,
        judge_context_recall,
        judge_faithfulness,
    )

    lm = _CountingFakeLM()
    fns = [
        judge_faithfulness(lm),
        judge_answer_correctness(lm),
        judge_context_precision(lm),
        judge_context_recall(lm),
        judge_answer_relevancy(lm),
    ]
    for fn in fns:
        assert hasattr(fn, "_recorder"), f"{fn} should expose _recorder"
        assert fn._recorder.model_label == lm.name
