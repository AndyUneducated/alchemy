"""Phase 4 Runner output_type='none' branch: Tasks that declare no LM calls must really not adjust lm.generate_until.

`rag_retrieval` is the first output_type='none' task; minimal stub task + spy LM is used here
Override the framework contract - to prevent the RAG task from collapsing the invariant "runner jumped to LM" when a bug occurs."""

from __future__ import annotations

from typing import Callable, ClassVar

from evals.api import Doc, Request, Response, SampleResult
from evals.models.base import LM
from evals.runner import evaluate_run
from evals.tasks.base import Task


class _SpyLM(LM):
    """When generate_until is called, the calls are recorded. After running, you can always see the number of calls from .calls."""

    def __init__(self) -> None:
        self.name = "spy"
        self.calls: list[list[Request]] = []

    def generate_until(self, requests: list[Request]) -> list[Response]:
        self.calls.append(list(requests))
        return [Response(doc_id=req.doc_id, text="x") for req in requests]


class _NoLMTask(Task):
    """Minimum stub: output_type='none' + process_docs identity + process_results empty indicator.

    Note: output_type is ClassVar, subclasses can be directly re-declared (same mode as sentiment_clf / mt)."""

    name: ClassVar[str] = "_no_lm_task_for_test"
    output_type: ClassVar[str] = "none"

    def docs(self):
        return [Doc(id="d1", input="q", target=None), Doc(id="d2", input="q2", target=None)]

    def doc_to_text(self, doc: Doc) -> str:
        return "should_never_be_called"

    def doc_to_target(self, doc: Doc) -> str:
        return ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        # response must be a phase 4 placeholder (text=None) - otherwise runner bug.
        assert response.text is None, "output_type='none' 分支不应该填 response.text"
        return SampleResult(doc_id=doc.id, prediction="", target="", metrics={"placeholder": 1.0})

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        return {"placeholder": lambda srs: float(len(srs))}

    def higher_is_better(self) -> dict[str, bool]:
        return {"placeholder": True}


def test_output_type_none_skips_lm_generate_until():
    """`output_type='none'` → spy LM is not called once, but sample_results is still output per-doc."""
    task = _NoLMTask()
    spy = _SpyLM()

    r = evaluate_run(task, spy)

    # Core: LM untouched
    assert spy.calls == []
    # But task.process_results is still output in doc order (the runner closed loop does not miss samples)
    assert r.n == 2
    assert {s.doc_id for s in r.per_sample} == {"d1", "d2"}
    assert r.aggregated["placeholder"] == 2.0


def test_output_type_none_uses_lm_name_in_run_id():
    """Although LM is not adjusted, model_label is still lm.name - retaining human-readable traces of storage / show."""
    task = _NoLMTask()
    spy = _SpyLM()

    r = evaluate_run(task, spy)
    assert r.model == "spy"
    assert r.mode == "run"
