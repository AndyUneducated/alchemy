"""Phase 4 Runner output_type='none' 分支：声明无 LM 调用的 task 必须真的不调 lm.generate_until.

`rag_retrieval` 是首个 output_type='none' task；这里用 minimal stub task + spy LM
覆盖框架契约——避免 RAG task 自己出 bug 时把"runner 跳了 LM"这条不变量也带塌.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from evals.api import Doc, Request, Response, SampleResult
from evals.models.base import LM
from evals.runner import evaluate_run
from evals.tasks.base import Task


class _SpyLM(LM):
    """generate_until 被调即记录调用，跑完总能从 .calls 看见调用次数."""

    def __init__(self) -> None:
        self.name = "spy"
        self.calls: list[list[Request]] = []

    def generate_until(self, requests: list[Request]) -> list[Response]:
        self.calls.append(list(requests))
        return [Response(doc_id=req.doc_id, text="x") for req in requests]


class _NoLMTask(Task):
    """最小 stub：output_type='none' + process_docs identity + process_results 空指标.

    注意：output_type 是 ClassVar，子类直接重新声明即可（与 sentiment_clf / mt 同模式）.
    """

    name: ClassVar[str] = "_no_lm_task_for_test"
    output_type: ClassVar[str] = "none"

    def docs(self):
        return [Doc(id="d1", input="q", target=None), Doc(id="d2", input="q2", target=None)]

    def doc_to_text(self, doc: Doc) -> str:
        return "should_never_be_called"

    def doc_to_target(self, doc: Doc) -> str:
        return ""

    def process_results(self, doc: Doc, response: Response) -> SampleResult:
        # response 必须是 phase 4 的 placeholder（text=None）——otherwise runner bug.
        assert response.text is None, "output_type='none' 分支不应该填 response.text"
        return SampleResult(doc_id=doc.id, prediction="", target="", metrics={"placeholder": 1.0})

    def aggregation(self) -> dict[str, Callable[[list[SampleResult]], float]]:
        return {"placeholder": lambda srs: float(len(srs))}

    def higher_is_better(self) -> dict[str, bool]:
        return {"placeholder": True}


def test_output_type_none_skips_lm_generate_until():
    """`output_type='none'` → spy LM 一次没被调，但 sample_results 仍 per-doc 产出."""
    task = _NoLMTask()
    spy = _SpyLM()

    r = evaluate_run(task, spy)

    # 核心：LM 没被触碰
    assert spy.calls == []
    # 但 task.process_results 仍按 doc 顺序产出（runner 闭环没漏样本）
    assert r.n == 2
    assert {s.doc_id for s in r.per_sample} == {"d1", "d2"}
    assert r.aggregated["placeholder"] == 2.0


def test_output_type_none_uses_lm_name_in_run_id():
    """虽然不调 LM，model_label 仍是 lm.name——保留 storage / show 的人类可读追踪."""
    task = _NoLMTask()
    spy = _SpyLM()

    r = evaluate_run(task, spy)
    assert r.model == "spy"
    assert r.mode == "run"
