"""Task registration set + registry unit lock.

ABC behavior around side effects of `tasks/__init__.py` import and `registry.py`:

  ① **Full set sentinel**: `list_tasks()` must be equal to a **explicitly enumerated set of 12 task names**——
     Newly added task. Forgot to add `from . import X` in `tasks/__init__.py`, `--task X` on CLI
     Direct unknown task error is reported, but local dev can easily pass silently under `pytest evals/tests`.
     This sentinel makes "leaky import side effects" explicit.

  ② **registry ABC behavior**: duplicate name → ValueError; unknown name → KeyError;
     `list_tasks()` sorting is stable. Historically, it was only covered indirectly through the cli; no one changed the registry error propagation
     Know immediately.

  ③ **Each task `output_type` lock**: output_type ∈ {generate_until, none} (loglikelihood
     phase 9+ enabled), drift will break the contract of runner output_type='none' jumping the LM branch."""

from __future__ import annotations

import pytest

from evals import tasks  # noqa: F401 — Trigger @register_task side effect
from evals.registry import _TASKS, get_task, list_tasks, register_task
from evals.tasks.base import Task

# The complete set of currently registered tasks (in lexicographic order by list_tasks).
# When adding a new task, update this collection + `tasks/__init__.py` synchronously; both must have the same source.
_EXPECTED_TASK_NAMES: frozenset[str] = frozenset({
    "agent_traj",
    "bfcl_slice",
    "iaa_nominal",
    "iaa_ordinal",
    "mmlu_slice",
    "mt",
    "nudge_fire_rate",
    "qa_open",
    "rag_qa",
    "rag_retrieval",
    "safety",
    "sentiment_clf",
})

# The output_type of each task, the dispatch of runner._build_request / runner.evaluate_run is based on this.
# Drift will cause the task with output_type='none' to actually adjust LM (or vice versa), destroying phase 4 RAG /
# Core contract for phase 5 agent_traj / phase 8 IAA run path.
_EXPECTED_OUTPUT_TYPES: dict[str, str] = {
    "agent_traj": "none",
    "bfcl_slice": "generate_until",
    "iaa_nominal": "none",
    "iaa_ordinal": "none",
    "mmlu_slice": "generate_until",
    "mt": "generate_until",
    "nudge_fire_rate": "none",
    "qa_open": "generate_until",
    "rag_qa": "generate_until",
    "rag_retrieval": "none",
    "safety": "generate_until",
    "sentiment_clf": "generate_until",
}


# ---------- ① Complete works sentinel ------------------------------------------------

def test_list_tasks_matches_expected_set():
    """`tasks/__init__.py` Side effects import complete: registered task collection == explicit enumeration collection.

    Adding a new task also leaks `from . import X` → CLI `--task X` unknown in real scenarios,
    But the local evals/tests complete set import chain can hit (other test files may directly `from evals.tasks.X import X`),
    This leads to "It looks completely green but the CLI is unavailable". This article of sentinel directly breaks this link."""
    assert set(list_tasks()) == set(_EXPECTED_TASK_NAMES), (
        f"task 注册集合漂移：\n"
        f"  expected: {sorted(_EXPECTED_TASK_NAMES)}\n"
        f"  actual:   {list_tasks()}\n"
        f"  missing:  {sorted(_EXPECTED_TASK_NAMES - set(list_tasks()))}\n"
        f"  unexpected: {sorted(set(list_tasks()) - _EXPECTED_TASK_NAMES)}"
    )


def test_list_tasks_is_sorted():
    """list_tasks() returns lexicographic order - CLI `python -m evals list-tasks` User experience depends on this contract."""
    names = list_tasks()
    assert names == sorted(names), f"list_tasks() 不再字典序：{names}"


def test_each_registered_task_has_expected_output_type():
    """The output_type of each task cannot drift; drift breaks the runner dispatch contract.

    runner.evaluate_run uses task.output_type == 'none' to jump to LM call (phase 4 RAG/
    phase 5 agent_traj key invariant); changing to 'generate_until' will trigger unnecessary LM calls +
    It is possible for mock LM to throw KeyError because there is no corresponding doc."""
    actual = {name: _TASKS[name].output_type for name in list_tasks()}
    assert actual == _EXPECTED_OUTPUT_TYPES, (
        f"output_type 漂移：\n  expected: {_EXPECTED_OUTPUT_TYPES}\n  actual:   {actual}"
    )


def test_each_registered_task_subclasses_task_abc():
    """Every task class must subclass Task ABC - @register_task does not enforce this constraint, but runner
    Depends on ABC interface (process_results / aggregation / docs / doc_to_text, etc.)."""
    for name, cls in _TASKS.items():
        assert issubclass(cls, Task), f"{name} 注册的 {cls!r} 不是 Task 子类"


# ---------- ② registry ABC behavior -----------------------------------------------

def test_get_task_unknown_name_raises_keyerror():
    """unknown name → KeyError, errmsg contains registered collections (debugging friendly)."""
    with pytest.raises(KeyError, match="unknown task"):
        get_task("totally_not_a_task")


def test_register_task_duplicate_name_raises_valueerror():
    """Duplicate registration → ValueError, prohibiting overwriting registered classes (avoiding silent replacement).

    Use a temporary name + temporary Task subclass to register, and fail-loud when verifying and then registering with the same name;
    The finally block cleans the _TASKS dict to avoid polluting subsequent tests."""
    name = "_test_dup_registration"
    assert name not in _TASKS  # sanity

    class _T1(Task):
        name_attr = name
        output_type = "generate_until"
        def docs(self): return []
        def doc_to_text(self, doc): return ""
        def doc_to_target(self, doc): return ""
        def process_results(self, doc, response): raise NotImplementedError
        def aggregation(self): return {}
        def higher_is_better(self): return {}

    class _T2(_T1):
        pass

    try:
        register_task(name)(_T1)
        assert _TASKS[name] is _T1
        with pytest.raises(ValueError, match="already registered"):
            register_task(name)(_T2)
        # The old class has not been replaced
        assert _TASKS[name] is _T1
    finally:
        _TASKS.pop(name, None)


def test_get_task_returns_fresh_instance_each_call():
    """get_task('X') returns new instance each time (instead of cache) - let the task carry stateful
    judge_lm / retrieve_fn will not pollute the next get_task. CLI `_build_task_with_optional_deps`
    Rely on this behavior: first use base_task to detect the type + second time press flag to reconstruct."""
    a = get_task("sentiment_clf")
    b = get_task("sentiment_clf")
    assert a is not b
    assert type(a) is type(b)
