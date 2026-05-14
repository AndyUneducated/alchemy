"""Task 注册全集 + registry 单元锁.

围绕 `tasks/__init__.py` 的副作用 import 与 `registry.py` 的 ABC 行为：

  ① **全集 sentinel**：`list_tasks()` 必须等于一份**显式枚举的 12 个 task name 集合**——
     新加 task 忘了在 `tasks/__init__.py` 加 `from . import X`，CLI 上 `--task X`
     直接 unknown task 报错，但本地 dev 容易在 `pytest evals/tests` 下静默通过.
     这条 sentinel 把"漏 import 副作用"变成显形.

  ② **registry ABC 行为**：duplicate name → ValueError；unknown name → KeyError；
     `list_tasks()` 排序稳定. 历史上仅通过 cli 间接覆盖；改 registry 错误传播没人
     立刻知道.

  ③ **每个 task `output_type` 锁**：output_type ∈ {generate_until, none}（loglikelihood
     phase 9+ 启用），漂移会破坏 runner output_type='none' 跳 LM 分支的契约.
"""

from __future__ import annotations

import pytest

from evals import tasks  # noqa: F401  — 触发 @register_task 副作用
from evals.registry import _TASKS, get_task, list_tasks, register_task
from evals.tasks.base import Task

# 当前 registered tasks 的全集（按 list_tasks 字典序）.
# 加新 task 时同步更新此集合 + `tasks/__init__.py`；二者必须同源.
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

# 每个 task 的 output_type，runner._build_request / runner.evaluate_run 的 dispatch 据此.
# 漂移会让 output_type='none' 的 task 反而真的去调 LM（或反之），破坏 phase 4 RAG /
# phase 5 agent_traj / phase 8 IAA run 路径的核心契约.
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


# ---------- ① 全集 sentinel ------------------------------------------------

def test_list_tasks_matches_expected_set():
    """`tasks/__init__.py` 副作用 import 完整：注册的 task 集合 == 显式枚举集合.

    新加 task 同时漏 `from . import X` → CLI 真实场景下 `--task X` unknown，
    但本地 evals/tests 全集 import 链能命中（其它测试文件可能直接 `from evals.tasks.X import X`），
    导致"看似全绿但 CLI 不可用". 本条 sentinel 直接断这条链.
    """
    assert set(list_tasks()) == set(_EXPECTED_TASK_NAMES), (
        f"task 注册集合漂移：\n"
        f"  expected: {sorted(_EXPECTED_TASK_NAMES)}\n"
        f"  actual:   {list_tasks()}\n"
        f"  missing:  {sorted(_EXPECTED_TASK_NAMES - set(list_tasks()))}\n"
        f"  unexpected: {sorted(set(list_tasks()) - _EXPECTED_TASK_NAMES)}"
    )


def test_list_tasks_is_sorted():
    """list_tasks() 返字典序——CLI `python -m evals list-tasks` 用户体验依赖此契约."""
    names = list_tasks()
    assert names == sorted(names), f"list_tasks() 不再字典序：{names}"


def test_each_registered_task_has_expected_output_type():
    """每个 task 的 output_type 不能漂移；漂移破坏 runner dispatch 契约.

    runner.evaluate_run 用 task.output_type == 'none' 跳 LM 调用（phase 4 RAG /
    phase 5 agent_traj 关键不变量）；改成 'generate_until' 会触发不必要的 LM 调用 +
    可能让 mock LM 因没有对应 doc 抛 KeyError.
    """
    actual = {name: _TASKS[name].output_type for name in list_tasks()}
    assert actual == _EXPECTED_OUTPUT_TYPES, (
        f"output_type 漂移：\n  expected: {_EXPECTED_OUTPUT_TYPES}\n  actual:   {actual}"
    )


def test_each_registered_task_subclasses_task_abc():
    """每个 task class 必须 Task ABC 子类——@register_task 不强制此约束，但 runner
    依赖 ABC 接口（process_results / aggregation / docs / doc_to_text 等）.
    """
    for name, cls in _TASKS.items():
        assert issubclass(cls, Task), f"{name} 注册的 {cls!r} 不是 Task 子类"


# ---------- ② registry ABC 行为 -------------------------------------------

def test_get_task_unknown_name_raises_keyerror():
    """unknown name → KeyError，errmsg 含已注册集合（debugging 友好）."""
    with pytest.raises(KeyError, match="unknown task"):
        get_task("totally_not_a_task")


def test_register_task_duplicate_name_raises_valueerror():
    """duplicate 注册 → ValueError，禁止覆盖已注册 class（避免静默替换）.

    用一个临时 name + 临时 Task 子类做注册，验证再注册同名时 fail-loud；
    finally 块清理 _TASKS dict 避免污染后续测试.
    """
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
        # 老 class 未被替换
        assert _TASKS[name] is _T1
    finally:
        _TASKS.pop(name, None)


def test_get_task_returns_fresh_instance_each_call():
    """get_task('X') 每次返回 new instance（而不是缓存）——让 task 携带 stateful
    judge_lm / retrieve_fn 不会污染下一次 get_task. CLI `_build_task_with_optional_deps`
    依赖此行为：第一次拿 base_task 探类型 + 第二次按 flag 重新构造.
    """
    a = get_task("sentiment_clf")
    b = get_task("sentiment_clf")
    assert a is not b
    assert type(a) is type(b)
