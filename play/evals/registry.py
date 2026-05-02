"""Task 注册表：装饰器 + 字典.

`@register_task("name")` 在类定义点直接登记，避免"改了 task 忘改注册表"的脏状态。
代价：import time 副作用——`tasks/__init__.py` 里显式 `from . import sentiment_clf`
触发装饰器，让 CLI 能看到。

同家族 pattern：Django URL、Flask/FastAPI route、pytest fixture。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .tasks.base import Task

_TASKS: dict[str, type["Task"]] = {}

T = TypeVar("T", bound="Task")


def register_task(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator 把 Task 子类登记进 _TASKS，key 就是 CLI 里 --task 的参数值."""

    def deco(cls: type[T]) -> type[T]:
        if name in _TASKS:
            raise ValueError(f"task already registered: {name!r}")
        _TASKS[name] = cls
        return cls

    return deco


def get_task(name: str) -> "Task":
    """字符串 → 实例化的 Task."""
    if name not in _TASKS:
        raise KeyError(
            f"unknown task: {name!r}; known = {sorted(_TASKS)}"
        )
    return _TASKS[name]()


def list_tasks() -> list[str]:
    """所有已注册 task 名（CLI list-tasks 用）."""
    return sorted(_TASKS)
