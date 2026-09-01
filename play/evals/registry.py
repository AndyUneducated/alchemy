"""Task registry: decorator + dict.

`@register_task("name")` registers at class definition time, avoiding dirty state
where the task changes but the registry is not updated.
Trade-off: import-time side effect — `tasks/__init__.py` explicitly
`from . import sentiment_clf` to trigger decorators so the CLI can see tasks.

Same-family pattern: Django URLs, Flask/FastAPI routes, pytest fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .tasks.base import Task

_TASKS: dict[str, type["Task"]] = {}

T = TypeVar("T", bound="Task")


def register_task(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registers Task subclass in _TASKS; key is the CLI --task value."""

    def deco(cls: type[T]) -> type[T]:
        if name in _TASKS:
            raise ValueError(f"task already registered: {name!r}")
        _TASKS[name] = cls
        return cls

    return deco


def get_task(name: str) -> "Task":
    """string → instantiated Task."""
    if name not in _TASKS:
        raise KeyError(
            f"unknown task: {name!r}; known = {sorted(_TASKS)}"
        )
    return _TASKS[name]()


def list_tasks() -> list[str]:
    """All registered task names (for CLI list-tasks)."""
    return sorted(_TASKS)
