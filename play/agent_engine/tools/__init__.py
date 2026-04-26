"""Tool registry: aggregates per-tool TOOL_DEF + handler into the public API.

Public surface re-exports keep callers (``run.py``, ``artifact.py``, future
workflow executors) using ``from tools import TOOL_DEFINITIONS, dispatch,
is_error, warn_if_error`` — same as the old single-file module.

Adding a new tool: drop a ``<name>.py`` defining ``TOOL_DEF`` + ``handler``
beside this file, then append two lines below. No decorator magic — explicit
aggregation makes scenario YAML schema validation easier to debug.
"""

from __future__ import annotations

import json
from typing import Callable

from . import retrieve_docs
from ._envelope import is_error, warn_if_error


TOOL_DEFINITIONS: list[dict] = [
    retrieve_docs.TOOL_DEF,
]

TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "retrieve_docs": retrieve_docs.handler,
}


def dispatch(name: str, arguments: dict) -> str:
    """Look up *name* in the registry and call the handler with *arguments*."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = json.dumps({"error": f"Unknown tool: {name}"})
    else:
        result = handler(**arguments)
    warn_if_error(name, result)
    return result


__all__ = [
    "TOOL_DEFINITIONS",
    "TOOL_HANDLERS",
    "dispatch",
    "is_error",
    "warn_if_error",
]
