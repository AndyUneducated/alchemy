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
