from __future__ import annotations

import re
from typing import Any

VAR_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _lookup(path: str, state: dict[str, Any]) -> Any:
    parts = path.split(".")
    cur: Any = state
    for p in parts:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            raise KeyError(
                f"workflow path '{path}' hit non-dict at segment '{p}' "
                f"(got {type(cur).__name__})"
            )
    return cur


def interpolate(value: Any, state: dict[str, Any]) -> Any:
    if isinstance(value, str):
        m = VAR_RE.fullmatch(value.strip())
        if m and value.strip() == value:
            return _lookup(m.group(1), state)
        return VAR_RE.sub(lambda mm: str(_lookup(mm.group(1), state)), value)
    if isinstance(value, dict):
        return {k: interpolate(v, state) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, state) for v in value]
    return value
