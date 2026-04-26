"""Path-access variable interpolation for workflow.yaml strings.

Supports **only** dotted path lookups; no filters / no expressions / no
conditionals (plan §4 / §12). Keeps the "template language" surface area
under ~50 lines so it can never grow into a Jinja2-lookalike.

Resolution rules:

- ``"{{ x.y.z }}"`` (whole-string) — looks up ``state['x']['y']['z']`` and
  returns the **raw** value, preserving its Python type (list / dict / int /
  str / etc.). This is how ``args: {requirements: "{{ stages.load.output }}"}``
  passes a ``list[dict]`` through.
- ``"prefix {{ x.y }} suffix"`` (inline) — every match is substituted as
  ``str(value)`` and the result is always a string.
- Other types (dict / list / scalar) are recursed / passed through.

Missing path → ``KeyError`` (plan §12 fail-fast: no friendly hints).
"""

from __future__ import annotations

import re
from typing import Any

VAR_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def _lookup(path: str, state: dict[str, Any]) -> Any:
    """Walk dotted *path* through *state*; raise KeyError on miss."""
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
    """Recursively substitute ``{{ x.y.z }}`` references in *value*.

    Strings entirely matching one ``{{ ... }}`` keep the looked-up value's
    native type; partial matches stringify via ``str()``. Dicts / lists are
    recursed; other types pass through unchanged.
    """
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
