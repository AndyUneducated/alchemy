"""Shared helpers for tool result envelopes.

A tool's "error" is canonical JSON ``{"error": "..."}`` — both the dispatcher
and ``ArtifactStore.dispatch`` agree on this so stderr surfacing and tracer
``ok`` stamping stay consistent.
"""

from __future__ import annotations

import json
import sys


def is_error(result: str) -> bool:
    """Return True if *result* parses as a JSON object with an ``error`` key."""
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def warn_if_error(name: str, result: str) -> None:
    """Print a one-line stderr notice when *result* is ``{"error": ...}`` JSON."""
    if not is_error(result):
        return
    payload = json.loads(result)
    first_line = str(payload["error"]).splitlines()[0]
    print(f"WARNING: tool {name} failed: {first_line}",
          file=sys.stderr, flush=True)
