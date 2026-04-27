from __future__ import annotations

import json
import sys


def is_error(result: str) -> bool:
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def warn_if_error(name: str, result: str) -> None:
    if not is_error(result):
        return
    payload = json.loads(result)
    first_line = str(payload["error"]).splitlines()[0]
    print(f"WARNING: tool {name} failed: {first_line}",
          file=sys.stderr, flush=True)
