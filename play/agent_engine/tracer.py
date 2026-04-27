from __future__ import annotations

import json
import sys
import time

from .tools import is_error


def _preview_args(arguments: dict) -> str:
    parts: list[str] = []
    for k, v in arguments.items():
        if isinstance(v, str):
            s = v if len(v) <= 40 else v[:37] + "..."
            parts.append(f"{k}={s!r}")
        else:
            s = repr(v)
            if len(s) > 40:
                s = s[:37] + "..."
            parts.append(f"{k}={s}")
    return ", ".join(parts)


def _preview_result(result: str, ok: bool) -> str:
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        flat = result.replace("\n", " ").strip()
        return flat if len(flat) <= 60 else flat[:57] + "..."
    if not ok and isinstance(payload, dict) and "error" in payload:
        first = str(payload["error"]).splitlines()[0]
        return f"error: {first}"
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list) and isinstance(payload.get("meta"), dict):
            n = len(payload["data"])
            m = payload["meta"]
            tags = [f"mode={m.get('mode')}"]
            if m.get("reranked"):
                tags.append("reranked")
            return f"[{n} items, " + ", ".join(tags) + "]"
        if isinstance(payload.get("results"), list):
            return f"{{results: {len(payload['results'])}}}"
        if "count" in payload:
            return f"{{count: {payload['count']}}}"
        keys = list(payload.keys())
        if len(keys) <= 3:
            return "{" + ", ".join(keys) + "}"
        return "{" + ", ".join(keys[:3]) + ", ...}"
    if isinstance(payload, list):
        return f"[{len(payload)} items]"
    flat = str(payload)
    return flat if len(flat) <= 60 else flat[:57] + "..."


class ToolTracer:
    def __init__(self) -> None:
        self._events: list[dict] = []

    def record(self, caller: str, tool: str, arguments: dict, result: str) -> None:
        ok = not is_error(result)
        print(
            f"🔧 [{caller}] {tool}({_preview_args(arguments)}) "
            f"→ {_preview_result(result, ok)}",
            file=sys.stderr, flush=True,
        )
        self._events.append({
            "type": "tool_call",
            "caller": caller,
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "ok": ok,
            "visible": False,
            "ts": time.time(),
        })

    def drain(self) -> list[dict]:
        events, self._events = self._events, []
        return events
