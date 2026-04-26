"""Tool tracer: collect non-artifact tool-call events across one Discussion.

Used by ``Discussion._run_turn``: events drained after each turn and appended
to ``Discussion.history`` with ``visible=False`` so other agents skip them
in memory projection (only ``Result.transcript`` / ``--save-transcript``
exposes them for replay).

Field names mirror OpenTelemetry GenAI semantic conventions
(``gen_ai.tool.name`` / ``.call.arguments`` / ``.call.response``). We borrow
the naming only — no SDK dependency, no spans, no exporter.
"""

from __future__ import annotations

import json
import sys
import time

from .tools import is_error


def _preview_args(arguments: dict) -> str:
    """Render a short k=v, k=v summary of tool arguments for the terminal."""
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
    """Render a short summary of a tool result for the terminal."""
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        flat = result.replace("\n", " ").strip()
        return flat if len(flat) <= 60 else flat[:57] + "..."
    if not ok and isinstance(payload, dict) and "error" in payload:
        first = str(payload["error"]).splitlines()[0]
        return f"error: {first}"
    if isinstance(payload, dict):
        # retrieve_docs returns {data, meta:{mode, reranked, top_k}}; surface
        # the retrieval path so workshop viewers can see which strategy ran.
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
    """Collect non-artifact tool-call events across one Discussion.

    Events are drained by ``Discussion._run_turn`` after each turn and
    appended to ``Discussion.history`` with ``visible=False``.
    """

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
