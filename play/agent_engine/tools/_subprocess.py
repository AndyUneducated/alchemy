"""Subprocess helper for tools that delegate to other plays via JSON envelope.

Used today only by ``retrieve_docs`` (subprocess to ``play/rag/query.py``).
Reserved env var name ``WORKFLOW_TRACEPARENT`` is **not** injected today;
plan section 9.1 reserves it for future W3C trace context propagation.
"""

from __future__ import annotations

import json
import subprocess
import sys


def run_json_subprocess(cmd: list[str]) -> tuple[int, dict | None]:
    """Run *cmd* with stdout piped, stderr inherited.

    Returns ``(returncode, parsed_payload | None)``. JSON parse failure or
    non-zero exit code surfaces as ``(rc, None)`` so callers can decide how
    to wrap the error.
    """
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return result.returncode, None
    try:
        return 0, json.loads(result.stdout)
    except (ValueError, TypeError):
        return 0, None
