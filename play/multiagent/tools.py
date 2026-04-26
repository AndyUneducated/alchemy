"""Tool registry: definitions (OpenAI format) + handlers for agent tool-use."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Callable

_QUERY_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "rag", "query.py")

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": (
                "Search a vector database for relevant document chunks. "
                "Default 'hybrid' mode (dense + BM25 fused via RRF) handles "
                "most queries well; flip 'rerank' on for ambiguous or "
                "semantically tricky queries to trade ~5s latency for higher "
                "precision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text.",
                    },
                    "vdb_dir": {
                        "type": "string",
                        "description": "Path to the VDB directory.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 3,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["dense", "bm25", "hybrid"],
                        "description": (
                            "Retrieval strategy. 'hybrid' (default) is the "
                            "strongest general-purpose option. 'dense' / "
                            "'bm25' are diagnostic-only single-retriever "
                            "paths — pick them only when comparing strategies."
                        ),
                        "default": "hybrid",
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": (
                            "Enable cross-encoder reranking for higher "
                            "precision on ambiguous queries. First call "
                            "lazily loads a ~1.2GB model (~5s); subsequent "
                            "calls add ~1-3s per query. Default false."
                        ),
                        "default": False,
                    },
                },
                "required": ["query", "vdb_dir"],
            },
        },
        # Non-OpenAI-standard hint read by run.py: parameter names whose
        # scenario-level default values are filesystem paths, so run.py should
        # resolve relative paths against the scenario file's directory.
        "_path_params": {"vdb_dir"},
    },
]

TOOL_HANDLERS: dict[str, Callable[..., str]] = {}


def _retrieve_docs(
    query: str,
    vdb_dir: str,
    top_k: int = 3,
    mode: str = "hybrid",
    rerank: bool = False,
) -> str:
    # Only pipe stdout (we need the JSON). stderr defaults to parent's stderr,
    # so any subprocess error (traceback, warnings) flows to the terminal for free.
    cmd = [
        sys.executable, _QUERY_SCRIPT,
        "--vdb", vdb_dir,
        "--query", query,
        "--top-k", str(top_k),
        "--mode", mode,
        "--json",
    ]
    if rerank:
        cmd.append("--rerank")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return json.dumps({"error": f"retrieve_docs exited with code {result.returncode}"})
    # rag CLI emits {query, data, meta} for evolution headroom (pagination,
    # timing, version). The LLM only needs the hits and a self-observable
    # tag of which retrieval path ran — strip the rest at the tool boundary,
    # like an SDK unwrapping an HTTP envelope.
    payload = json.loads(result.stdout)
    slim = {
        "data": payload["data"],
        "meta": {
            "mode": payload["meta"]["mode"],
            "reranked": payload["meta"]["reranked"],
            "top_k": payload["meta"]["top_k"],
        },
    }
    return json.dumps(slim, ensure_ascii=False)


TOOL_HANDLERS["retrieve_docs"] = _retrieve_docs


def is_error(result: str) -> bool:
    """Return True if *result* parses as a JSON object with an ``error`` key.

    Shared by :func:`warn_if_error` (stderr surface) and the tool tracer
    (to stamp ``ok`` on recorded events), so both agree on what "failed" means.
    """
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and "error" in payload


def warn_if_error(name: str, result: str) -> None:
    """Print a one-line stderr notice when *result* is ``{"error": ...}`` JSON.

    Extracted so other dispatchers (e.g. ``ArtifactStore.dispatch``) can share
    the same silent-failure catch-all used by :func:`dispatch`.
    """
    if not is_error(result):
        return
    payload = json.loads(result)
    first_line = str(payload["error"]).splitlines()[0]
    print(f"WARNING: tool {name} failed: {first_line}",
          file=sys.stderr, flush=True)


def dispatch(name: str, arguments: dict) -> str:
    """Look up *name* in the registry and call the handler with *arguments*."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = json.dumps({"error": f"Unknown tool: {name}"})
    else:
        result = handler(**arguments)
    warn_if_error(name, result)
    return result
