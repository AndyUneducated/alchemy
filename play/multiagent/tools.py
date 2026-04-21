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
            "description": "Search a vector database for relevant document chunks.",
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


def _retrieve_docs(query: str, vdb_dir: str, top_k: int = 3) -> str:
    # Only pipe stdout (we need the JSON). stderr defaults to parent's stderr,
    # so any subprocess error (traceback, warnings) flows to the terminal for free.
    result = subprocess.run(
        [
            sys.executable, _QUERY_SCRIPT,
            "--vdb", vdb_dir,
            "--query", query,
            "--top-k", str(top_k),
            "--json",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return json.dumps({"error": f"retrieve_docs exited with code {result.returncode}"})
    return result.stdout.strip()


TOOL_HANDLERS["retrieve_docs"] = _retrieve_docs


def dispatch(name: str, arguments: dict) -> str:
    """Look up *name* in the registry and call the handler with *arguments*."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = json.dumps({"error": f"Unknown tool: {name}"})
    else:
        result = handler(**arguments)
    # Catch-all: any handler that returns {"error": ...} JSON gets a one-line
    # notice on stderr, so silent failures (e.g. RAG returning a canned error
    # string the model then covers up) are impossible to miss.
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "error" in payload:
        first_line = str(payload["error"]).splitlines()[0]
        print(f"WARNING: tool {name} failed: {first_line}",
              file=sys.stderr, flush=True)
    return result
