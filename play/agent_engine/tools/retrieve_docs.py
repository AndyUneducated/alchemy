from __future__ import annotations

import json
import os
import sys

from ._subprocess import run_json_subprocess


_QUERY_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "rag", "query.py",
)


TOOL_DEF: dict = {
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
    # Non-OpenAI-standard hint read by scenario.py: parameter names whose
    # scenario-level default values are filesystem paths, so scenario.py
    # resolves relative paths against the scenario file's directory.
    "_path_params": {"vdb_dir"},
}


def handler(
    query: str,
    vdb_dir: str,
    top_k: int = 3,
    mode: str = "hybrid",
    rerank: bool = False,
) -> str:
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
    rc, payload = run_json_subprocess(cmd)
    if rc != 0:
        return json.dumps({"error": f"retrieve_docs exited with code {rc}"})
    if payload is None:
        return json.dumps({"error": "retrieve_docs returned non-JSON output"})
    # rag CLI emits {query, data, meta} for evolution headroom (pagination,
    # timing, version). The LLM only needs the hits and a self-observable
    # tag of which retrieval path ran — strip the rest at the tool boundary,
    # like an SDK unwrapping an HTTP envelope.
    slim = {
        "data": payload["data"],
        "meta": {
            "mode": payload["meta"]["mode"],
            "reranked": payload["meta"]["reranked"],
            "top_k": payload["meta"]["top_k"],
        },
    }
    return json.dumps(slim, ensure_ascii=False)
