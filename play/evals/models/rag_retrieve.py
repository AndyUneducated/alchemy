"""RAG retrieval closure factory: subprocess call `play/rag/query.py --json`, zero Python import.

Why subprocess instead of directly `from play.rag.query import search`:
  - Follow the monorepo decoupling principle (see DECISIONS §4/workshops.mdc for details):
    Sub-projects under `play/` do not import Python from each other, and cross-project communication uses CLI + JSON envelope.
  - `play/rag`’s built-in dependencies (chromadb / ollama / fastparquet, etc.) do not pollute the `evals` process
  - Smooth migration of future remote retriever (HTTP service) on the same set of interfaces: change transport
    Implementation without moving the task layer

Cost & Mitigation:
  - Cold boot ~2-4s (python + chromadb client + ollama embed loaded). phase 4 8 queries
    Running in turn takes about 16-32s - acceptable. Batch optimization (one subprocess for multiple queries) leaves phase 5+
  - Error propagation: expose stderr when subprocess.CalledProcessError (OllamaConnError /
    If VDB does not exist, you can see it immediately on the evals side)

Data contract:
  - retrieve_fn(query: str) -> tuple[list[source_id], list[content]]
    Where `source_id` = play/rag/ingest is written into the basename of chunk metadata['source'],
    Semantically aligned with `data/rag_retrieval/gold.jsonl::gold_doc_ids` field"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_DIR = REPO_ROOT / "play" / "rag"
RAG_QUERY_SCRIPT = RAG_DIR / "query.py"

SearchMode = Literal["dense", "bm25", "hybrid"]

RetrieveFn = Callable[[str], tuple[list[str], list[str]]]


def make_retrieve_fn(
    vdb_dir: str | Path,
    *,
    top_k: int = 5,
    mode: SearchMode = "hybrid",
    rerank: bool = False,
    timeout: float = 60.0,
) -> RetrieveFn:
    """Returns a `(query: str) -> (ids, contents)` closure.

    Each call forks a subprocess:
      `python play/rag/query.py --vdb <vdb_dir> --query <q> --top-k K --mode hybrid --json [--rerank]`

    Parse the JSON envelope on stdout (see play/rag/query.py::main for schema):
      `{"query": ..., "data": [{"content", "score", "source", "metadata"}], "meta": {...}}`

    Go to `data[*].source` (chunk source file name) as retrieval unit;
    When multiple chunks have the same origin, the first appearing position will be retained for deduplication (the higher the rank, the higher the priority)."""
    vdb_path = Path(vdb_dir).resolve()

    def _retrieve(query: str) -> tuple[list[str], list[str]]:
        cmd = [
            sys.executable, str(RAG_QUERY_SCRIPT),
            "--vdb", str(vdb_path),
            "--query", query,
            "--top-k", str(top_k),
            "--mode", mode,
            "--json",
        ]
        if rerank:
            cmd.append("--rerank")

        proc = subprocess.run(
            cmd,
            cwd=str(RAG_DIR),  # play/rag/query.py uses relative import config / bm25 etc.
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"play/rag/query.py exited with {proc.returncode}; "
                f"stderr={proc.stderr.strip()!r}"
            )
        envelope = json.loads(proc.stdout)
        hits = envelope.get("data", [])

        # Deduplication of homologous chunks, rank priority (retaining the first position)
        seen: set[str] = set()
        ids: list[str] = []
        contents: list[str] = []
        for hit in hits:
            src = hit.get("source", "")
            if not src or src in seen:
                continue
            seen.add(src)
            ids.append(src)
            contents.append(hit.get("content", ""))
        return ids, contents

    return _retrieve
