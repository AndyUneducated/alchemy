"""Shared rag test configuration.

Modules under `play/rag/` use bare imports (`from chunker import ...` / `from bm25 import ...`),
not a package. This adds `play/rag/` itself to sys.path so tests under `tests/` can
`from chunker import split_text` without path gymnastics.

The suite stays lightweight: no chromadb / ollama / VDB / HF cache — non–pure-function
paths are monkeypatched. CLI contract tests only statically read `query.py` text.
"""
