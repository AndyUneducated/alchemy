"""HuggingFace tokenizer wrapper for BM25 ingestion / query.

Reuses the embedding model's tokenizer so BM25 and dense retrieval share the
same lexical view. Only `tokenizer.json` (~10MB) is downloaded; model weights
are not pulled.
"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

from config import EMBED_TOKENIZER


@lru_cache(maxsize=4)
def _tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(name)


def tokenize(text: str, name: str | None = None) -> list[str]:
    """Tokenize *text* into a list of normalized subword strings for BM25.

    *name* lets the query side pass the tokenizer recorded in the VDB's
    metadata.json (sentinel pattern). Defaults to `EMBED_TOKENIZER` for
    ingestion.
    """
    encoded = _tokenizer(name or EMBED_TOKENIZER).encode(text)
    out: list[str] = []
    for t in encoded.tokens:
        if t.startswith("<") and t.endswith(">"):
            continue
        normalized = t.lstrip("Ġ▁").lower()
        if normalized:
            out.append(normalized)
    return out
