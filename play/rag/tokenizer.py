from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

from config import EMBED_TOKENIZER


@lru_cache(maxsize=4)
def _tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(name)


def tokenize(text: str, name: str | None = None) -> list[str]:
    encoded = _tokenizer(name or EMBED_TOKENIZER).encode(text)
    out: list[str] = []
    for t in encoded.tokens:
        if t.startswith("<") and t.endswith(">"):
            continue
        normalized = t.lstrip("Ġ▁").lower()
        if normalized:
            out.append(normalized)
    return out
