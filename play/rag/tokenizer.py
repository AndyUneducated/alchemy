from __future__ import annotations

from functools import lru_cache
import re

from tokenizers import Tokenizer

from config import EMBED_TOKENIZER

BASIC_TOKENIZER_NAME = "basic"
BASIC_TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _basic_tokenize(text: str) -> list[str]:
    return [token.lower() for token in BASIC_TOKEN_PATTERN.findall(text)]


@lru_cache(maxsize=4)
def _tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(name)


def tokenize(text: str, name: str | None = None) -> list[str]:
    tokenizer_name = name or EMBED_TOKENIZER
    if tokenizer_name == BASIC_TOKENIZER_NAME:
        return _basic_tokenize(text)

    encoded = _tokenizer(tokenizer_name).encode(text)
    out: list[str] = []
    for t in encoded.tokens:
        if t.startswith("<") and t.endswith(">"):
            continue
        normalized = t.lstrip("Ġ▁").lower()
        if normalized:
            out.append(normalized)
    return out
