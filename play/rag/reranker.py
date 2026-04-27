from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

if TYPE_CHECKING:
    from query import SearchResult


@lru_cache(maxsize=1)
def _model(name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(name)


def rerank(
    query_text: str, hits: list["SearchResult"], top_k: int,
) -> list["SearchResult"]:
    if not hits:
        return []

    pairs = [(query_text, h["content"]) for h in hits]
    scores = _model().predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])

    return [
        {
            **h,
            "score": float(s),
            "metadata": {**h["metadata"], "reranked": True},
        }
        for h, s in ranked[:top_k]
    ]
