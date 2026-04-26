"""BM25 search + Reciprocal Rank Fusion primitives for hybrid retrieval.

Pure functions over (already-tokenized) inputs. The query-side tokenizer name
is read from the VDB's metadata.json by `query.py` (sentinel pattern), so this
module never touches HF or config directly.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache

from config import RRF_K


@lru_cache(maxsize=4)
def _load_bm25(vdb_dir: str):
    """Load the pickled BM25 index for *vdb_dir*; cached per VDB across calls."""
    bm25_path = os.path.join(vdb_dir, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def dense_search(collection, query_text: str, k: int) -> list[tuple[str, float]]:
    """Return ordered (chroma_id, similarity) pairs from a Chroma collection."""
    res = collection.query(query_texts=[query_text], n_results=k)
    ids = res.get("ids", [[]])[0]
    distances = res.get("distances", [[]])[0]
    return [(i, 1.0 / (1.0 + d)) for i, d in zip(ids, distances)]


def bm25_search(
    vdb_dir: str, query_tokens: list[str], k: int
) -> list[tuple[str, float]]:
    """Return ordered (chroma_id, bm25_score) pairs from the persisted index."""
    index = _load_bm25(vdb_dir)
    ids: list[str] = index["ids"]
    model = index["model"]

    scores = model.get_scores(query_tokens)
    pairs = sorted(zip(ids, scores), key=lambda x: -x[1])
    return [(i, float(s)) for i, s in pairs[:k]]


def rrf_fuse(
    *rankings: list[tuple[str, float]],
    k_top: int,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Only ranks (list positions) are used; the input
    score values are ignored — they're kept in the signature so callers can
    pass `dense_search` / `bm25_search` outputs verbatim.

    Returns the top *k_top* (id, rrf_score) pairs, score-descending.
    """
    rrf: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    fused = sorted(rrf.items(), key=lambda x: -x[1])
    return fused[:k_top]
