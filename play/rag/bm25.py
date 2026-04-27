from __future__ import annotations

import os
import pickle
from functools import lru_cache

from config import RRF_K


@lru_cache(maxsize=4)
def _load_bm25(vdb_dir: str):
    bm25_path = os.path.join(vdb_dir, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def dense_search(collection, query_text: str, k: int) -> list[tuple[str, float]]:
    res = collection.query(query_texts=[query_text], n_results=k)
    ids = res.get("ids", [[]])[0]
    distances = res.get("distances", [[]])[0]
    return [(i, 1.0 / (1.0 + d)) for i, d in zip(ids, distances)]


def bm25_search(
    vdb_dir: str, query_tokens: list[str], k: int
) -> list[tuple[str, float]]:
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
    rrf: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    fused = sorted(rrf.items(), key=lambda x: -x[1])
    return fused[:k_top]
