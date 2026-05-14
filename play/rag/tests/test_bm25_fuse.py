"""dense_search / bm25_search / rrf_fuse 三个纯函数的不变量。

DECISIONS §4 的核心 claim：
  - `dense_search` 把 ChromaDB 距离折算为"越大越相似"的 score = 1/(1+d)
  - `bm25_search` 返回 (id, score) 按 score 降序的 top-k
  - `rrf_fuse` 只用排名不用 score（k=60 默认）；`1/(rrf_k + rank + 1)`

这些是 hybrid retrieval 的代数核心，都是 off-by-one 高危区。
所有测试都用 fake collection / fake bm25 model，不碰真实 chroma / pickle。
"""
from __future__ import annotations

import bm25 as bm25_module
from bm25 import bm25_search, dense_search, rrf_fuse


# ---------- dense_search -------------------------------------------------

class _FakeChromaCollection:
    def __init__(self, ids: list[str], distances: list[float]):
        self._ids = ids
        self._distances = distances

    def query(self, query_texts, n_results):
        return {
            "ids": [self._ids[:n_results]],
            "distances": [self._distances[:n_results]],
        }


def test_dense_search_converts_distance_to_similarity_monotonically():
    coll = _FakeChromaCollection(
        ids=["a", "b", "c"], distances=[0.0, 1.0, 3.0],
    )
    out = dense_search(coll, "q", k=3)

    assert out == [("a", 1.0), ("b", 0.5), ("c", 0.25)]
    assert out[0][1] > out[1][1] > out[2][1], (
        "score = 1/(1+distance) must preserve 'larger = more similar'"
    )


def test_dense_search_respects_k():
    coll = _FakeChromaCollection(
        ids=[f"id{i}" for i in range(10)],
        distances=[float(i) for i in range(10)],
    )
    out = dense_search(coll, "q", k=3)
    assert len(out) == 3


# ---------- bm25_search --------------------------------------------------

class _FakeBM25Model:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def get_scores(self, tokens):
        return self._scores


def test_bm25_search_returns_top_k_in_descending_score(monkeypatch):
    monkeypatch.setattr(
        bm25_module, "_load_bm25",
        lambda vdb_dir: {
            "ids": ["a", "b", "c", "d"],
            "model": _FakeBM25Model([0.1, 0.5, 0.3, 0.9]),
        },
    )

    out = bm25_search("/fake/vdb", ["q"], k=2)

    assert [doc for doc, _ in out] == ["d", "b"], "must sort by score desc"
    assert out[0][1] >= out[1][1]
    assert len(out) == 2


def test_bm25_search_truncates_to_k(monkeypatch):
    monkeypatch.setattr(
        bm25_module, "_load_bm25",
        lambda vdb_dir: {
            "ids": [f"x{i}" for i in range(5)],
            "model": _FakeBM25Model([float(i) for i in range(5)]),
        },
    )
    assert len(bm25_search("/fake/vdb", ["q"], k=3)) == 3


# ---------- rrf_fuse -----------------------------------------------------

def test_rrf_fuse_no_rankings_returns_empty():
    assert rrf_fuse(k_top=5) == []
    assert rrf_fuse([], [], k_top=5) == []


def test_rrf_fuse_single_ranking_preserves_order():
    ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = rrf_fuse(ranking, k_top=2)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_rrf_fuse_top_in_both_outranks_top_in_one():
    """If a doc appears at rank 0 in both rankings, its RRF score is the sum of
    two top-rank reciprocals; a doc appearing only at rank 0 of one ranking
    can at best add a much smaller second-ranking term — so the shared doc
    must rank strictly higher."""
    r_a = [("shared", 1.0), ("only_a", 0.9), ("only_b", 0.0)]
    r_b = [("shared", 1.0), ("only_b", 0.9), ("only_a", 0.0)]

    fused = rrf_fuse(r_a, r_b, k_top=10)
    rank = {doc: i for i, (doc, _) in enumerate(fused)}

    assert rank["shared"] < rank["only_a"]
    assert rank["shared"] < rank["only_b"]


def test_rrf_fuse_ignores_input_scores_uses_rank_only():
    """RRF's defining property: input score magnitudes are irrelevant; only the
    ordinal rank matters. Two docs symmetric across two rankings must tie."""
    r_a = [("x", 999.0), ("y", 1.0)]
    r_b = [("y", 999.0), ("x", 1.0)]

    fused = dict(rrf_fuse(r_a, r_b, k_top=10))
    assert fused["x"] == fused["y"], (
        "RRF must depend only on rank, not on the raw scores from each ranking"
    )


def test_rrf_fuse_truncates_to_k_top():
    ranking = [(str(i), float(-i)) for i in range(10)]
    fused = rrf_fuse(ranking, k_top=3)
    assert len(fused) == 3


def test_rrf_fuse_score_formula_matches_reciprocal_rank():
    """Doc only appears at rank 0 of one ranking → RRF score = 1/(60+0+1)."""
    fused = dict(rrf_fuse([("only", 0.0)], k_top=1))
    assert abs(fused["only"] - 1.0 / 61.0) < 1e-12
