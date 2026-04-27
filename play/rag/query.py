#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import (
    EMBED_MODEL,
    EMBED_TOKENIZER,
    HYBRID_OVERSAMPLE,
    OLLAMA_BASE_URL,
    RERANK_CANDIDATES,
)
from bm25 import bm25_search, dense_search, rrf_fuse
from reranker import rerank as do_rerank
from tokenizer import tokenize


SearchMode = Literal["dense", "bm25", "hybrid"]


class SearchResult(TypedDict):
    content: str
    score: float
    source: str
    metadata: dict


def _load_meta(vdb_dir: str) -> dict:
    meta_path = os.path.join(vdb_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _materialize(
    collection, scored_ids: list[tuple[str, float]], mode: SearchMode,
) -> list[SearchResult]:
    if not scored_ids:
        return []

    ids = [i for i, _ in scored_ids]
    res = collection.get(ids=ids, include=["documents", "metadatas"])

    by_id = {
        i: (doc, meta)
        for i, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    }

    hits: list[SearchResult] = []
    for chunk_id, score in scored_ids:
        if chunk_id not in by_id:
            continue
        doc, meta = by_id[chunk_id]
        meta = dict(meta) if meta else {}
        meta["retrieval"] = mode
        meta["reranked"] = False
        hits.append(SearchResult(
            content=doc,
            score=float(score),
            source=meta.get("source", ""),
            metadata=meta,
        ))
    return hits


def search(
    vdb_dir: str,
    query_text: str,
    *,
    top_k: int = 5,
    mode: SearchMode = "hybrid",
    rerank: bool = False,
    model: str | None = None,
    collection_name: str | None = None,
) -> list[SearchResult]:
    bm25_path = os.path.join(vdb_dir, "bm25.pkl")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"{bm25_path} missing — re-run ingest to rebuild the BM25 index."
        )

    meta = _load_meta(vdb_dir)
    stored_model = meta.get("embedding_model", "")
    effective_model = model or stored_model or EMBED_MODEL
    if model and stored_model and model != stored_model:
        print(
            f"WARNING: requested model '{model}' differs from VDB model "
            f"'{stored_model}'. Results may be meaningless.",
            file=sys.stderr,
        )

    stored_tokenizer = meta.get("tokenizer") or EMBED_TOKENIZER

    ef = OllamaEmbeddingFunction(url=OLLAMA_BASE_URL, model_name=effective_model)
    client = chromadb.PersistentClient(path=vdb_dir)
    collections = client.list_collections()
    if not collections:
        raise FileNotFoundError(f"No collections found in {vdb_dir}")

    if collection_name:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
    else:
        collection = client.get_collection(
            name=collections[0].name, embedding_function=ef
        )
        if len(collections) > 1:
            print(
                f"Multiple collections found; using '{collection.name}'. "
                f"Use --collection to specify.",
                file=sys.stderr,
            )

    retrieve_k = RERANK_CANDIDATES if rerank else top_k
    pool_k = retrieve_k * HYBRID_OVERSAMPLE

    if mode == "dense":
        scored = dense_search(collection, query_text, retrieve_k)
    elif mode == "bm25":
        query_tokens = tokenize(query_text, name=stored_tokenizer)
        scored = bm25_search(vdb_dir, query_tokens, retrieve_k)
    elif mode == "hybrid":
        query_tokens = tokenize(query_text, name=stored_tokenizer)
        dense = dense_search(collection, query_text, pool_k)
        lexical = bm25_search(vdb_dir, query_tokens, pool_k)
        scored = rrf_fuse(dense, lexical, k_top=retrieve_k)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    hits = _materialize(collection, scored, mode)

    if rerank:
        hits = do_rerank(query_text, hits, top_k=top_k)

    return hits


def query(
    vdb_dir: str,
    query_text: str,
    *,
    top_k: int = 5,
    mode: SearchMode = "hybrid",
    rerank: bool = False,
    model: str | None = None,
    collection_name: str | None = None,
) -> None:
    hits = search(
        vdb_dir, query_text,
        top_k=top_k, mode=mode, rerank=rerank,
        model=model, collection_name=collection_name,
    )

    rerank_tag = " +rerank" if rerank else ""
    print(f"\nQuery: {query_text}")
    print(f"Top {top_k} results (mode={mode}{rerank_tag})\n")

    for i, hit in enumerate(hits, 1):
        chunk_idx = hit["metadata"].get("chunk_index", "?")
        print(
            f"--- [{i}] source={hit['source']}  chunk={chunk_idx}  "
            f"score={hit['score']:.4f} ---"
        )
        print(hit["content"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid similarity search over a ChromaDB VDB produced by ingest.py. "
            "Defaults to human-readable output; --json switches to machine mode "
            "(JSON envelope on stdout, warnings on stderr).\n"
            "Typical usage: python query.py --vdb vdb/panel --query \"cash flow\" --top-k 5"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vdb", required=True,
        help="VDB directory (output of ingest --output); metadata.json is read "
             "at startup to pick the embedding model and BM25 tokenizer",
    )
    parser.add_argument(
        "--query", required=True,
        help="query text; Chinese / English / mixed all supported",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="return the top-N most similar chunks; clamped by ChromaDB if it "
             "exceeds the total chunk count",
    )
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "hybrid"], default="hybrid",
        help="retrieval strategy; 'hybrid' (default) fuses dense + BM25 via "
             "RRF. 'dense' / 'bm25' are diagnostic-only single-retriever paths",
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "explicitly override the embedding model; by default reuses the "
            "stored model from VDB metadata.json. A mismatch only logs a "
            "WARNING to stderr (does not fail)"
        ),
    )
    parser.add_argument(
        "--collection", default=None,
        help=(
            "collection name; optional for single-collection VDBs; for "
            "multi-collection VDBs without this flag, the first one is used "
            "and a notice is printed to stderr"
        ),
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help=(
            "enable cross-encoder reranking: retrieve a wider candidate pool "
            "then re-score for higher precision; ~1.2GB model loaded lazily "
            "on first use"
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help=(
            "machine-consumption mode: stdout emits a JSON envelope "
            "{query, data, meta} aligned with OpenAI / Pinecone / Cohere "
            "common subset; warnings still go to stderr"
        ),
    )
    args = parser.parse_args()

    if args.json:
        hits = search(
            args.vdb, args.query,
            top_k=args.top_k, mode=args.mode, rerank=args.rerank,
            model=args.model, collection_name=args.collection,
        )
        meta = _load_meta(args.vdb)
        envelope = {
            "query": args.query,
            "data": hits,
            "meta": {
                "vdb": args.vdb,
                "mode": args.mode,
                "reranked": args.rerank,
                "top_k": args.top_k,
                "embedding_model": (
                    args.model or meta.get("embedding_model") or EMBED_MODEL
                ),
            },
        }
        print(json.dumps(envelope, ensure_ascii=False))
    else:
        query(
            args.vdb, args.query,
            top_k=args.top_k, mode=args.mode, rerank=args.rerank,
            model=args.model, collection_name=args.collection,
        )


if __name__ == "__main__":
    main()
