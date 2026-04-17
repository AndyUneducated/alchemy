#!/usr/bin/env python3
"""CLI tool & programmatic API: query a ChromaDB VDB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TypedDict

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import EMBED_MODEL, OLLAMA_BASE_URL


class SearchResult(TypedDict):
    """Provider-agnostic retrieval result (not tied to ChromaDB)."""
    content: str
    score: float
    source: str
    metadata: dict


def search(
    vdb_dir: str,
    query_text: str,
    *,
    top_k: int = 5,
    model: str | None = None,
    collection_name: str | None = None,
) -> list[SearchResult]:
    """Return the *top_k* most similar chunks as structured dicts."""
    meta_path = os.path.join(vdb_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stored_model = meta.get("embedding_model", "")
        effective_model = model or stored_model or EMBED_MODEL
        if model and model != stored_model:
            print(
                f"WARNING: requested model '{model}' differs from VDB model "
                f"'{stored_model}'. Results may be meaningless.",
                file=sys.stderr,
            )
    else:
        effective_model = model or EMBED_MODEL

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

    results = collection.query(query_texts=[query_text], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        SearchResult(
            content=doc,
            score=1.0 / (1.0 + dist),
            source=meta.get("source", ""),
            metadata=meta,
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def query(
    vdb_dir: str,
    query_text: str,
    *,
    top_k: int = 5,
    model: str | None = None,
    collection_name: str | None = None,
) -> None:
    """Pretty-print search results to stdout (CLI entry-point)."""
    hits = search(
        vdb_dir, query_text,
        top_k=top_k, model=model, collection_name=collection_name,
    )

    print(f"\nQuery: {query_text}")
    print(f"Top {top_k} results\n")

    for i, hit in enumerate(hits, 1):
        chunk_idx = hit["metadata"].get("chunk_index", "?")
        print(f"--- [{i}] source={hit['source']}  chunk={chunk_idx}  score={hit['score']:.4f} ---")
        print(hit["content"])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a ChromaDB VDB")
    parser.add_argument("--vdb", required=True, help="path to VDB directory")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=5, help="number of results")
    parser.add_argument("--model", default=None, help="override embedding model")
    parser.add_argument("--collection", default=None, help="collection name")
    parser.add_argument("--json", action="store_true",
                        help="output JSON for machine consumption")
    args = parser.parse_args()

    if args.json:
        hits = search(
            args.vdb, args.query,
            top_k=args.top_k, model=args.model, collection_name=args.collection,
        )
        print(json.dumps(hits, ensure_ascii=False))
    else:
        query(
            args.vdb, args.query,
            top_k=args.top_k, model=args.model, collection_name=args.collection,
        )


if __name__ == "__main__":
    main()
