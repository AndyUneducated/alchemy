#!/usr/bin/env python3
"""CLI tool: query a ChromaDB VDB to test retrieval quality."""

from __future__ import annotations

import argparse
import json
import os
import sys

import chromadb

from config import EMBED_BACKEND, EMBED_MODEL

if EMBED_BACKEND == "ollama":
    from ollama_embedding import OllamaEmbeddingFunction as EmbeddingFunction
else:
    sys.exit(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}")


def query(
    vdb_dir: str,
    query_text: str,
    *,
    top_k: int = 5,
    model: str | None = None,
    collection_name: str | None = None,
) -> None:
    """Search *vdb_dir* for chunks most similar to *query_text*."""
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

    ef = EmbeddingFunction(model=effective_model)

    client = chromadb.PersistentClient(path=vdb_dir)
    collections = client.list_collections()
    if not collections:
        sys.exit(f"No collections found in {vdb_dir}")

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

    print(f"\nQuery: {query_text}")
    print(f"Collection: {collection.name}  |  Top {top_k} results\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        source = meta.get("source", "?")
        chunk_idx = meta.get("chunk_index", "?")
        print(f"--- [{i}] source={source}  chunk={chunk_idx}  distance={dist:.4f} ---")
        print(doc)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a ChromaDB VDB")
    parser.add_argument("--vdb", required=True, help="path to VDB directory")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=5, help="number of results")
    parser.add_argument("--model", default=None, help="override embedding model")
    parser.add_argument("--collection", default=None, help="collection name")
    args = parser.parse_args()

    query(
        args.vdb,
        args.query,
        top_k=args.top_k,
        model=args.model,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
