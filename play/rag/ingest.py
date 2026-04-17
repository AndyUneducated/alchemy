#!/usr/bin/env python3
"""CLI tool: read documents (txt/md/pdf), chunk, embed via Ollama, persist to ChromaDB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import fitz  # pymupdf

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, OLLAMA_URL

from chunker import split_text

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_file(path: str) -> str:
    """Extract text from a file; dispatches on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n\n".join(page.get_text() for page in doc)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _collect_docs(paths: list[str]) -> list[tuple[str, str]]:
    """Return a list of (relative_path, text) for files and/or directories."""
    docs: list[tuple[str, str]] = []
    for path in paths:
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = _read_file(path).strip()
            if text:
                docs.append((os.path.basename(path), text))
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for fname in sorted(files):
                    if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    fpath = os.path.join(root, fname)
                    text = _read_file(fpath).strip()
                    if text:
                        rel = os.path.relpath(fpath, path)
                        docs.append((rel, text))
    return docs


def ingest(
    doc_paths: list[str],
    output_dir: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    model: str = EMBED_MODEL,
    collection_name: str | None = None,
) -> None:
    """Read docs from *doc_paths*, chunk, embed, and persist a ChromaDB VDB."""
    docs = _collect_docs(doc_paths)
    if not docs:
        sys.exit(f"No supported files found in: {', '.join(doc_paths)}")

    print(f"Found {len(docs)} document(s)")

    ef = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=model)

    client = chromadb.PersistentClient(path=output_dir)
    col_name = collection_name or os.path.basename(os.path.normpath(output_dir))
    collection = client.get_or_create_collection(name=col_name, embedding_function=ef)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for rel_path, text in docs:
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"  {rel_path}: {len(chunks)} chunk(s)")
        for i, chunk in enumerate(chunks):
            ids.append(f"{rel_path}::{i}")
            documents.append(chunk)
            metadatas.append({"source": rel_path, "chunk_index": i})

    print(f"Embedding {len(documents)} chunk(s) via {model} ...")
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    metadata = {
        "embedding_model": model,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "doc_count": len(docs),
        "chunk_count": len(documents),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"VDB saved to {output_dir}  ({len(documents)} chunks)")
    print(f"Metadata written to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into a ChromaDB VDB")
    parser.add_argument("--docs", required=True, nargs="+",
                        help="files or directories of .txt/.md/.pdf documents")
    parser.add_argument("--output", required=True, help="output VDB directory")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--model", default=EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--collection", default=None, help="ChromaDB collection name")
    args = parser.parse_args()

    ingest(
        args.docs,  # now a list
        args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=args.model,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
