"""Invariant tests for the paragraph-aware chunker.

Core promises of `chunker.split_text` (DECISIONS §2 / README):
  1. Empty / all-whitespace input → `[]`
  2. Short text (≤chunk_size) stays one chunk without breaking paragraphs
  3. Multi-paragraph greedy packing: every sub-paragraph in a chunk is a full
     original paragraph
  4. Overlong paragraph (>chunk_size) uses `_split_long` char-level hard split
     with overlap between adjacent chunks
  5. Paragraph-level overlap: on the normal path, every segment after splitting
     a chunk on `\n\n` restores an input paragraph — "never start mid-paragraph"

Pure-function behavior; no chromadb / ollama / external deps.
"""
from __future__ import annotations

from chunker import split_text


def test_split_text_empty_returns_empty():
    assert split_text("") == []
    assert split_text("   \n\n   \n\n  ") == []


def test_split_text_short_fits_single_chunk():
    text = "hello\n\nworld"
    chunks = split_text(text, chunk_size=100, overlap=10)
    assert chunks == ["hello\n\nworld"]


def test_split_text_packs_paragraphs_greedily_without_mid_para_cut():
    paragraphs = ["a" * 30, "b" * 30, "c" * 30, "d" * 30]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, chunk_size=80, overlap=0)

    assert len(chunks) >= 2, "chunk_size=80 should not fit all 4×30-char paras in one chunk"
    for chunk in chunks:
        for sub in chunk.split("\n\n"):
            assert sub in paragraphs, (
                f"split_text produced a mid-paragraph cut: {sub!r} not in input"
            )


def test_split_text_overlap_uses_whole_paragraphs_not_partial():
    paragraphs = [f"para{i}" + "x" * 5 for i in range(10)]  # ~10 chars each
    text = "\n\n".join(paragraphs)

    chunks = split_text(text, chunk_size=30, overlap=10)

    assert len(chunks) >= 2, "small chunk_size should force multiple chunks"
    for chunk in chunks:
        for sub in chunk.split("\n\n"):
            assert sub in paragraphs, (
                f"overlap path created mid-paragraph fragment: {sub!r}"
            )


def test_split_text_long_paragraph_hard_split_respects_chunk_size():
    text = "x" * 1000
    chunks = split_text(text, chunk_size=200, overlap=20)

    assert len(chunks) >= 5
    assert all(len(c) <= 200 for c in chunks), (
        "_split_long must not emit chunks larger than chunk_size"
    )


def test_split_text_long_paragraph_hard_split_overlap_is_char_level():
    text = "abcdefghij" * 100  # 1000 chars, deterministic content
    chunks = split_text(text, chunk_size=200, overlap=20)

    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        if len(prev) < 200:
            continue  # tail chunk may be shorter than chunk_size; overlap invariant relaxed
        assert prev[-20:] == nxt[:20], (
            "consecutive hard-split chunks must overlap by exactly `overlap` chars"
        )


def test_split_text_long_paragraph_followed_by_short_one_does_not_lose_content():
    long_para = "x" * 600
    short_para = "tail"
    text = long_para + "\n\n" + short_para

    chunks = split_text(text, chunk_size=200, overlap=20)
    joined = "".join(chunks)

    assert long_para in joined or all(c.startswith("x") for c in chunks[:-1])
    assert short_para in chunks[-1], "short paragraph after hard-split must not be dropped"
