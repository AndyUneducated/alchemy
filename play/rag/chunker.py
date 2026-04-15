"""Paragraph-aware text chunking with overlap."""

from __future__ import annotations

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into chunks of roughly *chunk_size* characters.

    The algorithm respects paragraph boundaries (double newlines) so that a
    chunk never starts or ends in the middle of a paragraph when avoidable.
    Adjacent chunks share *overlap* characters of trailing/leading context.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = _carry_overlap(current, overlap)
            chunks.extend(_split_long(para, chunk_size, overlap))
            current = []
            current_len = 0
            continue

        sep_len = 2 if current else 0  # "\n\n" separator
        if current_len + sep_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            current, current_len = _carry_overlap(current, overlap)

        current.append(para)
        current_len += (2 if len(current) > 1 else 0) + para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _carry_overlap(
    parts: list[str], overlap: int
) -> tuple[list[str], int]:
    """Return trailing paragraphs whose total length <= *overlap*."""
    carry: list[str] = []
    total = 0
    for part in reversed(parts):
        if total + len(part) > overlap:
            break
        carry.insert(0, part)
        total += len(part) + 2
    return carry, max(total - 2, 0) if carry else 0


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard-split a single oversized paragraph by character boundaries."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    if len(chunks) > 1 and len(chunks[-1]) <= overlap:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks
