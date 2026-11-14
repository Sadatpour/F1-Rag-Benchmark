from __future__ import annotations

from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    stride = chunk_size - overlap
    for document in documents:
        words = str(document.get("text", "")).split()
        if not words:
            continue
        for start in range(0, len(words), stride):
            part = words[start : start + chunk_size]
            if not part:
                break
            chunks.append(
                {
                    **document,
                    "id": f"{document['id']}:{start}",
                    "text": " ".join(part),
                    "chunk_index": start // stride,
                }
            )
    return chunks
