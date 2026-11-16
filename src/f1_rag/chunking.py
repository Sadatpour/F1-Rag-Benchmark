from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    if tokenizer is not None:
        if max_tokens is None:
            raise ValueError("max_tokens is required when a tokenizer is provided.")
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        token_limit = min(chunk_size, max_tokens - special_tokens)
        if token_limit < 1 or overlap >= token_limit:
            raise ValueError("The token overlap must be smaller than the model's available input length.")
    else:
        token_limit = chunk_size
    stride = token_limit - overlap

    for document in documents:
        text = str(document.get("text", ""))
        if not text.strip():
            continue

        if tokenizer is not None:
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)

            def token_parts() -> Iterator[tuple[int, str]]:
                for start in range(0, len(tokens), stride):
                    if start and start - stride + token_limit >= len(tokens):
                        break
                    yield (
                        start,
                        tokenizer.decode(
                            tokens[start : start + token_limit],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        ),
                    )

            parts = token_parts()
        else:
            words = text.split()
            parts = (
                (start, " ".join(words[start : start + token_limit]))
                for start in range(0, len(words), stride)
            )

        for start, part in parts:
            if not part.strip():
                continue
            chunks.append(
                {
                    **document,
                    "id": f"{document['id']}:{start}",
                    "text": part,
                    "chunk_index": start // stride,
                }
            )
    return chunks
