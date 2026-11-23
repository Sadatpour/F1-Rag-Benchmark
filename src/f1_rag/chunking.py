from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


FIXED_CHUNKING = "Fixed chunking"
SEMANTIC_CHUNKING = "Semantic chunking"
RECURSIVE_CHUNKING = "Recursive chunking"
CHUNKING_METHODS = (FIXED_CHUNKING, SEMANTIC_CHUNKING, RECURSIVE_CHUNKING)
SEMANTIC_SIMILARITY_THRESHOLD = 0.72


def _encode(text: str, tokenizer: Any | None) -> list[Any]:
    return tokenizer.encode(text, add_special_tokens=False, verbose=False) if tokenizer else text.split()


def _decode(tokens: list[Any], tokenizer: Any | None) -> str:
    if tokenizer:
        return tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    return " ".join(str(token) for token in tokens)


def _token_count(text: str, tokenizer: Any | None) -> int:
    return len(_encode(text, tokenizer))


def _suffix(text: str, overlap: int, tokenizer: Any | None) -> str:
    if not overlap:
        return ""
    return _decode(_encode(text, tokenizer)[-overlap:], tokenizer)


def _split_recursive(text: str, tokenizer: Any | None, token_limit: int) -> list[str]:
    separators = (r"\n\s*\n+", r"\n+", r"(?<=[.!?])\s+", r"\s+")

    def split(value: str, separator_index: int) -> list[str]:
        if _token_count(value, tokenizer) <= token_limit:
            return [value]
        if separator_index >= len(separators):
            tokens = _encode(value, tokenizer)
            return [
                _decode(tokens[start : start + token_limit], tokenizer)
                for start in range(0, len(tokens), token_limit)
            ]

        parts = re.split(f"({separators[separator_index]})", value)
        if len(parts) == 1:
            return split(value, separator_index + 1)

        units = [
            parts[index] + (parts[index + 1] if index + 1 < len(parts) else "")
            for index in range(0, len(parts), 2)
            if parts[index] or (index + 1 < len(parts) and parts[index + 1])
        ]
        if len(units) == 1 and units[0] == value:
            return split(value, separator_index + 1)
        return [
            piece
            for unit in units
            for piece in split(unit, separator_index + 1)
        ]

    return split(text, 0)


def _pack_units(
    units: list[str],
    tokenizer: Any | None,
    token_limit: int,
    overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit.strip():
            continue
        if _token_count(unit, tokenizer) > token_limit:
            raise ValueError("A recursive chunking unit exceeds the embedding model's token limit.")
        candidate = current + unit
        if current and _token_count(candidate, tokenizer) > token_limit:
            chunks.append(current.strip())
            current = _suffix(current, overlap, tokenizer)
            if current and _token_count(current + unit, tokenizer) > token_limit:
                current = ""
            candidate = current + unit
        current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _semantic_groups(
    text: str,
    token_limit: int,
    tokenizer: Any | None,
    semantic_encoder: Callable[..., Any],
    similarity_threshold: float,
) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    if len(sentences) < 2:
        return [text]

    vectors = np.asarray(
        semantic_encoder(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
        raise ValueError("Semantic chunking requires one embedding vector per sentence.")

    groups: list[str] = []
    current = sentences[0]
    for index, sentence in enumerate(sentences[1:], start=1):
        previous_vector = vectors[index - 1]
        current_vector = vectors[index]
        denominator = float(np.linalg.norm(previous_vector) * np.linalg.norm(current_vector))
        similarity = (
            float(np.dot(previous_vector, current_vector) / denominator)
            if denominator
            else 0.0
        )
        combined = f"{current} {sentence}"
        if similarity < similarity_threshold or _token_count(combined, tokenizer) > token_limit:
            groups.append(current)
            current = sentence
        else:
            current = combined
    groups.append(current)
    return groups

def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
    method: str = FIXED_CHUNKING,
    semantic_encoder: Callable[..., Any] | None = None,
    semantic_similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    if method not in CHUNKING_METHODS:
        raise ValueError(f"Unknown chunking method {method!r}.")
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

        if method == FIXED_CHUNKING and tokenizer is not None:
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
        elif method == FIXED_CHUNKING:
            words = text.split()
            parts = (
                (start, " ".join(words[start : start + token_limit]))
                for start in range(0, len(words), stride)
            )
        elif method == RECURSIVE_CHUNKING:
            document_parts = _pack_units(
                _split_recursive(text, tokenizer, token_limit), tokenizer, token_limit, overlap
            )
            parts = (
                (index * stride, part)
                for index, part in enumerate(document_parts)
            )
        else:
            if semantic_encoder is None:
                raise ValueError("Semantic chunking requires an embedding encoder.")
            if not 0 <= semantic_similarity_threshold <= 1:
                raise ValueError("semantic_similarity_threshold must be between zero and one.")
            groups = _semantic_groups(
                text, token_limit, tokenizer, semantic_encoder, semantic_similarity_threshold
            )
            document_parts = [
                chunk
                for group in groups
                for chunk in _pack_units(
                    _split_recursive(group, tokenizer, token_limit),
                    tokenizer,
                    token_limit,
                    overlap,
                )
            ]
            parts = (
                (index * stride, part)
                for index, part in enumerate(document_parts)
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
