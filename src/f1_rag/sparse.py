from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any


_TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Create normalized word tokens for the sparse BM25 baseline."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


class BM25Index:
    """In-memory BM25 index over the same document chunks used by dense search."""

    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1.")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(str(document.get("text", ""))) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.document_frequency: Counter[str] = Counter(
            term for tokens in self.tokens for term in set(tokens)
        )

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []

        scored: list[tuple[float, int]] = []
        total_documents = len(self.documents)
        for index, document_tokens in enumerate(self.tokens):
            counts = Counter(document_tokens)
            score = 0.0
            document_length = self.lengths[index]
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = math.log1p(
                    (total_documents - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(self.average_length, 1e-12)
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**self.documents[index], "score": score}
            for score, index in scored[:limit]
        ]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict[str, Any]]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Merge ranked result lists without mixing incompatible raw score scales."""
    if limit < 1 or rank_constant < 1:
        raise ValueError("limit and rank_constant must be greater than zero.")

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, document in enumerate(ranked, start=1):
            document_id = str(document.get("id", ""))
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rank_constant + rank)
            documents.setdefault(document_id, document)

    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [
        {**documents[document_id], "score": scores[document_id], "retrieval_method": "hybrid_rrf"}
        for document_id in ordered[:limit]
    ]


def hybrid_search(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    limit: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    return reciprocal_rank_fusion((dense_hits, sparse_hits), limit, rank_constant)
