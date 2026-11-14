from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def relevant_sources(qa_row: dict[str, Any]) -> set[str]:
    source = qa_row.get("source")
    return {str(source)} if source else set()


def recall_at_n(retrieved: Iterable[dict[str, Any]], relevant: set[str], n: int = 10) -> float:
    if not relevant:
        return 0.0
    found = {str(row.get("source")) for row in list(retrieved)[:n]}
    return len(found & relevant) / len(relevant)


def ndcg_at_n(retrieved: Iterable[dict[str, Any]], relevant: set[str], n: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    seen_sources: set[str] = set()
    distinct: list[str] = []
    for row in retrieved:
        source = str(row.get("source"))
        if source not in seen_sources:
            distinct.append(source)
            seen_sources.add(source)
        if len(distinct) == n:
            break
    for rank, source in enumerate(distinct, start=1):
        if source in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), n) + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def retrieval_metrics(
    qa_rows: list[dict[str, Any]],
    retrieve: Any,
    n: int = 10,
) -> dict[str, float | int]:
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    for row in qa_rows:
        relevant = relevant_sources(row)
        hits = retrieve(str(row["question"]), n)
        recall_scores.append(recall_at_n(hits, relevant, n))
        ndcg_scores.append(ndcg_at_n(hits, relevant, n))
    return {
        "questions": len(qa_rows),
        "recall@10": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        "ndcg@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
    }


def qa_accuracy(verdicts: list[dict[str, Any]]) -> float:
    if not verdicts:
        return 0.0
    return sum(bool(verdict["correct"]) for verdict in verdicts) / len(verdicts)
