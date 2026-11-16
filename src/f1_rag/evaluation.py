from __future__ import annotations

import math
import random
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
        f"recall@{n}": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        f"ndcg@{n}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
    }


def stratified_sample(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("entity_type") or "unspecified")
        groups.setdefault(group, []).append(row)

    generator = random.Random(seed)
    group_names = sorted(groups)
    generator.shuffle(group_names)
    for group in group_names:
        generator.shuffle(groups[group])

    target_count = min(count, len(rows))
    allocations = {group: int(target_count >= len(groups)) for group in group_names}
    remaining = target_count - sum(allocations.values())
    while remaining:
        capacities = {group: len(groups[group]) - allocations[group] for group in group_names}
        total_capacity = sum(capacities.values())
        if not total_capacity:
            break
        shares = {
            group: remaining * capacity / total_capacity
            for group, capacity in capacities.items()
        }
        added = 0
        for group, share in shares.items():
            amount = min(capacities[group], int(share))
            allocations[group] += amount
            added += amount
        remaining -= added
        if remaining:
            for group in sorted(group_names, key=lambda name: shares[name] % 1, reverse=True):
                if remaining and allocations[group] < len(groups[group]):
                    allocations[group] += 1
                    remaining -= 1

    sample = [
        row
        for group in group_names
        for row in generator.sample(groups[group], allocations[group])
    ]
    generator.shuffle(sample)
    return sample


def qa_accuracy(verdicts: list[dict[str, Any]]) -> float:
    if not verdicts:
        return 0.0
    return sum(bool(verdict["correct"]) for verdict in verdicts) / len(verdicts)
