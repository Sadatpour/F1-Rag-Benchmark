from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any


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


def recall_ids_at_n(
    retrieved: Iterable[dict[str, Any]],
    relevant_ids: set[str],
    n: int = 10,
) -> float:
    if n < 1:
        raise ValueError("n must be greater than zero.")
    if not relevant_ids:
        return 0.0
    found = {str(row.get("id")) for row in list(retrieved)[:n]}
    return len(found & relevant_ids) / len(relevant_ids)


def ndcg_ids_at_n(
    retrieved: Iterable[dict[str, Any]],
    relevance_by_id: dict[str, int],
    n: int = 10,
) -> float:
    """Compute graded NDCG using document/chunk IDs and relevance grades."""
    if n < 1:
        raise ValueError("n must be greater than zero.")
    if not relevance_by_id:
        return 0.0
    dcg = 0.0
    for rank, row in enumerate(list(retrieved)[:n], start=1):
        grade = max(0, int(relevance_by_id.get(str(row.get("id")), 0)))
        dcg += (2**grade - 1) / math.log2(rank + 1)
    ideal_grades = sorted((int(grade) for grade in relevance_by_id.values() if int(grade) > 0), reverse=True)[:n]
    ideal_dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


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
