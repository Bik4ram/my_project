from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(cid in relevant for cid in retrieved) else 0.0


def average_rank(retrieved: list[str], relevant: set[str]) -> float:
    ranks = [i for i, cid in enumerate(retrieved, start=1) if cid in relevant]
    if not ranks:
        return float("inf")
    return sum(ranks) / len(ranks)


def file_recall(retrieved_files: set[str], expected_files: set[str]) -> float:
    if not expected_files:
        return 1.0
    found = expected_files & retrieved_files
    return len(found) / len(expected_files)
