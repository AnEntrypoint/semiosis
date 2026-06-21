"""Retrieval-quality eval harness: recall@k and MRR over a labeled set; measure, do not assume."""
from __future__ import annotations

from typing import Sequence


def recall_at_k(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> float:
    """Mean fraction of each query's relevant texts present in the top-k hits."""
    if not labeled:
        return 0.0
    total = 0.0
    for query, relevant in labeled:
        if not relevant:
            continue
        hits = set(kb.search_texts(query, k))
        total += len(hits & relevant) / len(relevant)
    return total / len(labeled)


def mrr(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> float:
    """Mean reciprocal rank of the first relevant hit across queries."""
    if not labeled:
        return 0.0
    total = 0.0
    for query, relevant in labeled:
        rank = 0.0
        for i, text in enumerate(kb.search_texts(query, k), start=1):
            if text in relevant:
                rank = 1.0 / i
                break
        total += rank
    return total / len(labeled)


def evaluate(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> dict[str, float]:
    """Combined retrieval metrics for one KnowledgeBase over a labeled set."""
    return {"recall_at_k": recall_at_k(kb, labeled, k), "mrr": mrr(kb, labeled, k), "k": float(k)}
