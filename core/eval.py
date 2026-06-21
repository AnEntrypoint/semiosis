"""Retrieval-quality eval harness: recall@k and MRR over a labeled set; measure, do not assume."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsSearchTexts(Protocol):
    def search_texts(self, query: str, k: int = ...) -> list[str]: ...


def recall_at_k(
    kb: SupportsSearchTexts, labeled: Sequence[tuple[str, set[str]]], k: int = 5
) -> float:
    """Mean fraction of each query's relevant texts present in the top-k hits."""
    total = 0.0
    scored = 0
    for query, relevant in labeled:
        if not relevant:
            continue  # empty-relevant queries are undefined, not zero -- exclude from the mean
        hits = set(kb.search_texts(query, k))
        total += len(hits & relevant) / len(relevant)
        scored += 1
    return total / scored if scored else 0.0


def mrr(kb: SupportsSearchTexts, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> float:
    """Mean reciprocal rank of the first relevant hit across queries."""
    total = 0.0
    scored = 0
    for query, relevant in labeled:
        if not relevant:
            continue  # symmetric with recall_at_k: undefined query, excluded from the mean
        scored += 1
        for i, text in enumerate(kb.search_texts(query, k), start=1):
            if text in relevant:
                total += 1.0 / i
                break
    return total / scored if scored else 0.0


def evaluate(
    kb: SupportsSearchTexts, labeled: Sequence[tuple[str, set[str]]], k: int = 5
) -> dict[str, float]:
    """Combined retrieval metrics for one KnowledgeBase over a labeled set."""
    return {"recall_at_k": recall_at_k(kb, labeled, k), "mrr": mrr(kb, labeled, k), "k": float(k)}
