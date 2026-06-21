"""Tests for the retrieval-quality eval harness and the centroid-knn improvement."""

from __future__ import annotations

import dataclasses

import pytest

from core.eval import evaluate, mrr, recall_at_k


class _StubKB:
    """Deterministic search_texts so eval-harness arithmetic is testable without an encoder."""

    def __init__(self, ranking: dict[str, list[str]]) -> None:
        self._ranking = ranking

    def search_texts(self, query: str, k: int = 5) -> list[str]:
        return self._ranking.get(query, [])[:k]


def test_empty_relevant_query_excluded_from_mean() -> None:
    kb = _StubKB({"q1": ["a"], "q2": ["x"]})
    # q2 has an empty relevant set: it must not deflate the mean toward zero.
    labeled = [("q1", {"a"}), ("q2", set())]
    assert recall_at_k(kb, labeled, k=3) == 1.0
    assert mrr(kb, labeled, k=3) == 1.0


def test_all_empty_relevant_is_zero() -> None:
    kb = _StubKB({"q1": ["a"]})
    assert recall_at_k(kb, [("q1", set())], k=3) == 0.0
    assert mrr(kb, [("q1", set())], k=3) == 0.0
    assert recall_at_k(kb, [], k=3) == 0.0


torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = [
    "alpha unique term",
    "beta distinct phrase",
    "gamma separate idea",
    "delta other concept",
    "epsilon final note",
    "zeta extra item",
]


def _kb() -> KnowledgeBase:
    s = Settings()
    s.cone.epochs = 4
    kb = KnowledgeBase(s)
    kb.ingest(FACTS)
    return kb


def test_evaluate_returns_metrics() -> None:
    kb = _kb()
    labeled = [(f, {f}) for f in FACTS]
    m = evaluate(kb, labeled, k=3)
    assert 0.0 <= m["recall_at_k"] <= 1.0
    assert 0.0 <= m["mrr"] <= 1.0


def test_centroid_knn_beats_apex_baseline() -> None:
    kb = _kb()
    labeled = [(f, {f}) for f in FACTS]
    centroid = recall_at_k(kb, labeled, k=1)
    # strip centroids to fall back to cone-apex ranking (the old behavior)
    store = kb._pipeline.store
    for nid, n in list(store._nodes.items()):
        store._nodes[nid] = dataclasses.replace(n, centroid=None)
    apex = recall_at_k(kb, labeled, k=1)
    assert centroid >= apex
