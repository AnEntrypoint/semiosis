"""Tests for the retrieval-quality eval harness and the centroid-knn improvement."""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.eval import evaluate, recall_at_k  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = [
    "alpha unique term", "beta distinct phrase", "gamma separate idea",
    "delta other concept", "epsilon final note", "zeta extra item",
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
