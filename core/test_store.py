"""Tests for InMemoryStore and InMemoryQuery."""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import CommitId, ConeNode, NodeId, PhraseId, Prefix
from core.store import InMemoryStore


def _make_node(nid: str, dim: int = 9, aperture: float = 0.3) -> ConeNode:
    rng = np.random.default_rng(abs(hash(nid)) % (2**32))
    apex = rng.standard_normal(dim)
    apex[0] = np.sqrt(1.0 + (apex[1:] ** 2).sum())  # on hyperboloid
    return ConeNode(
        id=NodeId(nid),
        apex=apex.astype(np.float64),
        aperture=aperture,
        prefix=Prefix(8),
        members=(PhraseId(f"p_{nid}"),),
    )


def test_write_returns_commit_id() -> None:
    store = InMemoryStore()
    nodes = [_make_node("a"), _make_node("b")]
    cid = store.write(nodes, CommitId("rev-1"))
    assert cid == CommitId("rev-1")


def test_knn_returns_results() -> None:
    store = InMemoryStore()
    nodes = [_make_node(f"n{i}") for i in range(5)]
    store.write(nodes, CommitId("r0"))
    q = np.ones(8, dtype=np.float32)
    results = store.knn(q, k=3, prefix=Prefix(8))
    assert 1 <= len(results) <= 3


def test_knn_empty_store() -> None:
    store = InMemoryStore()
    q = np.zeros(8, dtype=np.float32)
    assert store.knn(q, k=5, prefix=Prefix(8)) == []


def test_upsert_adds_node() -> None:
    store = InMemoryStore()
    n = _make_node("x")
    store.upsert(n)
    assert store.get(NodeId("x")).id == NodeId("x")


def test_save_load_roundtrip(tmp_path) -> None:
    store = InMemoryStore()
    nodes = [_make_node(f"m{i}") for i in range(3)]
    store.write(nodes, CommitId("r1"))
    path = tmp_path / "nodes.json"
    store.save(path)
    store2 = InMemoryStore()
    store2.load(path)
    assert set(store2._nodes.keys()) == set(store._nodes.keys())
