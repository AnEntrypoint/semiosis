"""Tests for Encoder and HierarchicalClusterer implementations."""

from __future__ import annotations

import numpy as np
import pytest

from core.encoder import AgglomerativeClusterer, RandomEncoder
from core.interfaces import Prefix


def test_random_encoder_dims() -> None:
    enc = RandomEncoder(octaves=(64, 128, 256))
    assert list(enc.dims) == [Prefix(64), Prefix(128), Prefix(256)]


def test_random_encoder_encode_shape() -> None:
    enc = RandomEncoder(octaves=(64, 128))
    vecs = enc.encode(["hello", "world"])
    assert vecs.shape == (2, 128)


def test_random_encoder_slice() -> None:
    enc = RandomEncoder(octaves=(64, 128))
    vecs = enc.encode(["test"])
    sliced = enc.slice(vecs[0], Prefix(64))
    assert sliced.shape == (64,)


def test_random_encoder_deterministic() -> None:
    enc = RandomEncoder(seed=42)
    a = enc.encode(["semiosis"])
    b = enc.encode(["semiosis"])
    np.testing.assert_array_equal(a, b)


def test_random_encoder_unit_vectors() -> None:
    enc = RandomEncoder(octaves=(128,))
    vecs = enc.encode(["a", "b", "c"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


scipy = pytest.importorskip("scipy")


def test_agglomerative_clusterer_returns_cluster_tree() -> None:
    enc = RandomEncoder(octaves=(64,), seed=0)
    vecs = enc.encode([f"doc{i}" for i in range(10)])
    clusterer = AgglomerativeClusterer(n_clusters=3)
    tree = clusterer.fit(vecs, Prefix(64))
    assert len(tree.edges) >= 1
    assert len(tree.assignments) == 10


def test_agglomerative_clusterer_all_assigned() -> None:
    enc = RandomEncoder(octaves=(32,))
    vecs = enc.encode([f"x{i}" for i in range(5)])
    tree = AgglomerativeClusterer(n_clusters=2).fit(vecs, Prefix(32))
    assert set(tree.assignments.keys()) == {f"doc_{i}" for i in range(5)}  # type: ignore[arg-type]


def test_sentence_transformer_encoder_skip_if_absent() -> None:
    pytest.importorskip("sentence_transformers")
    from core.encoder import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        octaves=(64, 128),
    )
    vecs = enc.encode(["hello world"])
    assert vecs.shape[0] == 1
    assert vecs.shape[1] == 128
    sliced = enc.slice(vecs[0], Prefix(64))
    assert sliced.shape == (64,)
    norm = float(np.linalg.norm(sliced))
    assert abs(norm - 1.0) < 1e-5
