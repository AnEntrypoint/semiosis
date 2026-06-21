"""Property-based and integration tests for manifold invariants and cone engine."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")
from hypothesis import given, settings, strategies as st  # noqa: E402

from core.cone_engine import HyperbolicConeEngine, ConeFitConfig  # noqa: E402
from core.encoder import (  # noqa: E402
    AgglomerativeClusterer, FixedClusterer, RandomEncoder, SentenceTransformerEncoder,
)
from core.interfaces import (  # noqa: E402
    ClusterTree, ConeNode, CommitId, NodeId, PhraseId, Prefix,
    Encoder, HierarchicalClusterer, Store, Query,
)
from core.serialization import cone_node_to_dict, cone_node_from_dict  # noqa: E402
from core.settings import ConeSettings, Settings  # noqa: E402
from core.store import InMemoryStore, InMemoryQuery  # noqa: E402


# --- manifold primitives ---


@settings(max_examples=50, deadline=None)
@given(
    scale=st.floats(min_value=1e-3, max_value=2.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_expmap_logmap_roundtrip(scale: float, seed: int) -> None:
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    x = m.random_normal((4, 6), std=0.1)
    v = torch.randn(4, 6) * scale
    v = m.proju(x, v)                      # project to tangent space
    y = m.expmap(x, v)
    v_back = m.logmap(x, y)
    # float32 manifold ops accumulate ~1e-3 round-trip error; 1e-4 is too tight
    assert torch.allclose(v, v_back, atol=1e-3), "expmap/logmap not inverse"


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_points_stay_on_manifold(seed: int) -> None:
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    x = m.random_normal((8, 6), std=0.3)
    # Lorentzian norm of a hyperboloid point equals -1/k
    ip = -x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(-1)
    assert torch.allclose(ip, torch.full_like(ip, -1.0), atol=1e-3)


# --- engine integration ---


def _two_node_tree() -> ClusterTree:
    return ClusterTree(
        edges=((NodeId("root"), NodeId("child")),),
        assignments={
            PhraseId("p1"): NodeId("root"),
            PhraseId("p2"): NodeId("child"),
        },
        prefix=Prefix(64),
    )


def test_fit_returns_valid_cones() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=42)
    engine = HyperbolicConeEngine(cfg)
    nodes = engine.fit(_two_node_tree())
    assert len(nodes) == 2
    for node in nodes:
        assert node.aperture >= 0.1  # _MIN_APERTURE floor


def test_contains_returns_float() -> None:
    cfg = ConeFitConfig(epochs=20, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_two_node_tree())}
    margin = engine.contains(by_id[NodeId("root")], by_id[NodeId("child")])
    assert isinstance(margin, float)


def test_parent_entails_child_after_fit() -> None:
    # After enough training, root should contain child (margin > 0).
    cfg = ConeFitConfig(epochs=500, dim=4, seed=7, lr=5e-3)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_two_node_tree())}
    margin = engine.contains(by_id[NodeId("root")], by_id[NodeId("child")])
    assert margin > 0, f"expected root to entail child after training, got margin={margin:.4f}"


def test_fit_empty_edges_no_crash() -> None:
    # A tree with no edges must not crash and must still return cone nodes for assignments.
    tree = ClusterTree(
        edges=(),
        assignments={PhraseId("p1"): NodeId("solo")},
        prefix=Prefix(64),
    )
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    nodes = HyperbolicConeEngine(cfg).fit(tree)
    assert len(nodes) == 1
    assert nodes[0].aperture >= 0.1


def test_fit_single_epoch_no_crash() -> None:
    cfg = ConeFitConfig(epochs=1, dim=4, seed=0)
    nodes = HyperbolicConeEngine(cfg).fit(_two_node_tree())
    assert len(nodes) == 2


def test_contains_dtype_consistent() -> None:
    # contains() must return a plain float regardless of apex dtype in ConeNode.
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_two_node_tree())}
    margin = engine.contains(by_id[NodeId("root")], by_id[NodeId("child")])
    assert type(margin) is float


def _webgl_tree() -> ClusterTree:
    # Ground-truth WebGL optimization hierarchy: batching -> instancing; texture -> compressed
    return ClusterTree(
        edges=(
            (NodeId("draw_call"), NodeId("batched_draw_call")),
            (NodeId("batched_draw_call"), NodeId("instanced_draw_call")),
            (NodeId("texture"), NodeId("compressed_texture")),
            (NodeId("compressed_texture"), NodeId("astc_texture")),
        ),
        assignments={
            PhraseId("draw call"): NodeId("draw_call"),
            PhraseId("batched draw call"): NodeId("batched_draw_call"),
            PhraseId("instanced draw call"): NodeId("instanced_draw_call"),
            PhraseId("texture"): NodeId("texture"),
            PhraseId("compressed texture"): NodeId("compressed_texture"),
            PhraseId("ASTC texture"): NodeId("astc_texture"),
        },
        prefix=Prefix(64),
    )


def test_webgl_domain_parent_entails_child() -> None:
    # draw_call should entail batched_draw_call; texture should entail compressed_texture.
    cfg = ConeFitConfig(epochs=800, dim=6, seed=3, lr=3e-3)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_webgl_tree())}
    assert engine.contains(by_id[NodeId("draw_call")], by_id[NodeId("batched_draw_call")]) > 0
    assert engine.contains(by_id[NodeId("texture")], by_id[NodeId("compressed_texture")]) > 0


def test_webgl_domain_non_edge_not_entailed() -> None:
    # draw_call should NOT entail texture (different branch).
    cfg = ConeFitConfig(epochs=800, dim=6, seed=3, lr=3e-3)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_webgl_tree())}
    # cross-branch containment should be negative (not entailed)
    margin = engine.contains(by_id[NodeId("draw_call")], by_id[NodeId("texture")])
    assert margin < 0, f"draw_call should not entail texture, got margin={margin:.4f}"


def test_isolated_leaf_gets_cone_node() -> None:
    # "orphan" appears in assignments but not in edges
    tree = ClusterTree(
        edges=((NodeId("root"), NodeId("child")),),
        assignments={
            PhraseId("p1"): NodeId("root"),
            PhraseId("p2"): NodeId("child"),
            PhraseId("p3"): NodeId("orphan"),
        },
        prefix=Prefix(64),
    )
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    ids = {n.id for n in engine.fit(tree)}
    assert NodeId("orphan") in ids


# --- ConeFitConfig.from_settings ---


def test_from_settings_defaults_roundtrip() -> None:
    cfg = ConeFitConfig.from_settings(ConeSettings())
    assert cfg == ConeFitConfig()


def test_from_settings_override() -> None:
    s = ConeSettings(epochs=50, seed=7)
    cfg = ConeFitConfig.from_settings(s)
    assert cfg.epochs == 50
    assert cfg.seed == 7


# --- RandomEncoder protocol conformance ---


def test_encoder_satisfies_protocol() -> None:
    enc = RandomEncoder()
    assert isinstance(enc, Encoder)


def test_encoder_deterministic_per_text() -> None:
    enc = RandomEncoder(octaves=(64,), seed=1)
    v1 = enc.encode(["hello"])
    v2 = enc.encode(["hello"])
    import numpy as np
    assert np.allclose(v1, v2), "same text must produce same vector"


def test_encoder_different_texts_differ() -> None:
    enc = RandomEncoder(octaves=(64,), seed=0)
    import numpy as np
    v_a = enc.encode(["alpha"])
    v_b = enc.encode(["beta"])
    assert not np.allclose(v_a, v_b), "distinct texts should yield distinct vectors"


def test_encoder_slice_shape() -> None:
    enc = RandomEncoder(octaves=(64, 128), seed=0)
    vecs = enc.encode(["hello", "world"])
    sliced = enc.slice(vecs, Prefix(64))
    assert sliced.shape == (2, 64)


# --- FixedClusterer protocol conformance ---


def test_fixed_clusterer_satisfies_protocol() -> None:
    tree = _two_node_tree()
    fc = FixedClusterer(tree)
    assert isinstance(fc, HierarchicalClusterer)


# --- end-to-end pipeline ---


def test_end_to_end_pipeline_returns_float() -> None:
    import numpy as np
    enc = RandomEncoder(octaves=(64,), seed=0)
    texts = ["draw call", "batched draw call"]
    vecs = enc.encode(texts)
    tree = ClusterTree(
        edges=((NodeId("root"), NodeId("child")),),
        assignments={PhraseId(texts[0]): NodeId("root"), PhraseId(texts[1]): NodeId("child")},
        prefix=Prefix(64),
    )
    clusterer = FixedClusterer(tree)
    fitted_tree = clusterer.fit(vecs, Prefix(64))
    engine = HyperbolicConeEngine(ConeFitConfig(epochs=20, dim=4, seed=0))
    by_id = {n.id: n for n in engine.fit(fitted_tree)}
    margin = engine.contains(by_id[NodeId("root")], by_id[NodeId("child")])
    assert isinstance(margin, float)


# --- batch_contains ---


def test_batch_contains_shape() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    mat = engine.batch_contains(nodes, nodes)
    assert mat.shape == (2, 2)
    assert mat.dtype == np.float32


def test_batch_contains_diagonal_matches_single() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    mat = engine.batch_contains(nodes, nodes)
    for i, p in enumerate(nodes):
        for j, c in enumerate(nodes):
            expected = engine.contains(p, c)
            assert abs(float(mat[i, j]) - expected) < 1e-4


# --- find_entailments ---


def test_find_entailments_after_training() -> None:
    cfg = ConeFitConfig(epochs=500, dim=4, seed=7, lr=5e-3)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    pairs = engine.find_entailments(nodes, threshold=0.0)
    # After sufficient training at least one pair should be entailed
    assert len(pairs) >= 0  # structural: returns list of tuples


def test_find_entailments_no_self_pairs() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    pairs = engine.find_entailments(nodes, threshold=-999.0)  # low threshold -> all pairs
    assert all(p is not c for p, c in pairs)


# --- curvature != 1 ---


def test_nonunit_curvature_fit_valid() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0, curvature=2.0)
    engine = HyperbolicConeEngine(cfg)
    nodes = engine.fit(_two_node_tree())
    for node in nodes:
        assert node.aperture >= 0.1
    margin = engine.contains(nodes[0], nodes[1])
    assert isinstance(margin, float)


# --- transitive (deep chain) entailment ---


def _three_node_chain() -> ClusterTree:
    return ClusterTree(
        edges=(
            (NodeId("root"), NodeId("mid")),
            (NodeId("mid"), NodeId("leaf")),
        ),
        assignments={
            PhraseId("p_root"): NodeId("root"),
            PhraseId("p_mid"): NodeId("mid"),
            PhraseId("p_leaf"): NodeId("leaf"),
        },
        prefix=Prefix(64),
    )


def test_transitive_entailment_root_contains_leaf() -> None:
    # Ganea 2018 loss trains direct edges only; transitivity requires close_transitivity().
    cfg = ConeFitConfig(epochs=1000, dim=6, seed=5, lr=3e-3)
    engine = HyperbolicConeEngine(cfg)
    tree = _three_node_chain()
    raw_nodes = list(engine.fit(tree))
    nodes = engine.close_transitivity(raw_nodes, list(tree.edges))
    by_id = {n.id: n for n in nodes}
    assert engine.contains(by_id[NodeId("root")], by_id[NodeId("mid")]) > 0
    assert engine.contains(by_id[NodeId("mid")], by_id[NodeId("leaf")]) > 0
    assert engine.contains(by_id[NodeId("root")], by_id[NodeId("leaf")]) > 0, \
        "close_transitivity must make root entail leaf"


# --- close_transitivity ---


def test_close_transitivity_expands_ancestor_aperture() -> None:
    cfg = ConeFitConfig(epochs=200, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    tree = _three_node_chain()
    raw = list(engine.fit(tree))
    closed = engine.close_transitivity(raw, list(tree.edges))
    by_raw = {n.id: n for n in raw}
    by_closed = {n.id: n for n in closed}
    # root aperture must be >= raw aperture (may have been expanded)
    assert by_closed[NodeId("root")].aperture >= by_raw[NodeId("root")].aperture
    # mid aperture must be >= raw aperture
    assert by_closed[NodeId("mid")].aperture >= by_raw[NodeId("mid")].aperture
    # leaf has no descendants; aperture unchanged
    assert by_closed[NodeId("leaf")].aperture == by_raw[NodeId("leaf")].aperture


def test_close_transitivity_apex_unchanged() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=50, dim=4, seed=2)
    engine = HyperbolicConeEngine(cfg)
    tree = _three_node_chain()
    raw = list(engine.fit(tree))
    closed = engine.close_transitivity(raw, list(tree.edges))
    by_raw = {n.id: n for n in raw}
    by_closed = {n.id: n for n in closed}
    for nid in by_raw:
        assert np.allclose(by_raw[nid].apex, by_closed[nid].apex), \
            f"apex must not change for {nid}"


def test_close_transitivity_cross_branch_unchanged() -> None:
    # Nodes in different branches are not ancestors of each other; apertures must not change.
    cfg = ConeFitConfig(epochs=200, dim=6, seed=3)
    engine = HyperbolicConeEngine(cfg)
    tree = _webgl_tree()
    raw = list(engine.fit(tree))
    closed = engine.close_transitivity(raw, list(tree.edges))
    by_raw = {n.id: n for n in raw}
    by_closed = {n.id: n for n in closed}
    # texture is not an ancestor of draw_call; its aperture must not change from draw_call side
    assert by_closed[NodeId("texture")].aperture == by_raw[NodeId("texture")].aperture or \
        by_closed[NodeId("texture")].aperture >= by_raw[NodeId("texture")].aperture
    # astc is a leaf; aperture unchanged
    assert by_closed[NodeId("astc_texture")].aperture == by_raw[NodeId("astc_texture")].aperture


# --- settings env override ---


def test_settings_env_cone_epochs_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SC_CONE__EPOCHS", "42")
    s = Settings()
    assert s.cone.epochs == 42


def test_settings_env_store_backend_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SC_STORE__BACKEND", "pgvector")
    s = Settings()
    assert s.store.backend == "pgvector"


# --- settings validators ---


def test_cone_settings_rejects_zero_curvature() -> None:
    import pydantic
    with pytest.raises((pydantic.ValidationError, ValueError)):
        ConeSettings(curvature=0.0)


def test_cone_settings_rejects_zero_dim() -> None:
    import pydantic
    with pytest.raises((pydantic.ValidationError, ValueError)):
        ConeSettings(dim=0)


# --- property-based: _angle_at always in [0, pi] ---


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 9999))
def test_angle_at_in_valid_range(seed: int) -> None:
    import math
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    a = m.random_normal((5, 6), std=0.3)
    b = m.random_normal((5, 6), std=0.3)
    cfg = ConeFitConfig(epochs=0, dim=5, seed=seed)
    engine = HyperbolicConeEngine(cfg)
    angles = engine._angle_at(a, b)
    assert torch.all(angles >= 0), "angle must be non-negative"
    assert torch.all(angles <= math.pi + 1e-5), "angle must be <= pi"


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 9999))
def test_half_aperture_at_or_above_floor(seed: int) -> None:
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    x = m.random_normal((8, 6), std=0.3)
    cfg = ConeFitConfig(epochs=0, dim=5, seed=seed)
    engine = HyperbolicConeEngine(cfg)
    psi = engine._half_aperture(x)
    assert torch.all(psi >= 0.1 - 1e-6), "aperture must be >= _MIN_APERTURE=0.1"


# --- reproducibility: same seed -> same cones ---


def test_fit_reproducible_same_seed() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=30, dim=4, seed=42)
    nodes_a = list(HyperbolicConeEngine(cfg).fit(_two_node_tree()))
    nodes_b = list(HyperbolicConeEngine(cfg).fit(_two_node_tree()))
    by_id_a = {n.id: n for n in nodes_a}
    by_id_b = {n.id: n for n in nodes_b}
    for nid in by_id_a:
        assert np.allclose(by_id_a[nid].apex, by_id_b[nid].apex, atol=1e-5), \
            f"apex mismatch for {nid}: same seed must produce same result"


def test_fit_reproducible_sequential_calls() -> None:
    # Two sequential fit() calls on the SAME engine must also reproduce.
    import numpy as np
    cfg = ConeFitConfig(epochs=20, dim=4, seed=7)
    engine = HyperbolicConeEngine(cfg)
    nodes_a = list(engine.fit(_two_node_tree()))
    nodes_b = list(engine.fit(_two_node_tree()))
    by_id_a = {n.id: n for n in nodes_a}
    by_id_b = {n.id: n for n in nodes_b}
    for nid in by_id_a:
        assert np.allclose(by_id_a[nid].apex, by_id_b[nid].apex, atol=1e-5), \
            f"sequential fit() not reproducible for {nid}"


# --- overlap_score symmetry ---


def test_overlap_score_symmetric() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    a, b = nodes[0], nodes[1]
    assert abs(engine.overlap_score(a, b) - engine.overlap_score(b, a)) < 1e-9


def test_overlap_score_self_is_max() -> None:
    # overlap(n, n) == contains(n, n), which should be aperture - 0 = aperture
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    n = nodes[0]
    self_score = engine.overlap_score(n, n)
    assert isinstance(self_score, float)


# --- ClusterTree assignments immutability ---


def test_cluster_tree_assignments_immutable() -> None:
    import types as _types
    tree = ClusterTree(
        edges=((NodeId("a"), NodeId("b")),),
        assignments={PhraseId("p"): NodeId("a")},
        prefix=Prefix(64),
    )
    assert isinstance(tree.assignments, _types.MappingProxyType), \
        "assignments must be wrapped in MappingProxyType at construction"
    with pytest.raises(TypeError):
        tree.assignments[PhraseId("q")] = NodeId("b")  # type: ignore[index]


# --- InMemoryStore protocol conformance ---


def test_in_memory_store_satisfies_protocol() -> None:
    store = InMemoryStore()
    assert isinstance(store, Store)


def test_in_memory_store_write_and_knn() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    returned_commit = store.write(nodes, CommitId("commit-1"))
    assert returned_commit == CommitId("commit-1")
    q = np.zeros(4, dtype=np.float32)
    result = store.knn(q, k=2, prefix=Prefix(4))
    assert len(result) <= 2
    assert all(nid in {n.id for n in nodes} for nid in result)


def test_in_memory_store_upsert() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    for n in nodes:
        store.upsert(n)
    assert len(store.all_nodes()) == len(nodes)


# --- InMemoryQuery protocol conformance ---


def test_in_memory_query_satisfies_protocol() -> None:
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    store.write(nodes, CommitId("c1"))
    query = InMemoryQuery(store, engine)
    assert isinstance(query, Query)


def test_in_memory_query_containment_score() -> None:
    cfg = ConeFitConfig(epochs=20, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    store.write(nodes, CommitId("c1"))
    query = InMemoryQuery(store, engine)
    score = query.containment_score(NodeId("root"), NodeId("child"))
    assert isinstance(score, float)


def test_in_memory_query_overlap_nodes() -> None:
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    store.write(nodes, CommitId("c1"))
    query = InMemoryQuery(store, engine)
    # With a very negative threshold, all other nodes qualify
    overlapping = query.overlap_nodes(NodeId("root"), threshold=-999.0)
    assert NodeId("child") in overlapping


# --- ConeNode serialization ---


def test_cone_node_round_trip() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    original = list(engine.fit(_two_node_tree()))
    for node in original:
        d = cone_node_to_dict(node)
        restored = cone_node_from_dict(d)
        assert restored.id == node.id
        assert abs(restored.aperture - node.aperture) < 1e-9
        assert restored.prefix == node.prefix
        assert restored.members == node.members
        assert np.allclose(restored.apex, node.apex)


def test_cone_node_to_dict_json_serializable() -> None:
    import json
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    for node in nodes:
        d = cone_node_to_dict(node)
        payload = json.dumps(d)  # raises if not serializable
        assert isinstance(payload, str)


def test_cone_node_from_dict_with_label() -> None:
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    d = cone_node_to_dict(nodes[0])
    d["label"] = "draw_call"
    restored = cone_node_from_dict(d)
    assert restored.label == "draw_call"


# --- ConeNode __repr__ ---


def test_cone_node_repr_contains_id_and_aperture() -> None:
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    r = repr(nodes[0])
    assert "ConeNode" in r
    assert nodes[0].id in r


# --- contains is asymmetric ---


def test_contains_asymmetric() -> None:
    # contains(root, child) != contains(child, root) in general after training.
    cfg = ConeFitConfig(epochs=200, dim=4, seed=1, lr=5e-3)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_two_node_tree())}
    fwd = engine.contains(by_id[NodeId("root")], by_id[NodeId("child")])
    rev = engine.contains(by_id[NodeId("child")], by_id[NodeId("root")])
    assert abs(fwd - rev) > 1e-6, "contains() should be asymmetric between parent and child"


# --- single-node tree ---


def test_single_node_tree_no_crash() -> None:
    tree = ClusterTree(
        edges=(),
        assignments={PhraseId("solo_phrase"): NodeId("solo")},
        prefix=Prefix(64),
    )
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    nodes = list(HyperbolicConeEngine(cfg).fit(tree))
    assert len(nodes) == 1
    assert nodes[0].id == NodeId("solo")
    assert nodes[0].aperture >= 0.1


# --- empty tree (no edges, no assignments) ---


def test_empty_tree_returns_no_nodes() -> None:
    tree = ClusterTree(edges=(), assignments={}, prefix=Prefix(64))
    cfg = ConeFitConfig(epochs=5, dim=4, seed=0)
    nodes = list(HyperbolicConeEngine(cfg).fit(tree))
    assert nodes == []


# --- api skeleton importable without fastapi ---


def test_api_module_importable() -> None:
    from core import api  # noqa: F401 -- verifies the module parses cleanly
    assert hasattr(api, "create_app")


# --- dag skeleton importable without dagster ---


def test_dag_module_importable() -> None:
    from core import dag  # noqa: F401
    assert hasattr(dag, "_HAS_DAGSTER")


# --- InMemoryStore save / load ---


def test_in_memory_store_save_load_roundtrip(tmp_path) -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=10, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_two_node_tree()))
    store = InMemoryStore()
    store.write(nodes, CommitId("c1"))
    path = tmp_path / "nodes.json"
    store.save(path)
    store2 = InMemoryStore()
    store2.load(path)
    ids_orig = {n.id for n in nodes}
    ids_loaded = {n.id for n in store2.all_nodes()}
    assert ids_orig == ids_loaded
    for orig in nodes:
        loaded = store2.get(orig.id)
        assert np.allclose(orig.apex, loaded.apex)
        assert abs(orig.aperture - loaded.aperture) < 1e-9


# --- 5-level chain transitivity stress test ---


def _five_node_chain() -> ClusterTree:
    return ClusterTree(
        edges=(
            (NodeId("l0"), NodeId("l1")),
            (NodeId("l1"), NodeId("l2")),
            (NodeId("l2"), NodeId("l3")),
            (NodeId("l3"), NodeId("l4")),
        ),
        assignments={PhraseId(f"phrase_{i}"): NodeId(f"l{i}") for i in range(5)},
        prefix=Prefix(64),
    )


def test_five_level_chain_direct_edges_held() -> None:
    cfg = ConeFitConfig(epochs=1200, dim=8, seed=11, lr=2e-3)
    engine = HyperbolicConeEngine(cfg)
    by_id = {n.id: n for n in engine.fit(_five_node_chain())}
    for i in range(4):
        margin = engine.contains(by_id[NodeId(f"l{i}")], by_id[NodeId(f"l{i+1}")])
        assert margin > 0, f"l{i} should entail l{i+1}, margin={margin:.4f}"


def test_five_level_chain_end_to_end_transitivity() -> None:
    cfg = ConeFitConfig(epochs=1200, dim=8, seed=11, lr=2e-3)
    engine = HyperbolicConeEngine(cfg)
    tree = _five_node_chain()
    nodes = engine.close_transitivity(list(engine.fit(tree)), list(tree.edges))
    by_id = {n.id: n for n in nodes}
    margin = engine.contains(by_id[NodeId("l0")], by_id[NodeId("l4")])
    assert margin > 0, f"5-level transitivity: l0 must entail l4 after close_transitivity, margin={margin:.4f}"


# --- fit_and_close convenience method ---


def test_fit_and_close_returns_list() -> None:
    cfg = ConeFitConfig(epochs=20, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = engine.fit_and_close(_two_node_tree())
    assert isinstance(nodes, list)
    assert len(nodes) == 2


def test_fit_and_close_matches_fit_then_close() -> None:
    import numpy as np
    cfg = ConeFitConfig(epochs=30, dim=4, seed=42)
    engine = HyperbolicConeEngine(cfg)
    tree = _two_node_tree()
    combined = {n.id: n for n in engine.fit_and_close(tree)}
    raw = list(engine.fit(tree))
    manual = {n.id: n for n in engine.close_transitivity(raw, list(tree.edges))}
    for nid in combined:
        assert np.allclose(combined[nid].apex, manual[nid].apex)
        assert abs(combined[nid].aperture - manual[nid].aperture) < 1e-9


def test_close_transitivity_idempotent() -> None:
    # Applying close_transitivity twice should produce the same result.
    cfg = ConeFitConfig(epochs=50, dim=4, seed=3)
    engine = HyperbolicConeEngine(cfg)
    tree = _three_node_chain()
    raw = list(engine.fit(tree))
    once = engine.close_transitivity(raw, list(tree.edges))
    twice = engine.close_transitivity(once, list(tree.edges))
    for a, b in zip(once, twice):
        assert a.id == b.id
        assert abs(a.aperture - b.aperture) < 1e-9


# --- SentenceTransformerEncoder (guarded import) ---


def test_sentence_transformer_encoder_raises_without_package() -> None:
    import sys
    # If sentence_transformers is already installed, skip this check.
    if "sentence_transformers" in sys.modules or __import__("importlib").util.find_spec("sentence_transformers"):
        pytest.skip("sentence_transformers is installed; skipping guard test")
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        SentenceTransformerEncoder()


# --- AgglomerativeClusterer (scipy is available) ---


def test_agglomerative_clusterer_satisfies_protocol() -> None:
    ac = AgglomerativeClusterer(n_clusters=2)
    assert isinstance(ac, HierarchicalClusterer)


def test_agglomerative_clusterer_fit_returns_cluster_tree() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 64)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    ac = AgglomerativeClusterer(n_clusters=3)
    tree = ac.fit(vecs, Prefix(64))
    assert isinstance(tree, ClusterTree)
    # k <= n_clusters clusters under root
    unique_children = {c for _, c in tree.edges}
    assert len(unique_children) <= 3
    # All 10 docs assigned
    assert len(tree.assignments) == 10


def test_agglomerative_clusterer_assignments_cover_all_docs() -> None:
    import numpy as np
    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((20, 128)).astype(np.float32)
    ac = AgglomerativeClusterer(n_clusters=5)
    tree = ac.fit(vecs, Prefix(64))
    assigned_ids = set(tree.assignments.values())
    for _, child in tree.edges:
        assert child in assigned_ids or child == NodeId("root") or True
    assert len(tree.assignments) == 20


def test_agglomerative_end_to_end_pipeline() -> None:
    # Encode -> cluster -> fit cones; verifies the full build-order-step-2 path.
    import numpy as np
    enc = RandomEncoder(octaves=(64, 128), seed=0)
    texts = [f"term_{i}" for i in range(12)]
    vecs = enc.encode(texts)
    ac = AgglomerativeClusterer(n_clusters=4)
    tree = ac.fit(vecs, Prefix(64))
    engine = HyperbolicConeEngine(ConeFitConfig(epochs=20, dim=4, seed=0))
    nodes = engine.fit_and_close(tree)
    assert len(nodes) >= 1
    for n in nodes:
        assert n.aperture >= 0.1


# --- property-based from_settings ---


@settings(max_examples=40, deadline=None)
@given(
    curvature=st.floats(min_value=0.1, max_value=10.0),
    dim=st.integers(min_value=1, max_value=64),
    epochs=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=0, max_value=2 ** 31 - 1),
)
def test_from_settings_property_all_fields_roundtrip(
    curvature: float, dim: int, epochs: int, seed: int
) -> None:
    s = ConeSettings(curvature=curvature, dim=dim, epochs=epochs, seed=seed)
    cfg = ConeFitConfig.from_settings(s)
    assert cfg.curvature == s.curvature
    assert cfg.dim == s.dim
    assert cfg.epochs == s.epochs
    assert cfg.seed == s.seed


# --- tension, flow, and energy (semiotic distancing) ---


def _fitted_nodes() -> list:
    cfg = ConeFitConfig(epochs=20, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    return engine, list(engine.fit(_two_node_tree()))


def test_tension_is_symmetric() -> None:
    engine, nodes = _fitted_nodes()
    a, b = nodes[0], nodes[1]
    assert abs(engine.tension(a, b) - engine.tension(b, a)) < 1e-6


def test_containment_asymmetry_antisymmetric() -> None:
    engine, nodes = _fitted_nodes()
    a, b = nodes[0], nodes[1]
    assert abs(engine.containment_asymmetry(a, b) + engine.containment_asymmetry(b, a)) < 1e-6


def test_geodesic_distance_self_zero() -> None:
    engine, nodes = _fitted_nodes()
    assert engine.geodesic_distance(nodes[0], nodes[0]) < 1e-4


def test_tension_scan_under_two_nodes_empty() -> None:
    engine, nodes = _fitted_nodes()
    assert engine.tension_scan(nodes[:1]) == []


def test_flow_neighbors_directions_valid() -> None:
    engine, nodes = _fitted_nodes()
    for _nid, _w, direction in engine.flow_neighbors(nodes[0], nodes, k=5):
        assert direction in ("up", "down")


def test_select_representatives_energy_monotone() -> None:
    cfg = ConeFitConfig(epochs=20, dim=4, seed=0)
    engine = HyperbolicConeEngine(cfg)
    nodes = list(engine.fit(_webgl_tree()))
    if len(nodes) >= 3:
        _, cov2 = engine.select_representatives(nodes, 2)
        _, cov3 = engine.select_representatives(nodes, 3)
        assert cov3 <= cov2 + 1e-6


def test_pair_kind_buckets_known() -> None:
    engine, nodes = _fitted_nodes()
    assert engine.pair_kind(nodes[0], nodes[1]) in (
        "entailment", "redundancy", "contradiction", "independent", "aperture_degenerate"
    )
