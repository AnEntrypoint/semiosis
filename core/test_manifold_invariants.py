"""Single integration test surface for semiosis - all octaves, manifold, KB."""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from hypothesis import given, settings as h_settings, strategies as st  # noqa: E402

from core.cone_engine import HyperbolicConeEngine, ConeFitConfig  # noqa: E402
from core.encoder import RandomEncoder, FixedClusterer  # noqa: E402
from core.interfaces import ConeNode, ClusterTree, CommitId, NodeId, Prefix  # noqa: E402
from core.serialization import cone_node_to_dict, cone_node_from_dict  # noqa: E402
from core.settings import ConeSettings, Settings  # noqa: E402
from core.store import InMemoryStore, InMemoryQuery  # noqa: E402
from core.agent_api import KnowledgeBase, SearchHit, DiagnoseReport  # noqa: E402

FACTS = [
    "alpha unique term", "beta distinct phrase", "gamma separate idea",
    "delta other concept", "epsilon final note", "zeta extra item",
    "eta seventh thing", "theta eighth thing",
]

WEBGL = [
    "instanced drawing reduces draw calls from 400 to 1 using ANGLE_instanced_arrays",
    "VAO with OES_vertex_array_object replaces N attribute calls with 1 bind",
    "compressed textures via WEBGL_compressed_texture_s3tc stay compressed on GPU",
    "CPU frustum culling cuts GPU work before geometry reaches the rasterizer",
    "gl state cache on the JS side skips redundant useProgram and bindTexture calls",
    "draw call sorting by shader then texture minimizes state changes per frame",
    "mediump precision in fragment shaders saves memory bandwidth on mobile GPUs",
    "bufferSubData instead of bufferData for per-frame data avoids reallocation",
]


def _settings() -> Settings:
    s = Settings()
    s.cone.epochs = 4
    return s


def _kb(facts=None) -> KnowledgeBase:
    kb = KnowledgeBase(_settings())
    kb.ingest(facts or FACTS)
    return kb


def _tmp() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


# --- manifold primitives ---

@h_settings(max_examples=30, deadline=None)
@given(scale=st.floats(min_value=1e-3, max_value=2.0), seed=st.integers(0, 9999))
def test_expmap_logmap_roundtrip(scale: float, seed: int) -> None:
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    x = m.random_normal((4, 6), std=0.1)
    v = m.proju(x, torch.randn(4, 6) * scale)
    assert torch.allclose(v, m.logmap(x, m.expmap(x, v)), atol=1e-3)


@h_settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 9999))
def test_points_stay_on_manifold(seed: int) -> None:
    torch.manual_seed(seed)
    m = geoopt.Lorentz(k=torch.tensor(1.0))
    x = m.random_normal((8, 6), std=0.3)
    ip = -x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(-1)
    assert torch.allclose(ip, torch.full((8,), -1.0), atol=1e-4)


# --- cone engine ---

def _dummy_apex(dim: int = 64):
    import numpy as np
    v = np.zeros(dim + 1, dtype=np.float64)
    v[0] = 1.0
    return v


def _dummy_tree(n: int = 2, prefix: int = 64) -> ClusterTree:
    """Build a minimal ClusterTree with one root->leaf edge per extra node."""
    ids = [NodeId(f"n{i}") for i in range(n)]
    edges = tuple((ids[0], ids[i]) for i in range(1, n))
    assignments = {f"p{i}": ids[i % n] for i in range(n)}
    return ClusterTree(edges=edges, assignments=assignments, prefix=Prefix(prefix))


def test_cone_engine_fit_two_nodes() -> None:
    engine = HyperbolicConeEngine(ConeSettings())
    tree = _dummy_tree(n=2)
    nodes = engine.fit(tree)
    assert all(n.aperture >= 0.1 for n in nodes)


def test_batch_contains_shape() -> None:
    engine = HyperbolicConeEngine(ConeSettings())
    tree = _dummy_tree(n=3)
    nodes = engine.fit(tree)
    mat = engine.batch_contains(nodes, nodes)
    assert mat.shape == (len(nodes), len(nodes))


def test_cone_fit_config_from_settings_roundtrip() -> None:
    cfg = ConeFitConfig.from_settings(_settings().cone)
    assert cfg.epochs == _settings().cone.epochs


# --- store / query protocol ---

def test_store_write_and_knn() -> None:
    enc = RandomEncoder(octaves=(64,))
    store = InMemoryStore()
    engine = HyperbolicConeEngine(ConeSettings())
    query = InMemoryQuery(store, engine)
    vecs = enc.encode(["foo", "bar", "baz"])
    nodes = [ConeNode(id=NodeId(f"n{i}"), apex=_dummy_apex(), centroid=v, aperture=0.5, prefix=Prefix(64), members=()) for i, v in enumerate(vecs)]
    store.write(nodes, CommitId("c0"))
    results = query.knn(vecs[0], k=2, prefix=Prefix(64))
    assert len(results) >= 1


def test_store_upsert_and_save_load() -> None:
    enc = RandomEncoder(octaves=(64,))
    store = InMemoryStore()
    vecs = enc.encode(["hello", "world"])
    nodes = [ConeNode(id=NodeId(f"n{i}"), apex=_dummy_apex(), centroid=v, aperture=0.5, prefix=Prefix(64), members=()) for i, v in enumerate(vecs)]
    store.write(nodes, CommitId("c0"))
    store.upsert(nodes[0])
    path = _tmp()
    try:
        store.save(path)
        store2 = InMemoryStore()
        store2.load(path)
        assert len(store2.all_nodes()) == len(store.all_nodes())
    finally:
        os.remove(path)


# --- serialization ---

def test_cone_node_serialization_roundtrip() -> None:
    import numpy as np
    node = ConeNode(id=NodeId("x1"), apex=_dummy_apex(), centroid=np.zeros(64), aperture=0.3, prefix=Prefix(64), members=())
    d = cone_node_to_dict(node)
    node2 = cone_node_from_dict(d)
    assert node2.id == node.id
    assert abs(node2.aperture - node.aperture) < 1e-6


# --- pipeline / KB surface ---

def test_pipeline_ingest_and_search() -> None:
    kb = _kb(WEBGL)
    hits = kb.search("draw call optimization", k=3)
    assert isinstance(hits, list) and len(hits) >= 1
    assert all(isinstance(h, SearchHit) for h in hits)


def test_kb_empty_search_safe() -> None:
    kb = KnowledgeBase()
    assert kb.search("x") == []
    assert kb.diagnose().nodes == 0


def test_kb_diagnose_shape() -> None:
    kb = _kb()
    rep = kb.diagnose()
    assert isinstance(rep, DiagnoseReport) and rep.nodes > 0


def test_kb_exact_text_top_hit() -> None:
    kb = _kb()
    hits = kb.search("gamma separate idea", k=1)
    assert hits and hits[0].text == "gamma separate idea"


def test_kb_explain_hierarchy() -> None:
    kb = _kb(WEBGL)
    info = kb.explain_hierarchy("texture compression")
    assert "node_id" in info and "aperture" in info


def test_kb_scan_tension() -> None:
    kb = _kb(WEBGL)
    out = kb.scan_tension(top_n=3)
    assert isinstance(out, list)


def test_kb_recall_pinned_fact() -> None:
    kb = _kb()
    kb.remember("user prefers ascii only", "p1")
    block = kb.recall("draw call", budget_tokens=300)
    assert "ascii" in block
    assert kb.forget("p1") is True


def test_kb_input_validation() -> None:
    kb = _kb()
    with pytest.raises(ValueError):
        kb.search("x", k=0)
    with pytest.raises(ValueError):
        kb.build_context_pack("x", max_tokens=-1)


def test_kb_batch_search() -> None:
    kb = _kb()
    out = kb.batch_search(["alpha", "beta"], k=2)
    assert len(out) == 2


# --- persist ---

def test_save_load_roundtrip() -> None:
    kb = _kb()
    kb.remember("user prefers ascii", "p1")
    kb.record_outcome("alpha", ["beta distinct phrase"])
    path = _tmp()
    try:
        kb.save(path)
        kb2 = KnowledgeBase.load(path, _settings())
        assert kb.search_texts("alpha unique term", 2) == kb2.search_texts("alpha unique term", 2)
        assert "ascii" in kb2.recall("x", 200)
    finally:
        os.remove(path)


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.load(os.path.join(tempfile.gettempdir(), "no-such-file-xyz.json"))


# --- semantic direction ---

def test_semantic_distance_symmetric() -> None:
    kb = _kb()
    d1 = kb.semantic_distance("alpha unique term", "beta distinct phrase", octave=64)
    d2 = kb.semantic_distance("beta distinct phrase", "alpha unique term", octave=64)
    assert abs(float(d1) - float(d2)) < 1e-5


def test_semantic_distance_self_near_zero() -> None:
    kb = _kb()
    assert float(kb.semantic_distance("alpha unique term", "alpha unique term", octave=64)) < 0.05


# --- session7 surface ---

def test_sense_complexity_label() -> None:
    kb = _kb()
    mc = kb.sense_complexity("machine learning", k=5)
    assert mc.complexity_label in ("constant", "linear", "quadratic", "exponential")


def test_find_analogy_returns_result() -> None:
    from core.agent_api import AnalogyResult
    kb = _kb()
    r = kb.find_analogy("cat", "animal", "dog")
    assert isinstance(r, AnalogyResult)


def test_attention_score_valid() -> None:
    from core.agent_api import AttentionScore
    kb = _kb()
    hits = kb.search("alpha unique term", k=1)
    if not hits:
        pytest.skip("no hits")
    a = kb.attention_score(hits[0].node_id, "alpha unique term")
    assert isinstance(a, AttentionScore) and 0.0 <= a.weight <= 1.0


def test_entropy_dispel_returns_report() -> None:
    from core.agent_api import DispelReport
    kb = _kb()
    r = kb.entropy_dispel()
    assert isinstance(r, DispelReport) and r.entropy_before >= 0.0
