"""Single integration test surface for semiosis - all octaves, manifold, KB."""
from __future__ import annotations
import os, tempfile
import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")
from hypothesis import given, settings as h_settings, strategies as st  # noqa: E402
from core.cone_engine import HyperbolicConeEngine, ConeFitConfig  # noqa: E402
from core.interfaces import ConeNode, ClusterTree, CommitId, NodeId, Prefix  # noqa: E402
from core.serialization import cone_node_to_dict, cone_node_from_dict  # noqa: E402
from core.settings import ConeSettings, Settings  # noqa: E402
from core.store import InMemoryStore, InMemoryQuery  # noqa: E402
from core.agent_api import (  # noqa: E402
    KnowledgeBase, SearchHit, DiagnoseReport, AnalogyResult, AttentionScore, DispelReport,
)
from core.research_loop import ResearchLoop  # noqa: E402
from core.kb_types import Observation, ResearchResult  # noqa: E402
from core.eval import evaluate  # noqa: E402

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

def _s() -> Settings:
    s = Settings(); s.cone.epochs = 4; return s

def _kb(facts=None) -> KnowledgeBase:
    kb = KnowledgeBase(_s()); kb.ingest(facts or FACTS); return kb

def _tmp() -> str:
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd); return p

def _apex(dim: int = 64):
    import numpy as np; v = np.zeros(dim + 1, dtype=np.float64); v[0] = 1.0; return v

def _tree(n: int = 2) -> ClusterTree:
    ids = [NodeId(f"n{i}") for i in range(n)]
    return ClusterTree(edges=tuple((ids[0], ids[i]) for i in range(1, n)),
                       assignments={f"p{i}": ids[i % n] for i in range(n)},
                       prefix=Prefix(64))

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
    assert torch.allclose(-x[:, 0] ** 2 + (x[:, 1:] ** 2).sum(-1), torch.full((8,), -1.0), atol=1e-4)

def test_cone_engine_and_config() -> None:
    engine = HyperbolicConeEngine(ConeSettings())
    nodes = engine.fit(_tree(n=3))
    mat = engine.batch_contains(nodes, nodes)
    assert mat.shape == (len(nodes), len(nodes)) and all(n.aperture >= 0.1 for n in nodes)
    assert ConeFitConfig.from_settings(_s().cone).epochs == _s().cone.epochs

def test_cone_node_serialization_roundtrip() -> None:
    import numpy as np
    node = ConeNode(id=NodeId("x1"), apex=_apex(), centroid=np.zeros(64), aperture=0.3,
                    prefix=Prefix(64), members=())
    node2 = cone_node_from_dict(cone_node_to_dict(node))
    assert node2.id == node.id and abs(node2.aperture - node.aperture) < 1e-6

def test_store_upsert_and_save_load() -> None:
    from core.encoder import RandomEncoder
    vecs = RandomEncoder(octaves=(64,)).encode(["foo", "bar", "baz"])
    store = InMemoryStore()
    nodes = [ConeNode(id=NodeId(f"n{i}"), apex=_apex(), centroid=v, aperture=0.5,
                      prefix=Prefix(64), members=()) for i, v in enumerate(vecs)]
    store.write(nodes, CommitId("c0"))
    assert len(InMemoryQuery(store, HyperbolicConeEngine(ConeSettings())).knn(vecs[0], k=2, prefix=Prefix(64))) >= 1
    store.upsert(nodes[0])
    path = _tmp()
    try:
        store.save(path); s2 = InMemoryStore(); s2.load(path)
        assert len(s2.all_nodes()) == len(store.all_nodes())
    finally:
        os.remove(path)

def test_pipeline_ingest_and_search() -> None:
    hits = _kb(WEBGL).search("draw call optimization", k=3)
    assert hits and all(isinstance(h, SearchHit) for h in hits)

def test_kb_empty_and_diagnose() -> None:
    assert KnowledgeBase().search("x") == [] and KnowledgeBase().diagnose().nodes == 0
    rep = _kb().diagnose()
    assert isinstance(rep, DiagnoseReport) and rep.nodes > 0

def test_kb_exact_text_top_hit() -> None:
    hits = _kb().search("gamma separate idea", k=1)
    assert hits and hits[0].text == "gamma separate idea"

def test_kb_explain_hierarchy() -> None:
    info = _kb(WEBGL).explain_hierarchy("texture compression")
    assert "node_id" in info and "aperture" in info

def test_kb_recall_and_forget() -> None:
    kb = _kb()
    kb.remember("user prefers ascii only", "p1")
    assert "ascii" in kb.recall("draw call", budget_tokens=300) and kb.forget("p1") is True

def test_kb_input_validation() -> None:
    kb = _kb()
    with pytest.raises(ValueError): kb.search("x", k=0)
    with pytest.raises(ValueError): kb.build_context_pack("x", max_tokens=-1)

def test_kb_batch_search() -> None:
    assert len(_kb().batch_search(["alpha", "beta"], k=2)) == 2

def test_save_load_roundtrip() -> None:
    kb = _kb(); kb.remember("user prefers ascii", "p1"); kb.record_outcome("alpha", ["beta distinct phrase"])
    path = _tmp()
    try:
        kb.save(path); kb2 = KnowledgeBase.load(path, _s())
        assert kb.search_texts("alpha unique term", 2) == kb2.search_texts("alpha unique term", 2)
        assert "ascii" in kb2.recall("x", 200)
    finally:
        os.remove(path)
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.load(os.path.join(tempfile.gettempdir(), "no-such-file-xyz.json"))

def test_semantic_distance_properties() -> None:
    kb = _kb()
    d1 = kb.semantic_distance("alpha unique term", "beta distinct phrase", octave=64)
    d2 = kb.semantic_distance("beta distinct phrase", "alpha unique term", octave=64)
    assert abs(float(d1) - float(d2)) < 1e-5
    assert float(kb.semantic_distance("alpha unique term", "alpha unique term", octave=64)) < 0.05

def test_sense_complexity_label() -> None:
    assert _kb().sense_complexity("machine learning", k=5).complexity_label in (
        "constant", "linear", "quadratic", "exponential")

def test_session7_primitives() -> None:
    kb = _kb()
    assert isinstance(kb.find_analogy("cat", "animal", "dog"), AnalogyResult)
    disp = kb.entropy_dispel()
    assert isinstance(disp, DispelReport) and disp.entropy_before >= 0.0
    hits = kb.search("alpha unique term", k=1)
    if hits:
        a = kb.attention_score(hits[0].node_id, "alpha unique term")
        assert isinstance(a, AttentionScore) and 0.0 <= a.weight <= 1.0

def _agent(kb):
    def fn(d):
        hits = kb.search(d.target_query, k=2) if d.target_query else []
        return Observation(directive_stage=d.stage, result_text="ok",
                           evidence=tuple(h.text for h in hits), success_signal=0.85)
    return fn

def test_research_loop_e2e_and_edges() -> None:
    kb = _kb()
    res = ResearchLoop(kb).run(_agent(kb))
    assert isinstance(res, ResearchResult) and res.converged and res.hypotheses
    h = [x.text for x in res.hypotheses]
    assert len(h) == len(set(h))                                  # dedup
    assert "look near it" in res.refined_instructions["propose"]   # instructions trained
    empty = ResearchLoop(KnowledgeBase()).run(_agent(kb))
    assert empty.converged and not empty.steps                     # empty-KB
    silent = ResearchLoop(kb).run(lambda d: Observation(directive_stage=d.stage))
    assert not silent.converged                                    # no-observation degrade
    s = _s(); s.research.max_cycles = 2; s.research.convergence_energy_delta = -1.0
    assert len(ResearchLoop(kb, s).run(_agent(kb)).steps) <= 2     # non-convergence cap

def test_real_encoder_eval_baseline() -> None:
    st = pytest.importorskip("sentence_transformers")
    s = Settings(); s.cone.epochs = 2
    kb = KnowledgeBase(s); kb.ingest(WEBGL)
    labeled = [
        ("reduce draw calls", {WEBGL[0]}),
        ("attribute binding overhead", {WEBGL[1]}),
        ("texture memory on GPU", {WEBGL[2]}),
    ]
    metrics = evaluate(kb, labeled, k=3)
    # baseline witnessed against nomic-embed-text-v1.5 on this fixture (2026-07-06); a real
    # encoder must beat a random one -- regression below this signals encoder/octave misconfig.
    assert metrics["recall_at_k"] >= 0.5
    assert metrics["mrr"] >= 0.5

def test_research_loop_persist_roundtrip() -> None:
    kb = _kb()
    p = os.path.join(tempfile.gettempdir(), "ri.json")
    if os.path.exists(p): os.remove(p)
    s = _s(); s.research.instruction_persist_path = p
    ResearchLoop(kb, s).run(_agent(kb))
    assert "look near it" in ResearchLoop(kb, s).instructions["propose"]
    with open(p, "w") as f: f.write("{bad")
    assert "keeps coming up" in ResearchLoop(kb, s).instructions["propose"]
    os.remove(p)
