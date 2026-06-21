"""Integration tests for KnowledgePipeline and KnowledgeBase."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.interfaces import Prefix  # noqa: E402
from core.pipeline import KnowledgePipeline  # noqa: E402


WEBGL_FACTS = [
    "instanced drawing reduces draw calls from 400 to 1 using ANGLE_instanced_arrays",
    "VAO with OES_vertex_array_object replaces N attribute calls with 1 bind",
    "compressed textures via WEBGL_compressed_texture_s3tc stay compressed on GPU",
    "CPU frustum culling cuts GPU work before geometry reaches the rasterizer",
    "gl state cache on the JS side skips redundant useProgram and bindTexture calls",
    "draw call sorting by shader then texture minimizes state changes per frame",
    "mediump precision in fragment shaders saves memory bandwidth on mobile GPUs",
    "bufferSubData instead of bufferData for per-frame data avoids reallocation",
]


def test_pipeline_constructs_without_error() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    assert pipe.store.all_nodes()


def test_pipeline_knn_returns_node_ids() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    enc = pipe._encoder
    q = enc.encode(["reduce draw calls"])[0]
    prefix = Prefix(enc.dims[0])
    results = pipe.query.knn(q[:prefix], k=2, prefix=prefix)
    assert len(results) >= 1


def test_pipeline_containment_score_is_float() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    nodes = pipe.store.all_nodes()
    assert len(nodes) >= 2
    score = pipe.query.containment_score(nodes[0].id, nodes[1].id)
    assert isinstance(score, float)


def test_knowledge_base_ingest_and_search() -> None:
    kb = KnowledgeBase()
    kb.ingest(WEBGL_FACTS)
    results = kb.search("draw call optimization", k=3)
    assert isinstance(results, list)


def test_knowledge_base_explain_hierarchy() -> None:
    kb = KnowledgeBase()
    kb.ingest(WEBGL_FACTS)
    info = kb.explain_hierarchy("texture compression")
    assert "node_id" in info
    assert "aperture" in info


def test_knowledge_base_containment() -> None:
    kb = KnowledgeBase()
    kb.ingest(WEBGL_FACTS)
    score = kb.containment("rendering", "instancing")
    assert isinstance(score, float)


def test_knowledge_base_empty_search() -> None:
    kb = KnowledgeBase()
    assert kb.search("anything") == []
    assert kb.explain_hierarchy("x") == {}
    assert kb.containment("a", "b") == 0.0


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.ingest(WEBGL_FACTS)
    return kb


def test_kb_deep_search_returns_bounded_evidence() -> None:
    kb = _kb()
    out = kb.deep_search("reduce draw calls and compress textures", k=3)
    assert len(out.texts) <= 3


def test_kb_navigate_returns_directions() -> None:
    kb = _kb()
    out = kb.navigate("texture", k=3)
    assert all(n.direction in ("up", "down") for n in out)


def test_entropy_weighted_knn_scored() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    enc = pipe._encoder
    q = enc.encode(["draw call optimization"])[0]
    prefix = Prefix(enc.dims[0])
    q_prefix = q[:prefix]
    baseline = pipe.store.knn_scored(q_prefix, k=5, prefix=prefix, entropy_weight=0.0)
    weighted = pipe.store.knn_scored(q_prefix, k=5, prefix=prefix, entropy_weight=0.3)
    assert len(baseline) == len(weighted)
    assert all(isinstance(score, float) for _, score in baseline)
    assert all(isinstance(score, float) for _, score in weighted)


def test_detect_hierarchy_boundaries() -> None:
    from core.eval import detect_hierarchy_boundaries
    pipe = KnowledgePipeline(WEBGL_FACTS)
    boundaries = detect_hierarchy_boundaries(pipe)
    assert isinstance(boundaries, dict)
    for octave_str, stats in boundaries.items():
        assert "mean_aperture" in stats
        assert "node_count" in stats
        assert stats["node_count"] >= 1


def test_tune_apertures_by_entropy() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    nodes = pipe.store.all_nodes()
    engine = pipe.engine
    tuned = engine.tune_apertures_by_entropy(nodes)
    assert len(tuned) == len(nodes)
    for orig, adj in zip(nodes, tuned):
        assert orig.id == adj.id
        assert adj.aperture >= 0.1  # respects _MIN_APERTURE


def test_decompose_by_octaves() -> None:
    pipe = KnowledgePipeline(WEBGL_FACTS)
    from core.recursive import RecursiveAnswerEngine
    engine = RecursiveAnswerEngine(pipe)
    decomp = engine.decompose_by_octaves("reduce draw calls and compress textures")
    assert isinstance(decomp, dict)
    for octave_idx, clauses in decomp.items():
        assert isinstance(octave_idx, int)
        assert isinstance(clauses, list)


def test_kb_scan_tension_shape() -> None:
    kb = _kb()
    out = kb.scan_tension(top_n=3)
    assert isinstance(out, list)
    for row in out:
        assert isinstance(row.text_a, str) and isinstance(row.kind, str)


def test_kb_build_context_pack_respects_budget() -> None:
    kb = _kb()
    pack = kb.build_context_pack("draw call optimization", max_tokens=100)
    assert pack.total_tokens <= 100


def test_kb_recall_includes_pinned_fact() -> None:
    kb = _kb()
    kb.remember("user prefers ascii only", "p1")
    block = kb.recall("draw call", budget_tokens=300)
    assert "ascii" in block
    assert kb.forget("p1") is True


def test_kb_compress_context_reduces_energy() -> None:
    kb = _kb()
    out = kb.compress_context("gpu memory", k=2)
    assert len(out.texts) <= 2
    assert out.energy_reduction >= 0.0


def test_kb_input_validation() -> None:
    kb = _kb()
    with pytest.raises(ValueError):
        kb.search("x", k=0)
    with pytest.raises(ValueError):
        kb.build_context_pack("x", max_tokens=-1)


def test_incremental_ingest_octaves_distinct_and_cached() -> None:
    kb = KnowledgeBase()
    kb.ingest(WEBGL_FACTS[:5])
    octs = {n.prefix for n in kb._pipeline.store.all_nodes()}
    assert len(octs) == len(kb._pipeline._encoder.dims)  # all octaves coexist (no id collision)
    before = len(kb._pipeline._vec_cache)
    kb.ingest(WEBGL_FACTS[5:])
    # cache reused old embeddings, encoded only the new texts
    assert len(kb._pipeline._vec_cache) == len(set(WEBGL_FACTS))
    assert len(kb._pipeline._vec_cache) > before


def test_incremental_ingest_empty_noop() -> None:
    kb = _kb()
    n0 = len(kb._pipeline.store.all_nodes())
    kb.ingest([])
    assert len(kb._pipeline.store.all_nodes()) == n0
