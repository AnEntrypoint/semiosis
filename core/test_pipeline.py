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
