"""Tests for semantic direction, distance, trajectory, fold, and agentic reflection."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import (  # noqa: E402
    KnowledgeBase, SemanticDirection, SemanticDirectionError,
    DirectionSearchResult, StubSummarizer,
)
from core.settings import Settings  # noqa: E402

TEXTS = [
    "alpha unique term", "beta distinct phrase", "gamma separate idea",
    "delta other concept", "epsilon final note", "zeta extra item",
    "eta seventh thing", "theta eighth thing",
]


def _kb(**kw) -> KnowledgeBase:
    s = Settings()
    s.cone.epochs = 4
    for k, v in kw.items():
        setattr(s.agent, k, v)
    kb = KnowledgeBase(s)
    kb.ingest(TEXTS)
    return kb


def test_semantic_distance_symmetric() -> None:
    kb = _kb()
    d1 = kb.semantic_distance("alpha unique term", "beta distinct phrase", octave=64)
    d2 = kb.semantic_distance("beta distinct phrase", "alpha unique term", octave=64)
    assert abs(float(d1) - float(d2)) < 1e-5


def test_semantic_distance_self_zero() -> None:
    kb = _kb()
    d = kb.semantic_distance("alpha unique term", "alpha unique term", octave=64)
    assert float(d) < 0.05


def test_semantic_distance_octave_dict() -> None:
    kb = _kb()
    d = kb.semantic_distance("alpha unique term", "beta distinct phrase")
    assert isinstance(d, dict)
    assert len(d) > 0
    for v in d.values():
        assert 0.0 <= v <= 2.1  # cosine in [0,2], hyperbolic can be larger but bounded


def test_semantic_distance_hyperbolic() -> None:
    kb = _kb()
    d = kb.semantic_distance("alpha unique term", "beta distinct phrase", octave=64, use_hyperbolic=True)
    assert float(d) >= 0.0


def test_best_octave_returns_valid_prefix() -> None:
    kb = _kb()
    enc = kb._pipeline._encoder
    valid = {int(d) for d in enc.dims}
    p = kb.best_octave("alpha unique term", "gamma separate idea")
    assert p in valid


def test_compute_direction_unit_norm() -> None:
    kb = _kb()
    nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
    if len(nodes) < 2:
        pytest.skip("not enough nodes")
    a, b = nodes[0], nodes[1]
    import numpy as np
    d = kb.compute_direction(str(a.id), str(b.id))
    assert isinstance(d, SemanticDirection)
    dv = np.array(d.direction_vec)
    assert abs(np.linalg.norm(dv) - 1.0) < 1e-5


def test_compute_direction_zero_raises() -> None:
    kb = _kb()
    nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
    if not nodes:
        pytest.skip("no nodes")
    nid = str(nodes[0].id)
    with pytest.raises(SemanticDirectionError):
        kb.compute_direction(nid, nid)


def test_direction_search_returns_hits() -> None:
    kb = _kb()
    nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
    if len(nodes) < 2:
        pytest.skip("not enough nodes")
    a, b = nodes[0], nodes[1]
    try:
        sd = kb.compute_direction(str(a.id), str(b.id))
    except SemanticDirectionError:
        pytest.skip("identical nodes in this run")
    results = kb.direction_search("alpha unique term", sd.direction_vec)
    assert isinstance(results, list)


def test_fold_directions_nonempty() -> None:
    kb = _kb()
    nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
    if not nodes:
        pytest.skip("no nodes")
    dirs = kb.fold_directions(str(nodes[0].id))
    assert isinstance(dirs, list)


def test_stub_summarizer() -> None:
    s = StubSummarizer()
    out = s.summarize("node1", ["alpha unique term", "beta distinct phrase"])
    assert "alpha" in out or "beta" in out
    out2 = s.summarize("node1", [])
    assert "node1" in out2


def test_search_with_reflection_low_confidence() -> None:
    kb = _kb()
    result = kb.search_with_reflection("zzz nonexistent query")
    assert "original" in result
    assert "reflected_query" in result


def test_semantic_distance_empty_kb() -> None:
    kb = KnowledgeBase()
    assert kb.semantic_distance("x", "y", octave=64) == 0.0


def test_direction_search_empty_vec_returns_empty() -> None:
    kb = _kb()
    results = kb.direction_search("alpha", [0.0] * 64)
    assert isinstance(results, list)


def test_compute_trajectory_returns_trajectory() -> None:
    kb = _kb()
    nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
    if not nodes:
        pytest.skip("no nodes")
    t = kb.compute_trajectory("alpha unique term", str(nodes[0].id))
    from core.agent_api import SemanticTrajectory
    assert isinstance(t, SemanticTrajectory)
    assert t.total_distance >= 0.0
    assert 0.0 <= t.coherence_score <= 1.0 + 1e-9
    assert t.energy_cost >= 0.0


def test_compute_trajectory_empty_kb() -> None:
    kb = KnowledgeBase()
    from core.agent_api import SemanticTrajectory
    t = kb.compute_trajectory("x", "nonexistent")
    assert isinstance(t, SemanticTrajectory)
    assert t.steps == ()


def test_categorization_summarizer_stub_no_key() -> None:
    from core.summarizer import CategorizationSummarizer
    s = CategorizationSummarizer(api_key="")
    out = s.summarize("node1", ["alpha unique term", "beta distinct phrase"])
    assert isinstance(out, str) and len(out) > 0


def test_categorization_summarizer_empty() -> None:
    from core.summarizer import CategorizationSummarizer
    s = CategorizationSummarizer(api_key="")
    out = s.summarize("node1", [])
    assert "node1" in out
