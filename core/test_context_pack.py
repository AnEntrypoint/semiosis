"""Property and edge-case tests for the token-budgeted context pack builder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.context_pack import ContextPack, HeuristicTokenCounter  # noqa: E402

FACTS = [
    "instanced drawing reduces draw calls",
    "vertex array objects bind attributes once",
    "compressed textures save gpu memory",
    "frustum culling skips offscreen geometry",
    "state sorting minimizes shader switches",
    "bufferSubData avoids per-frame reallocation",
]


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.ingest(FACTS)
    return kb


def test_heuristic_counter_positive() -> None:
    assert HeuristicTokenCounter().count("hello world") >= 1


def test_budget_monotonicity() -> None:
    kb = _kb()
    for budget in (16, 64, 256, 1024):
        pack = kb.build_context_pack("draw call", max_tokens=budget)
        assert pack.total_tokens <= budget


def test_budget_zero_is_empty_truncated() -> None:
    kb = _kb()
    pack = kb.build_context_pack("draw call", max_tokens=0)
    assert pack.entries == ()
    assert pack.truncated is True


def test_negative_budget_raises() -> None:
    kb = _kb()
    with pytest.raises(ValueError):
        kb.build_context_pack("draw call", max_tokens=-5)


def test_empty_kb_returns_empty_pack() -> None:
    kb = KnowledgeBase()
    pack = kb.build_context_pack("anything", max_tokens=128)
    assert isinstance(pack, ContextPack)
    assert pack.entries == ()


def test_no_redundant_entries() -> None:
    kb = _kb()
    pack = kb.build_context_pack("draw call", max_tokens=512)
    engine = kb._pipeline.engine
    store = kb._pipeline.store
    full = [e for e in pack.entries if not e.is_summary]
    thr = kb._settings.context.overlap_threshold
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            a, b = store.get(full[i].node_id), store.get(full[j].node_id)
            assert engine.overlap_score(a, b) <= thr


def test_determinism_same_settings() -> None:
    p1 = _kb().build_context_pack("texture memory", max_tokens=256)
    p2 = _kb().build_context_pack("texture memory", max_tokens=256)
    assert [e.text for e in p1.entries] == [e.text for e in p2.entries]


def test_render_is_string() -> None:
    kb = _kb()
    pack = kb.build_context_pack("draw call", max_tokens=256)
    assert isinstance(pack.render(), str)
