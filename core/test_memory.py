"""Tests for the layered SemioticMemory."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.semiotic_memory import SemioticMemory  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = [
    "instanced drawing reduces draw calls",
    "compressed textures save gpu memory",
    "frustum culling skips offscreen geometry",
    "state sorting minimizes shader switches",
]


def test_remember_and_forget_round_trip() -> None:
    kb = KnowledgeBase()
    kb.ingest(FACTS)
    fid = kb.remember("user prefers ascii", "p1")
    assert fid == "p1"
    block = kb.recall("draw call", budget_tokens=200)
    assert "ascii" in block
    assert kb.forget("p1") is True
    assert kb.forget("p1") is False


def test_budget_zero_keeps_pinned_fact() -> None:
    kb = KnowledgeBase()
    kb.ingest(FACTS)
    kb.remember("pinned fact text", "p1")
    block = kb.recall("anything", budget_tokens=0)
    assert "pinned fact text" in block


def test_empty_kb_recall_is_safe() -> None:
    kb = KnowledgeBase()
    block = kb.recall("nothing here", budget_tokens=128)
    assert isinstance(block, str)


def test_max_pinned_eviction() -> None:
    s = Settings()
    s.memory.max_pinned = 2
    mem = SemioticMemory(None, [], s)
    mem.remember("a", "f1")
    mem.remember("b", "f2")
    mem.remember("c", "f3")
    ids = {f.id for f in mem.facts()}
    assert len(ids) == 2
    assert "f1" not in ids


def test_assemble_respects_budget() -> None:
    kb = KnowledgeBase()
    kb.ingest(FACTS)
    block = kb.recall("draw call", budget_tokens=80)
    counter_tokens = max(1, len(block) // 4)
    assert counter_tokens <= 80 + 40
