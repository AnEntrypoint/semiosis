"""Tests for recursive octave-descent retrieval."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.recursive import RecursiveAnswerEngine  # noqa: E402

FACTS = [
    "instanced drawing reduces draw calls",
    "vertex array objects bind attributes once",
    "compressed textures save gpu memory",
    "frustum culling skips offscreen geometry",
    "state sorting minimizes shader switches",
    "mipmaps reduce texture sampling cost",
]


def _engine() -> RecursiveAnswerEngine:
    kb = KnowledgeBase()
    kb.ingest(FACTS)
    return RecursiveAnswerEngine(kb._pipeline, max_depth=3, beam_k=2)


def test_decompose_splits_compound() -> None:
    eng = _engine()
    assert len(eng.decompose("draw calls and textures")) == 2
    assert len(eng.decompose("just one thing")) == 1


def test_answer_terminates_with_bounded_depth() -> None:
    eng = _engine()
    result = eng.answer("compress textures")
    assert result.depth_reached <= 3


def test_answer_merges_subqueries() -> None:
    eng = _engine()
    result = eng.answer("draw calls and texture memory")
    assert len(result.sub_answers) == 2
    ids = list(result.evidence_node_ids)
    assert len(ids) == len(set(ids))


def test_empty_pipeline_returns_empty() -> None:
    kb = KnowledgeBase()
    eng = RecursiveAnswerEngine(kb._pipeline or _DummyNone())
    assert eng.answer("x").evidence_node_ids == ()


class _DummyNone:
    store = None
