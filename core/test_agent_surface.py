"""Tests for the agent-facing KnowledgeBase surface: typed hits, MMR, learning loop, diagnose."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import DiagnoseReport, KnowledgeBase, SearchHit  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = [
    "alpha unique term",
    "beta distinct phrase",
    "gamma separate idea",
    "delta other concept",
    "epsilon final note",
    "zeta extra item",
    "eta seventh thing",
    "theta eighth thing",
]


def _kb(**agent) -> KnowledgeBase:
    s = Settings()
    s.cone.epochs = 4
    for key, val in agent.items():
        setattr(s.agent, key, val)
    kb = KnowledgeBase(s)
    kb.ingest(FACTS)
    return kb


def test_search_returns_typed_hits() -> None:
    kb = _kb()
    hits = kb.search("alpha unique term", k=3)
    assert hits and all(isinstance(h, SearchHit) for h in hits)
    assert all(0.0 <= h.score for h in hits)


def test_search_texts_back_compat() -> None:
    kb = _kb()
    assert isinstance(kb.search_texts("beta", k=2), list)


def test_exact_text_query_ranks_itself_first() -> None:
    kb = _kb()
    hits = kb.search("gamma separate idea", k=1)
    assert hits[0].text == "gamma separate idea"


def test_mmr_diversity_no_duplicate_text() -> None:
    kb = _kb(mmr_lambda=0.5)
    hits = kb.search("alpha", k=5)
    texts = [h.text for h in hits]
    assert len(texts) == len(set(texts))


def test_record_outcome_shifts_usage() -> None:
    kb = _kb(usage_weight=5.0)
    kb.record_outcome("alpha", ["zeta extra item"])
    kb.record_outcome("alpha", ["zeta extra item"])
    assert kb._usage["zeta extra item"] == 2


def test_record_outcome_ignores_unknown() -> None:
    kb = _kb()
    out = kb.record_outcome("q", ["not in kb"])
    assert out["applied"] == 0 and out["ignored"] == 1


def test_consolidate_idempotent() -> None:
    kb = _kb()
    r1 = kb.consolidate()
    r2 = kb.consolidate()
    assert "actions" in r1 and "actions" in r2


def test_diagnose_shape() -> None:
    kb = _kb()
    rep = kb.diagnose()
    assert isinstance(rep, DiagnoseReport)
    assert rep.octaves == len(kb._pipeline._encoder.dims)
    assert rep.nodes > 0


def test_batch_search() -> None:
    kb = _kb()
    out = kb.batch_search(["alpha", "beta"], k=2)
    assert len(out) == 2


def test_explain_retrieval_trace() -> None:
    kb = _kb()
    steps = kb.explain_retrieval("alpha unique term", k=2)
    assert steps and all(hasattr(s, "containment_to_top") for s in steps)


def test_octave_fusion_runs() -> None:
    kb = _kb(octave_fusion=True)
    hits = kb.search("delta other concept", k=2)
    assert hits


def test_metrics_counters() -> None:
    kb = _kb()
    kb.search("alpha", 2)
    m = kb.metrics()
    assert m["queries"] >= 1 and m["nodes"] > 0


def test_empty_kb_surface_safe() -> None:
    kb = KnowledgeBase()
    assert kb.search("x") == []
    assert kb.diagnose().nodes == 0
    assert kb.consolidate()["reason"] == "empty"
