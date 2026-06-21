"""Tests for the FastAPI serving surface over a warm KnowledgeBase."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from core.agent_api import KnowledgeBase  # noqa: E402
from core.api import create_app  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = ["alpha unique term", "beta distinct phrase", "gamma separate idea", "delta other concept"]


def _client() -> TestClient:
    s = Settings()
    s.cone.epochs = 4
    kb = KnowledgeBase(s)
    kb.ingest(FACTS)
    return TestClient(create_app(s, kb))


def test_health_ok() -> None:
    assert _client().get("/health").status_code == 200


def test_ready_warm() -> None:
    assert _client().get("/ready").json()["status"] == "ready"


def test_tools_manifest() -> None:
    tools = _client().get("/tools").json()["tools"]
    assert any(t["name"] == "search" for t in tools)


def test_search_endpoint() -> None:
    r = _client().post("/search", json={"query": "alpha unique term", "k": 2})
    assert r.status_code == 200
    assert len(r.json()["hits"]) <= 2


def test_recall_endpoint() -> None:
    r = _client().post("/recall", json={"query": "alpha", "budget_tokens": 200})
    assert r.status_code == 200 and isinstance(r.json()["block"], str)


def test_context_pack_endpoint() -> None:
    r = _client().post("/context_pack", json={"query": "alpha", "max_tokens": 128})
    assert r.status_code == 200


def test_diagnose_endpoint() -> None:
    assert _client().get("/diagnose").json()["report"]["nodes"] > 0


def test_search_validation_422() -> None:
    c = _client()
    assert c.post("/search", json={"query": "", "k": 2}).status_code == 422
    assert c.post("/search", json={"query": "x", "k": 0}).status_code == 422
    assert c.post("/context_pack", json={"query": "x", "max_tokens": -1}).status_code == 422
