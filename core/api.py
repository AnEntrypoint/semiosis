"""FastAPI serving surface -- health, readiness, and agent-callable KnowledgeBase endpoints."""
from __future__ import annotations

import dataclasses
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from .agent_api import KnowledgeBase
from .settings import Settings

_settings: Settings | None = None
_kb: KnowledgeBase | None = None

TOOL_MANIFEST: list[dict[str, Any]] = [
    {"name": "search", "method": "POST", "path": "/search",
     "description": "Diversified scored hits with provenance.",
     "params": {"query": "str", "k": "int>0"}},
    {"name": "recall", "method": "POST", "path": "/recall",
     "description": "Layered memory block under a token budget.",
     "params": {"query": "str", "budget_tokens": "int>=0?"}},
    {"name": "context_pack", "method": "POST", "path": "/context_pack",
     "description": "Token-budgeted, redundancy-free context pack.",
     "params": {"query": "str", "max_tokens": "int>=0"}},
    {"name": "navigate", "method": "POST", "path": "/navigate",
     "description": "Neighbor cones ranked by entailment gradient.",
     "params": {"query": "str", "k": "int>0"}},
    {"name": "deep_search", "method": "POST", "path": "/deep_search",
     "description": "Recursive octave-descent retrieval.",
     "params": {"query": "str", "k": "int>0"}},
    {"name": "tension", "method": "POST", "path": "/tension",
     "description": "Worst redundancy/contradiction pairs.",
     "params": {"top_n": "int>0"}},
    {"name": "diagnose", "method": "GET", "path": "/diagnose",
     "description": "KB health snapshot.", "params": {}},
]


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(_get_settings())
    return _kb


def _check_query(query: Any) -> str:
    if not isinstance(query, str) or not query:
        raise HTTPException(status_code=422, detail="query must be a non-empty string")
    if len(query) > _get_settings().agent.max_query_chars:
        raise HTTPException(status_code=422, detail="query too long")
    return query


def _check_k(k: Any, name: str = "k") -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise HTTPException(status_code=422, detail=f"{name} must be a positive int")
    return k


def create_app(settings: Settings | None = None, kb: KnowledgeBase | None = None) -> "FastAPI":
    """Return a configured FastAPI app exposing health, readiness, and agent KB endpoints."""
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI is required; install the 'serving' extra.")

    global _settings, _kb
    if settings is not None:
        _settings = settings
    if kb is not None:
        _kb = kb

    app = FastAPI(title="semiosis", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        """Always 200 while the process is alive."""
        return JSONResponse({"status": "ok", "env": _get_settings().env.value})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """200 when the KB is warm (has ingested nodes); 503 otherwise."""
        k = _get_kb()
        warm = k._pipeline is not None and bool(k._pipeline.store.all_nodes())
        if warm:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready", "reason": "kb empty"}, status_code=503)

    @app.get("/tools")
    async def tools() -> JSONResponse:
        """Capability manifest for agent discovery (MCP-shaped)."""
        return JSONResponse({"tools": TOOL_MANIFEST})

    @app.post("/ingest")
    async def ingest(body: dict) -> JSONResponse:
        texts = body.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise HTTPException(status_code=422, detail="texts must be a list of strings")
        _get_kb().ingest(texts)
        return JSONResponse({"ingested": len(texts)})

    @app.post("/search")
    async def search(body: dict) -> JSONResponse:
        q = _check_query(body.get("query"))
        k = _check_k(body.get("k", 5))
        return JSONResponse({"hits": [dataclasses.asdict(h) for h in _get_kb().search(q, k)]})

    @app.post("/recall")
    async def recall(body: dict) -> JSONResponse:
        q = _check_query(body.get("query"))
        bt = body.get("budget_tokens")
        if bt is not None and (not isinstance(bt, int) or isinstance(bt, bool) or bt < 0):
            raise HTTPException(status_code=422, detail="budget_tokens must be a non-negative int")
        return JSONResponse({"block": _get_kb().recall(q, bt)})

    @app.post("/context_pack")
    async def context_pack(body: dict) -> JSONResponse:
        q = _check_query(body.get("query"))
        mt = body.get("max_tokens")
        if not isinstance(mt, int) or isinstance(mt, bool) or mt < 0:
            raise HTTPException(status_code=422, detail="max_tokens must be a non-negative int")
        pack = _get_kb().build_context_pack(q, mt)
        return JSONResponse({"pack": dataclasses.asdict(pack)})

    @app.post("/navigate")
    async def navigate(body: dict) -> JSONResponse:
        q = _check_query(body.get("query"))
        k = _check_k(body.get("k", 5))
        return JSONResponse({"neighbors": [dataclasses.asdict(n) for n in _get_kb().navigate(q, k)]})

    @app.post("/deep_search")
    async def deep_search(body: dict) -> JSONResponse:
        q = _check_query(body.get("query"))
        k = _check_k(body.get("k", 5))
        return JSONResponse({"result": dataclasses.asdict(_get_kb().deep_search(q, k))})

    @app.post("/tension")
    async def tension(body: dict) -> JSONResponse:
        top_n = _check_k(body.get("top_n", 10), "top_n")
        return JSONResponse({"pairs": [dataclasses.asdict(p) for p in _get_kb().scan_tension(top_n)]})

    @app.get("/diagnose")
    async def diagnose() -> JSONResponse:
        rep = _get_kb().diagnose()
        d = dataclasses.asdict(rep)
        d["failure_mode"] = rep.failure_mode.value
        return JSONResponse({"report": d})

    return app
