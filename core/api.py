"""FastAPI serving surface -- /health and /ready; import requires the 'serving' extra."""
from __future__ import annotations

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False

from .settings import Settings

_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def create_app(settings: Settings | None = None) -> "FastAPI":
    """Return a configured FastAPI application; settings defaults to env-loaded Settings."""
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI is required; install the 'serving' extra.")

    global _settings
    if settings is not None:
        _settings = settings

    app = FastAPI(title="semiosis", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        """Always 200 while the process is alive."""
        return JSONResponse({"status": "ok", "env": _get_settings().env.value})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """200 when the cone engine and store are warm; 503 otherwise."""
        # stub: readiness = process alive + settings loadable
        try:
            _get_settings()
            return JSONResponse({"status": "ready"})
        except Exception as exc:
            return JSONResponse({"status": "not_ready", "reason": str(exc)}, status_code=503)

    return app
