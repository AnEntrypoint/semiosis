"""KnowledgeBase mixin: save/load snapshot round-tripping."""
from __future__ import annotations

import json
import os

from .settings import Settings


class PersistenceMixin:
    """JSON snapshot save/load; cones are rebuilt deterministically from texts on load."""

    def save(self, path: "str | os.PathLike[str]") -> None:
        """Persist texts, usage, facts, session, and a Settings snapshot for reproducible reload."""
        data = {
            "version": 1,
            "texts": self._texts,
            "usage": self._usage,
            "metrics": self._metrics,
            "memory": self._memory.snapshot(),
            "settings": self._settings.model_dump(mode="json"),
            "commit": str(self._pipeline.commit) if self._pipeline else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: "str | os.PathLike[str]", settings: Settings | None = None) -> "PersistenceMixin":
        """Reconstruct a KnowledgeBase from a save(); cones rebuilt deterministically from texts."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        kb = cls(settings or Settings(**data.get("settings", {})))
        kb._usage = {str(k): int(v) for k, v in data.get("usage", {}).items()}
        kb._metrics.update(data.get("metrics", {}))
        if data.get("texts"):
            kb.ingest(list(data["texts"]))
        kb._memory.restore(data.get("memory", {}))
        return kb
