"""KnowledgeBase mixin: save/load; primary store is a human-readable markdown tree."""
from __future__ import annotations

import json
import os

from .serialization import cone_node_from_dict, cone_node_to_dict
from .settings import Settings


def _restore(kb, texts: list[str], nodes: list) -> None:
    """Rehydrate pipeline + BM25 from restored nodes without re-encoding or refitting."""
    from .pipeline import KnowledgePipeline
    kb._texts = list(texts)
    for i, t in enumerate(texts):
        kb._bm25.add(str(i), t)
    if nodes and texts:
        kb._pipeline = KnowledgePipeline(texts, kb._settings, prebuilt_nodes=nodes)
        kb._memory._pipeline = kb._pipeline
        kb._memory._texts = list(texts)
    elif texts:
        kb._texts = []
        kb.ingest(list(texts))


class PersistenceMixin:
    """Markdown-tree save/load (directory path) with a legacy JSON snapshot path (*.json)."""

    def _snapshot_meta(self) -> dict:
        return {
            "usage": self._usage,
            "metrics": self._metrics,
            "memory": self._memory.snapshot(),
            "settings": self._settings.model_dump(mode="json"),
            "commit": str(self._pipeline.commit) if self._pipeline else None,
        }

    def save(self, path: "str | os.PathLike[str]") -> None:
        """Directory path -> browsable markdown tree + _meta; *.json path -> single-file snapshot."""
        nodes = self._pipeline.store.all_nodes() if self._pipeline else []
        if str(path).endswith(".json"):
            data = {"version": 2, "texts": self._texts,
                    "nodes": [cone_node_to_dict(n) for n in nodes]}
            data.update(self._snapshot_meta())
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return
        from .markdown_store import save_markdown_tree
        save_markdown_tree(path, self._texts, nodes, self._snapshot_meta())

    @classmethod
    def load(cls, path: "str | os.PathLike[str]", settings: Settings | None = None) -> "PersistenceMixin":
        """Reconstruct a KnowledgeBase; cones restore verbatim (no refit); v1 JSON falls back to re-ingest."""
        if str(path).endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            from .markdown_store import load_markdown_tree
            texts, nodes, meta = load_markdown_tree(path)
            data = dict(meta)
            data["texts"] = texts
            data["_nodes_loaded"] = nodes
        kb = cls(settings or Settings(**data.get("settings", {})))
        kb._usage = {str(k): int(v) for k, v in data.get("usage", {}).items()}
        kb._metrics.update(data.get("metrics", {}))
        nodes = data.get("_nodes_loaded")
        if nodes is None:
            nodes = [cone_node_from_dict(d) for d in data.get("nodes", [])]
        _restore(kb, list(data.get("texts", [])), nodes)
        kb._memory.restore(data.get("memory", {}))
        return kb
