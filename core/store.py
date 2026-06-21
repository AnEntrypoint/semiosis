"""In-memory Store and Query stubs satisfying the Store/Query protocols; for testing only."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .interfaces import CommitId, ConeNode, EuclideanVec, NodeId, Phrase, Prefix
from .serialization import cone_node_from_dict, cone_node_to_dict

if TYPE_CHECKING:
    from .cone_engine import HyperbolicConeEngine


class InMemoryStore:
    """Satisfies the Store protocol with a plain dict; no HNSW, no lakeFS."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, ConeNode] = {}

    def write(self, nodes: Sequence[ConeNode], at: CommitId) -> CommitId:
        """Store nodes under the given commit handle and return it."""
        for n in nodes:
            self._nodes[n.id] = n
        return at

    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]:
        """Return up to k node IDs ranked by cosine similarity to q on spatial apex dims."""
        if not self._nodes:
            return []
        q_flat = np.asarray(q, dtype=np.float32).ravel()[:prefix]
        scores: list[tuple[float, NodeId]] = []
        for nid, node in self._nodes.items():
            vec = self._retrieval_vec(node, prefix)
            length = min(len(vec), len(q_flat))
            s = float(np.dot(vec[:length], q_flat[:length]))
            scores.append((s, nid))
        scores.sort(reverse=True)
        return [nid for _, nid in scores[:k]]

    @staticmethod
    def _retrieval_vec(node: ConeNode, prefix: Prefix) -> np.ndarray:
        """Embedding centroid (prefix-sliced) when present, else the cone apex spatial dims."""
        if node.centroid is not None:
            return np.asarray(node.centroid, dtype=np.float32)[:prefix]
        return node.apex[1:prefix + 1].astype(np.float32)

    def knn_scored(self, q: EuclideanVec, k: int, prefix: Prefix) -> list[tuple[NodeId, float]]:
        """Like knn but return (node_id, cosine-in-[0,1]) pairs for calibrated relevance."""
        if not self._nodes:
            return []
        q_flat = np.asarray(q, dtype=np.float32).ravel()[:prefix]
        qn = float(np.linalg.norm(q_flat)) or 1.0
        scores: list[tuple[float, NodeId]] = []
        for nid, node in self._nodes.items():
            vec = self._retrieval_vec(node, prefix)
            length = min(len(vec), len(q_flat))
            sn = float(np.linalg.norm(vec[:length])) or 1.0
            cos = float(np.dot(vec[:length], q_flat[:length]) / (qn * sn))
            scores.append((cos, nid))
        scores.sort(reverse=True)
        return [(nid, (c + 1.0) / 2.0) for c, nid in scores[:k]]

    def upsert(self, node: ConeNode) -> None:
        self._nodes[node.id] = node

    def get(self, nid: NodeId) -> ConeNode:
        return self._nodes[nid]

    def all_nodes(self) -> list[ConeNode]:
        return list(self._nodes.values())

    def nodes_at(self, prefix: Prefix) -> list[ConeNode]:
        """Return nodes belonging to one Matryoshka octave (prefix)."""
        return [n for n in self._nodes.values() if n.prefix == prefix]

    def members_to_nodes(self, prefix: Prefix) -> dict[PhraseId, NodeId]:
        """Reverse map PhraseId -> NodeId at a given octave; the recurse-into-members edge."""
        out: dict[PhraseId, NodeId] = {}
        for n in self._nodes.values():
            if n.prefix != prefix:
                continue
            for m in n.members:
                out[m] = n.id
        return out

    def save(self, path: "str | os.PathLike[str]") -> None:
        """Write all nodes to a JSON file; lossless via cone_node_to_dict."""
        data = [cone_node_to_dict(n) for n in self._nodes.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: "str | os.PathLike[str]") -> None:
        """Read nodes from a JSON file written by save(); merges into existing store."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            node = cone_node_from_dict(d)
            self._nodes[node.id] = node


class InMemoryQuery:
    """Satisfies the Query protocol backed by InMemoryStore + HyperbolicConeEngine."""

    def __init__(self, store: InMemoryStore, engine: "HyperbolicConeEngine") -> None:
        self._store = store
        self._engine = engine

    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]:
        return self._store.knn(q, k, prefix)

    def containment_score(self, parent: NodeId, child: NodeId) -> float:
        return self._engine.contains(self._store.get(parent), self._store.get(child))

    def analogy(self, a: NodeId, b: NodeId, c: NodeId) -> Sequence[NodeId]:
        # stub: return top-k neighbors of c (proper impl needs parallel transport)
        node_c = self._store.get(c)
        q = node_c.apex[1:].astype(np.float32)
        return self._store.knn(q, k=5, prefix=Prefix(len(q)))

    def overlap_nodes(self, node: NodeId, threshold: float) -> Sequence[NodeId]:
        """Return nodes whose symmetric overlap with node exceeds threshold."""
        n = self._store.get(node)
        return [
            c.id for c in self._store.all_nodes()
            if c.id != node and self._engine.overlap_score(n, c) > threshold
        ]

    def tension_nodes(self, node: NodeId, threshold: float) -> Sequence[NodeId]:
        """Return same-octave nodes whose semantic tension with node exceeds threshold."""
        n = self._store.get(node)
        return [
            c.id for c in self._store.all_nodes()
            if c.id != node and c.prefix == n.prefix and self._engine.tension(n, c) > threshold
        ]
