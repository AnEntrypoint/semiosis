"""In-memory Store + Query implementations; leaf-scoped centroid knn over the recursive cone tree."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .interfaces import CommitId, ConeNode, EuclideanVec, NodeId, PhraseId, Prefix
from .locality_index import CatapultCache, HilbertBucketIndex
from .serialization import cone_node_from_dict, cone_node_to_dict

if TYPE_CHECKING:
    from .cone_engine import HyperbolicConeEngine


class InMemoryStore:
    """Satisfies the Store protocol with a plain dict; commit ids are uuid handles."""

    def __init__(self, n_partitions: int = 16, catapult_size: int = 512) -> None:
        self._nodes: dict[NodeId, ConeNode] = {}
        self._locality = HilbertBucketIndex(n_partitions=n_partitions)
        self._catapult = CatapultCache(max_size=catapult_size)
        self._member_index_cache: dict[Prefix, dict[PhraseId, NodeId]] = {}
        self._children_cache: dict[Prefix, dict[NodeId, list[NodeId]]] = {}

    def _index_node(self, node: ConeNode) -> None:
        vec = self._retrieval_vec(node, node.prefix)
        if len(vec):
            self._locality.upsert(node.id, node.prefix, vec)

    def _invalidate(self, prefix: Prefix) -> None:
        self._member_index_cache.pop(prefix, None)
        self._children_cache.pop(prefix, None)

    def write(self, nodes: Sequence[ConeNode], at: CommitId) -> CommitId:
        """Store nodes under the given commit handle and return it."""
        for n in nodes:
            self._nodes[n.id] = n
            self._index_node(n)
            self._invalidate(n.prefix)
        return at

    def children_of(self, prefix: Prefix) -> dict[NodeId, list[NodeId]]:
        """Parent -> children map at one octave, derived from ConeNode.parent; cached."""
        if prefix in self._children_cache:
            return self._children_cache[prefix]
        out: dict[NodeId, list[NodeId]] = {}
        for n in self._nodes.values():
            if n.prefix == prefix and n.parent is not None:
                out.setdefault(n.parent, []).append(n.id)
        self._children_cache[prefix] = out
        return out

    def leaves_at(self, prefix: Prefix) -> list[ConeNode]:
        """Nodes at this octave with no children -- the retrieval candidate pool."""
        children = self.children_of(prefix)
        return [n for n in self._nodes.values()
                if n.prefix == prefix and n.id not in children]

    def _candidate_nodes(self, prefix: Prefix, q_flat: np.ndarray,
                         leaf_only: bool = True) -> dict[NodeId, ConeNode]:
        """Leaf pool (default), Hilbert-pruned when the octave is large enough; else full scan."""
        if leaf_only:
            pool = {n.id: n for n in self.leaves_at(prefix)}
        else:
            pool = {nid: n for nid, n in self._nodes.items() if n.prefix == prefix}
        cand_ids = self._locality.candidates(prefix, q_flat)
        if cand_ids is None:
            return pool
        pruned = {nid: pool[nid] for nid in cand_ids if nid in pool}
        return pruned or pool

    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]:
        """Return up to k leaf node IDs ranked by dot product to q on centroid dims."""
        if not self._nodes:
            return []
        q_flat = np.asarray(q, dtype=np.float32).ravel()[:prefix]
        candidates = self._candidate_nodes(prefix, q_flat)
        scores: list[tuple[float, NodeId]] = []
        for nid, node in candidates.items():
            vec = self._retrieval_vec(node, prefix)
            length = min(len(vec), len(q_flat))
            s = float(np.dot(vec[:length], q_flat[:length]))
            scores.append((s, nid))
        scores.sort(reverse=True)
        result = [nid for _, nid in scores[:k]]
        if result:
            self._catapult.record(prefix, q_flat, result[0])
        return result

    @staticmethod
    def _retrieval_vec(node: ConeNode, prefix: Prefix) -> np.ndarray:
        """Embedding centroid (prefix-sliced) when present, else the cone apex spatial dims."""
        if node.centroid is not None:
            return np.asarray(node.centroid, dtype=np.float32)[:prefix]
        return node.apex[1:prefix + 1].astype(np.float32)

    def knn_scored(self, q: EuclideanVec, k: int, prefix: Prefix,
                   entropy_weight: float = 0.0) -> list[tuple[NodeId, float]]:
        """Like knn but return (node_id, cosine-in-[0,1]) pairs for calibrated relevance."""
        if not self._nodes:
            return []
        q_flat = np.asarray(q, dtype=np.float32).ravel()[:prefix]
        qn = float(np.linalg.norm(q_flat)) or 1.0
        candidates = self._candidate_nodes(prefix, q_flat)
        scores: list[tuple[float, NodeId]] = []
        for nid, node in candidates.items():
            vec = self._retrieval_vec(node, prefix)
            length = min(len(vec), len(q_flat))
            sn = float(np.linalg.norm(vec[:length])) or 1.0
            cos = float(np.dot(vec[:length], q_flat[:length]) / (qn * sn))
            # entropy weight: penalize wide-aperture (diffuse) clusters using the fitted
            # cone aperture as the entropy proxy -- no per-member vectors are stored here,
            # so aperture (already entropy-tuned upstream) is the available uncertainty signal.
            if entropy_weight > 0 and node.members:
                h = min(1.0, float(getattr(node, "aperture", 0.0)) / (np.pi / 2))
                cos = cos * (1.0 - entropy_weight * h)
            scores.append((cos, nid))
        scores.sort(reverse=True)
        result = scores[:k]
        if result:
            self._catapult.record(prefix, q_flat, result[0][1])
        return [(nid, (c + 1.0) / 2.0) for c, nid in result]

    def upsert(self, node: ConeNode) -> None:
        self._nodes[node.id] = node
        self._index_node(node)
        self._invalidate(node.prefix)

    def delete(self, nid: NodeId) -> bool:
        """Remove a node by id; returns False if it was already absent."""
        node = self._nodes.pop(nid, None)
        if node is None:
            return False
        self._locality.remove(nid, node.prefix)
        self._invalidate(node.prefix)
        return True

    def get(self, nid: NodeId) -> ConeNode:
        return self._nodes[nid]

    def all_nodes(self) -> list[ConeNode]:
        return list(self._nodes.values())

    def nodes_by_id(self) -> dict[NodeId, ConeNode]:
        """Read-only view of the id->node map; callers must not mutate the store through it."""
        return dict(self._nodes)

    def nodes_at(self, prefix: Prefix) -> list[ConeNode]:
        """Return nodes belonging to one Matryoshka octave (prefix)."""
        return [n for n in self._nodes.values() if n.prefix == prefix]

    def members_to_nodes(self, prefix: Prefix) -> dict[PhraseId, NodeId]:
        """Reverse map PhraseId -> leaf NodeId at a given octave; cached, invalidated on upsert/write."""
        if prefix in self._member_index_cache:
            return self._member_index_cache[prefix]
        out: dict[PhraseId, NodeId] = {}
        for n in self.leaves_at(prefix):
            for m in n.members:
                out[m] = n.id
        self._member_index_cache[prefix] = out
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
            self._index_node(node)
            self._invalidate(node.prefix)


class InMemoryQuery:
    """Satisfies the Query protocol backed by InMemoryStore + HyperbolicConeEngine."""

    def __init__(self, store: InMemoryStore, engine: "HyperbolicConeEngine") -> None:
        self._store = store
        self._engine = engine

    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]:
        return self._store.knn(q, k, prefix)

    def containment_score(self, parent: NodeId, child: NodeId) -> float:
        return self._engine.contains(self._store.get(parent), self._store.get(child))

    def analogy(self, a: NodeId, b: NodeId, c: NodeId, k: int = 5) -> Sequence[NodeId]:
        """A:B :: C:X via centroid arithmetic: knn around centroid(c) + (centroid(b) - centroid(a))."""
        def _vec(nid: NodeId) -> np.ndarray:
            n = self._store.get(nid)
            if n.centroid is not None:
                return np.asarray(n.centroid, dtype=np.float32)
            return n.apex[1:].astype(np.float32)
        va, vb, vc = _vec(a), _vec(b), _vec(c)
        m = min(len(va), len(vb), len(vc))
        q = vc[:m] + (vb[:m] - va[:m])
        return self._store.knn(q, k=k, prefix=Prefix(m))

    def overlap_nodes(self, node: NodeId, threshold: float) -> Sequence[NodeId]:
        """Return same-octave nodes whose centroid overlap with node exceeds threshold."""
        n = self._store.get(node)
        return [
            c.id for c in self._store.nodes_at(n.prefix)
            if c.id != node and self._engine.centroid_overlap(n, c) > threshold
        ]

    def tension_nodes(self, node: NodeId, threshold: float) -> Sequence[NodeId]:
        """Return same-octave nodes whose semantic tension with node exceeds threshold."""
        n = self._store.get(node)
        return [
            c.id for c in self._store.nodes_at(n.prefix)
            if c.id != node and self._engine.tension(n, c) > threshold
        ]
