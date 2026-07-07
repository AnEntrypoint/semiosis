"""Recursive tree-descent retrieval; treats the cone hierarchy as an external environment (RLM)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .interfaces import NodeId, Prefix, phrase_to_text_index

_SPLITS = (" and ", ";", " vs ", " versus ", " compare ")


@dataclass
class RecursiveResult:
    answer: "str | None" = None
    evidence_node_ids: tuple[NodeId, ...] = ()
    evidence_texts: tuple[str, ...] = ()
    depth_reached: int = 0
    sub_answers: tuple["RecursiveResult", ...] = ()
    trace: tuple[tuple[int, str, float], ...] = ()


class RecursiveAnswerEngine:
    """Decompose a query, beam-descend the within-octave cone tree, merge bounded evidence."""

    def __init__(self, pipeline, max_depth: int = 4, max_breadth: int = 8,
                 beam_k: int = 3, min_aperture_stop: float = 0.1) -> None:
        self._pipeline = pipeline
        self._max_depth = max(1, max_depth)
        self._max_breadth = max(1, max_breadth)
        self._beam_k = max(1, beam_k)
        self._min_aperture = min_aperture_stop
        self._encode_cache: dict[str, object] = {}

    def _finest(self) -> Prefix:
        return Prefix(int(sorted(int(d) for d in self._pipeline._encoder.dims)[-1]))

    def _encode(self, text: str):
        if text not in self._encode_cache:
            self._encode_cache[text] = self._pipeline._encoder.encode([text])[0]
        return self._encode_cache[text]

    def decompose(self, query: str) -> list[str]:
        """Rule-based split of a compound query into independently answerable clauses."""
        parts = [query]
        for sep in _SPLITS:
            nxt: list[str] = []
            for p in parts:
                nxt.extend(p.split(sep))
            parts = nxt
        return [c.strip() for c in parts if c.strip()] or [query.strip()]

    def descend(self, q_vec, prefix: Prefix, beam_k: int, max_depth: int,
                visited: "set[NodeId] | None" = None, trace: "list | None" = None) -> list[NodeId]:
        """Beam-walk one octave's cone tree from its roots toward leaves, containment-gated."""
        store = self._pipeline.store
        engine = self._pipeline.engine
        children_map = store.children_of(prefix)
        if visited is None:
            visited = set()
        roots = [n.id for n in store.nodes_at(prefix) if n.parent is None]
        if not roots:
            return []
        q = np.asarray(q_vec[:int(prefix)], dtype=np.float64)
        qn = float(np.linalg.norm(q)) or 1.0

        def _sim(nid: NodeId) -> float:
            node = store.get(nid)
            if not node.centroid:
                return -1e18
            v = np.asarray(node.centroid, dtype=np.float64)[:len(q)]
            return float(np.dot(v, q[:len(v)]) / ((float(np.linalg.norm(v)) or 1.0) * qn))

        evidence: list[NodeId] = []
        frontier = sorted(roots, key=_sim, reverse=True)[:beam_k]
        depth = 0
        while frontier and depth < max_depth:
            nxt: list[NodeId] = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                node = store.get(nid)
                if trace is not None:
                    trace.append((depth, str(nid), float(node.aperture)))
                kids = [c for c in children_map.get(nid, []) if c not in visited]
                if not kids or node.aperture < self._min_aperture:
                    evidence.append(nid)
                    continue
                gated = [c for c in kids if engine.contains(node, store.get(c)) > 0.0] or kids
                gated.sort(key=_sim, reverse=True)
                nxt.extend(gated[:self._max_breadth])
            frontier = sorted(dict.fromkeys(nxt), key=_sim, reverse=True)[:beam_k]
            depth += 1
        evidence.extend(n for n in frontier if n not in evidence)
        return list(dict.fromkeys(evidence))

    def _evidence_texts(self, ids: Sequence[NodeId]) -> list[str]:
        texts = getattr(self._pipeline, "_texts", None)
        if texts is None:
            return []
        out: list[str] = []
        store = self._pipeline.store
        for nid in ids:
            node = store.get(nid)
            for m in node.members:
                idx = phrase_to_text_index(m, len(texts))
                if idx is not None and texts[idx] not in out:
                    out.append(texts[idx])
                    break
        return out

    def answer(self, query: str) -> RecursiveResult:
        pipeline = self._pipeline
        if pipeline is None or getattr(pipeline, "store", None) is None \
                or not pipeline.store.all_nodes():
            return RecursiveResult()
        clauses = self.decompose(query)
        if len(clauses) == 1:
            return self._answer_one(clauses[0])
        per_beam = max(1, self._beam_k // len(clauses))
        subs = [self._answer_one(c, per_beam) for c in clauses]
        merged: list[NodeId] = []
        for s in subs:
            merged.extend(s.evidence_node_ids)
        merged = list(dict.fromkeys(merged))
        return RecursiveResult(
            answer="; ".join(c for c in clauses),
            evidence_node_ids=tuple(merged),
            evidence_texts=tuple(self._evidence_texts(merged)),
            depth_reached=max((s.depth_reached for s in subs), default=0),
            sub_answers=tuple(subs),
            trace=sum((s.trace for s in subs), ()),
        )

    def _answer_one(self, query: str, beam_k: "int | None" = None) -> RecursiveResult:
        q_vec = self._encode(query)
        trace: list = []
        ids = self.descend(q_vec, self._finest(), beam_k or self._beam_k,
                           self._max_depth, set(), trace)
        return RecursiveResult(
            answer=query if ids else None,
            evidence_node_ids=tuple(ids),
            evidence_texts=tuple(self._evidence_texts(ids)),
            depth_reached=max((t[0] for t in trace), default=0),
            trace=tuple(trace),
        )
