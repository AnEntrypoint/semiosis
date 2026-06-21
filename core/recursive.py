"""Recursive octave-descent retrieval; treats the cone hierarchy as an external environment (RLM)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    """Decompose a query, recurse coarse-to-fine over octaves, merge bounded evidence."""

    def __init__(self, pipeline, max_depth: int = 4, max_breadth: int = 8,
                 beam_k: int = 3, min_aperture_stop: float = 0.1) -> None:
        self._pipeline = pipeline
        self._max_depth = max(1, max_depth)
        self._max_breadth = max(1, max_breadth)
        self._beam_k = max(1, beam_k)
        self._min_aperture = min_aperture_stop
        self._encode_cache: dict[str, object] = {}

    def _octaves(self) -> list[Prefix]:
        return [Prefix(d) for d in sorted(self._pipeline._encoder.dims)]

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

    def decompose_by_octaves(self, query: str) -> dict[int, list[str]]:
        """Decompose query clauses by octave affinity: route each clause to its best-matching octave level."""
        clauses = self.decompose(query)
        octaves = self._octaves()
        result: dict[int, list[str]] = {i: [] for i in range(len(octaves))}
        pipeline = self._pipeline
        for clause in clauses:
            q_vec = self._encode(clause)
            best_octave = 0
            best_score = -1.0
            for octave_idx, prefix in enumerate(octaves):
                knn_results = pipeline.store.knn_scored(q_vec[:prefix], k=1, prefix=prefix)
                if knn_results:
                    _, score = knn_results[0]
                    if score > best_score:
                        best_score = score
                        best_octave = octave_idx
            result[best_octave].append(clause)
        return result

    def descend(self, q_vec, octave_idx: int, beam_k: int, max_depth: int,
                visited: "set[NodeId] | None" = None, trace: "list | None" = None,
                depth: int = 0) -> list[NodeId]:
        """Recurse octave_idx coarse->fine, gathering evidence node ids under depth/breadth bounds."""
        pipeline = self._pipeline
        octaves = self._octaves()
        if visited is None:
            visited = set()
        if octave_idx >= len(octaves) or depth >= max_depth:
            return []
        prefix = octaves[octave_idx]
        ids = list(pipeline.query.knn(q_vec[:prefix], k=self._max_breadth, prefix=prefix))
        frontier = [i for i in ids if i not in visited][:beam_k]
        store = pipeline.store
        engine = pipeline.engine
        evidence: list[NodeId] = []
        is_finest = octave_idx == len(octaves) - 1
        finer_prefix = None if is_finest else octaves[octave_idx + 1]
        member_map = store.members_to_nodes(finer_prefix) if finer_prefix is not None else {}
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            node = store.get(nid)
            if trace is not None:
                trace.append((octave_idx, str(nid), float(node.aperture)))
            stop = is_finest or node.aperture < self._min_aperture
            if stop:
                evidence.append(nid)
                continue
            child_ids = {member_map[m] for m in node.members if m in member_map}
            gated = [c for c in child_ids if c not in visited
                     and engine.contains(node, store.get(c)) > 0.0]
            if not gated:
                gated = [c for c in child_ids if c not in visited]
            if not gated:
                evidence.append(nid)
                continue
            for c in gated[:beam_k]:
                sub = self.descend(q_vec, octave_idx + 1, beam_k, max_depth,
                                   visited, trace, depth + 1)
                evidence.extend(sub or [c])
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
        per_depth = max(1, self._max_depth)
        subs = [self._answer_one(c, per_beam, per_depth) for c in clauses]
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
        )

    def _answer_one(self, query: str, beam_k: "int | None" = None,
                    max_depth: "int | None" = None) -> RecursiveResult:
        q_vec = self._encode(query)
        trace: list = []
        ids = self.descend(q_vec, 0, beam_k or self._beam_k,
                           max_depth or self._max_depth, set(), trace)
        return RecursiveResult(
            answer=query if ids else None,
            evidence_node_ids=tuple(ids),
            evidence_texts=tuple(self._evidence_texts(ids)),
            depth_reached=max((t[0] for t in trace), default=0),
            trace=tuple(trace),
        )
