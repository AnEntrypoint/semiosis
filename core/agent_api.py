"""High-level KnowledgeBase API for agents -- hides cone/manifold internals."""
from __future__ import annotations

from typing import Any

from .context_pack import ContextPack, ContextPackBuilder, ContextPackConfig
from .interfaces import Prefix, phrase_to_text_index
from .pipeline import KnowledgePipeline
from .recursive import RecursiveAnswerEngine, RecursiveResult
from .semiotic_memory import SemioticMemory
from .settings import Settings


class KnowledgeBase:
    """Ingest texts, search, navigate meaning flow, dispel tension, and recall layered memory."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._pipeline: KnowledgePipeline | None = None
        self._texts: list[str] = []
        self._memory = SemioticMemory(None, [], self._settings)

    def ingest(self, texts: list[str]) -> None:
        """Add texts and rebuild the semantic cone structure; pinned facts survive the rebuild."""
        self._texts.extend(texts)
        self._pipeline = KnowledgePipeline(self._texts, self._settings)
        self._memory._pipeline = self._pipeline
        self._memory._texts = list(self._texts)

    # --- helpers -------------------------------------------------------------
    def _primary_text(self, node) -> str:
        for m in node.members:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None:
                return self._texts[idx]
        return node.digest or node.label or ""

    def _resolve_top(self, query: str):
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        ids = self._pipeline.query.knn(q_vec[:prefix], k=1, prefix=prefix)
        return self._pipeline.store.get(ids[0]) if ids else None

    # --- retrieval -----------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[str]:
        """Return up to k texts closest to query by hyperbolic cone proximity."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return []
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        node_ids = self._pipeline.query.knn(q_vec[:prefix], k=k, prefix=prefix)
        results: list[str] = []
        for nid in node_ids:
            node = self._pipeline.store.get(nid)
            for phrase_id in node.members:
                idx = phrase_to_text_index(phrase_id, len(self._texts))
                if idx is not None:
                    results.append(self._texts[idx])
                    break
        return results

    def deep_search(self, query: str, k: int = 5) -> dict[str, Any]:
        """Recursive octave-descent retrieval over the cone hierarchy (RLM-style)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return {"texts": [], "evidence": [], "trace": []}
        r = self._settings.recursive
        engine = RecursiveAnswerEngine(
            self._pipeline, max_depth=r.max_depth, max_breadth=r.max_breadth,
            beam_k=r.beam_k, min_aperture_stop=r.min_aperture_stop,
        )
        result: RecursiveResult = engine.answer(query)
        return {
            "texts": list(result.evidence_texts)[:k],
            "evidence": [str(n) for n in result.evidence_node_ids][:k],
            "trace": [list(t) for t in result.trace],
        }

    def build_context_pack(self, query: str, max_tokens: int, counter=None) -> ContextPack:
        """Budget-bounded, redundancy-free context pack mitigating context rot."""
        if not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an int")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self._pipeline is None or not query:
            return ContextPack()
        cfg = ContextPackConfig(
            max_tokens=max_tokens,
            overlap_threshold=self._settings.context.overlap_threshold,
            distance_summary_threshold=self._settings.context.distance_summary_threshold,
            max_members_per_node=self._settings.context.max_members_per_node,
            reserve_tokens=self._settings.context.reserve_tokens,
            max_dedup_candidates=self._settings.context.max_dedup_candidates,
        )
        return ContextPackBuilder(self._pipeline, self._texts, cfg, counter).build(query, max_tokens)

    # --- layered memory ------------------------------------------------------
    def remember(self, fact: str, fact_id: str | None = None) -> str:
        """Pin an explicit long-term fact, always surfaced by recall()."""
        if not fact:
            raise ValueError("fact must be non-empty")
        return self._memory.remember(fact, fact_id)

    def forget(self, fact_id: str) -> bool:
        """Remove a pinned long-term fact by id."""
        return self._memory.forget(fact_id)

    def recall(self, query: str, budget_tokens: int | None = None) -> str:
        """Assemble the layered memory block (facts, summaries, working, session) under budget."""
        if not query:
            query = ""
        return self._memory.assemble_context(query, budget_tokens)

    # --- meaning flow and tension --------------------------------------------
    def navigate(self, focus_query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return neighbor cones ranked by entailment gradient with up/down direction labels."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not focus_query:
            return []
        focus = self._resolve_top(focus_query)
        if focus is None:
            return []
        engine = self._pipeline.engine
        nodes = [n for n in self._pipeline.store.all_nodes()
                 if n.prefix == focus.prefix and n.members]
        out: list[dict[str, Any]] = []
        for nid, weight, direction in engine.flow_neighbors(focus, nodes, k):
            out.append({
                "text": self._primary_text(self._pipeline.store.get(nid)),
                "gradient": weight,
                "direction": direction,
            })
        return out

    def scan_tension(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Surface the worst redundancy/contradiction pairs as human-readable text pairs."""
        if self._pipeline is None:
            return []
        engine = self._pipeline.engine
        nodes = self._pipeline.store.all_nodes()
        store = self._pipeline.store
        out: list[dict[str, Any]] = []
        for a_id, b_id, tension, kind in engine.tension_scan(nodes, top_n=top_n):
            out.append({
                "text_a": self._primary_text(store.get(a_id)),
                "text_b": self._primary_text(store.get(b_id)),
                "tension": tension,
                "kind": kind,
            })
        return out

    def compress_context(self, query: str, k: int) -> dict[str, Any]:
        """Pick k energy-minimizing representatives over a candidate pool; report energy reduction."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return {"texts": [], "energy_reduction": 0.0}
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        ids = self._pipeline.query.knn(q_vec[:prefix], k=max(k * 4, k), prefix=prefix)
        pool = [n for n in (self._pipeline.store.get(i) for i in ids) if n.members]
        engine = self._pipeline.engine
        reps, coverage = engine.select_representatives(pool, k)
        base = engine.select_representatives(pool, 1)[1]
        return {
            "texts": [self._primary_text(n) for n in reps],
            "energy_reduction": float(max(0.0, base - coverage)),
        }

    # --- hierarchy / containment (unchanged surface) -------------------------
    def explain_hierarchy(self, query: str) -> dict[str, Any]:
        """Return cone hierarchy info for the top node matching query."""
        if self._pipeline is None or not query:
            return {}
        node = self._resolve_top(query)
        if node is None:
            return {}
        return {
            "node_id": str(node.id),
            "aperture": node.aperture,
            "members": [str(m) for m in node.members],
            "label": node.label,
            "digest": node.digest,
        }

    def containment(self, parent_query: str, child_query: str) -> float:
        """Soft containment score >0 means parent entails child in hyperbolic space."""
        if self._pipeline is None:
            return 0.0
        p, c = self._resolve_top(parent_query), self._resolve_top(child_query)
        if p is None or c is None:
            return 0.0
        return self._pipeline.query.containment_score(p.id, c.id)
