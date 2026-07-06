"""KnowledgeBase mixin: search, ranking, reflection, and reranking."""
from __future__ import annotations

import math
from typing import Any

from .interfaces import Prefix, phrase_to_text_index
from .kb_types import (
    QueryPriority, SearchHit, DeepSearchResult, RetrievalStep,
    SparseSearchResult, QueryDecomposition, CategoricalParentHit, Directive,
)
from .recursive import RecursiveAnswerEngine, RecursiveResult


class RetrievalMixin:
    """Ranking, search variants, and agentic reflect/decompose loops over the pipeline."""

    def _ranked(self, query: str, k: int, priority: "QueryPriority | None" = None) -> list[tuple[Any, float, int]]:
        """Candidate nodes with blended score; returns (node, score, octave_hit_count)."""
        enc = self._pipeline._encoder
        store = self._pipeline.store
        a = self._settings.agent
        q_vec = enc.encode([query])[0]
        scored: dict[str, float] = {}
        octave_hits: dict[str, int] = {}
        p = priority or QueryPriority.MEDIUM
        if a.octave_fusion and p != QueryPriority.LOW:
            dims = enc.dims if p == QueryPriority.HIGH else enc.dims[:2]
            for prefix in (Prefix(d) for d in dims):
                for rank, (nid, _s) in enumerate(store.knn_scored(q_vec[:prefix], k * 4, prefix)):
                    scored[nid] = scored.get(nid, 0.0) + 1.0 / (60 + rank)
                    octave_hits[nid] = octave_hits.get(nid, 0) + 1
        else:
            prefix = Prefix(enc.dims[0])
            for nid, s in store.knn_scored(q_vec[:prefix], k * 4, prefix):
                scored[nid] = s
                octave_hits[nid] = 1
        if a.hybrid_lexical:
            self._fuse_lexical_rrf(query, scored, octave_hits, prefix=Prefix(enc.dims[0]))
        cands = []
        for nid, base in scored.items():
            node = store.get(nid)
            if not node.members:
                continue
            cands.append((node, base + a.usage_weight * math.log1p(self._node_uses(node)), octave_hits.get(nid, 1)))
        cands.sort(key=lambda t: t[1], reverse=True)
        lam = (0.3 if p == QueryPriority.HIGH else (0.7 if p == QueryPriority.LOW else a.mmr_lambda))
        return self._mmr(cands, k, lam)

    def _fuse_lexical_rrf(self, query: str, scored: dict[str, float], octave_hits: dict[str, int], prefix: Prefix) -> None:
        """Fuse BM25 lexical rank into the vector-side RRF accumulator (SeekStorm-style hybrid fusion)."""
        bm25_hits = self._bm25.score(query)[:40]
        if not bm25_hits:
            return
        store = self._pipeline.store
        members_to_nodes = store.members_to_nodes(prefix)
        for rank, (doc_id, _s) in enumerate(bm25_hits):
            idx = int(doc_id) if doc_id.isdigit() else -1
            if idx < 0 or idx >= len(self._texts):
                continue
            phrase_id = next((pid for pid in members_to_nodes if phrase_to_text_index(pid, len(self._texts)) == idx), None)
            if phrase_id is None:
                continue
            nid = members_to_nodes[phrase_id]
            scored[nid] = scored.get(nid, 0.0) + 1.0 / (60 + rank)
            octave_hits[nid] = octave_hits.get(nid, 0) + 1

    def _mmr(self, cands: list[tuple[Any, float, int]], k: int, lam: float) -> list[tuple[Any, float, int]]:
        engine = self._pipeline.engine
        selected: list[tuple[Any, float, int]] = []
        seen_text: set[str] = set()
        pool = list(cands)
        while pool and len(selected) < k:
            best, best_mmr, best_i = None, -1e18, -1
            for i, (node, rel, epc) in enumerate(pool):
                pt = self._primary_text(node)
                if pt in seen_text:
                    best_i = i if best_i < 0 else best_i
                    continue
                # centroid_overlap (embedding cosine), not overlap_score (untrained cone
                # apex angle -- seed noise for sibling pairs; see cone_engine.py comment).
                div = max((engine.centroid_overlap(s, node) for s, _, _e in selected), default=0.0)
                mmr = lam * rel - (1.0 - lam) * div
                if mmr > best_mmr:
                    best, best_mmr, best_i = (node, rel, epc), mmr, i
            if best is None:
                break
            pool.pop(best_i)
            seen_text.add(self._primary_text(best[0]))
            selected.append(best)
        return selected

    def search(self, query: str, k: int = 5, priority: QueryPriority = QueryPriority.MEDIUM) -> list[SearchHit]:
        """Return up to k diversified, scored hits; raises ValueError on invalid k or empty/whitespace query, may return fewer than k hits if the corpus lacks that many distinct candidates."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        with self._lock:
            if self._pipeline is None:
                return []
            self._metrics["queries"] += 1
            engine = self._pipeline.engine
            enc = self._pipeline._encoder
            hits: list[SearchHit] = []
            ranked = self._ranked(query, k, priority)
        scores = [float(score) for _, score, _ in ranked]
        for i, (node, score, epc) in enumerate(ranked):
            members = self._member_texts(node)
            ap = float(getattr(node, "aperture", 0.0))
            try:
                import numpy as np
                if len(members) >= 2:
                    vecs = np.asarray(enc.encode(members), dtype=np.float64)
                    entropy = engine._member_entropy(vecs)
                else:
                    entropy = 0.0
            except Exception:
                entropy = 0.0
            # uncertainty: relative gap to the next-best hit, not self-normalized against
            # the top hit (that construction made the top hit's uncertainty always 0.0).
            nxt = scores[i + 1] if i + 1 < len(scores) else None
            if nxt is None or score <= 1e-9:
                uncertainty = 0.0 if score > 1e-9 else 1.0
            else:
                uncertainty = float(max(0.0, min(1.0, nxt / score)))
            hits.append(SearchHit(
                text=members[0] if members else self._primary_text(node),
                score=float(score), node_id=str(node.id), octave=int(node.prefix),
                members=tuple(members),
                aperture=ap,
                local_entropy=float(entropy),
                evidence_path_count=int(epc),
                uncertainty_score=uncertainty,
            ))
        return hits

    def search_texts(self, query: str, k: int = 5) -> list[str]:
        """Back-compat: just the hit texts as a flat list."""
        return [h.text for h in self.search(query, k)]

    def batch_search(self, queries: list[str], k: int = 5) -> list[list[SearchHit]]:
        """Encode all queries in one model call; per-query diversified hits."""
        if k <= 0:
            raise ValueError("k must be positive")
        return [self.search(q, k) for q in queries]

    def deep_search(self, query: str, k: int = 5) -> DeepSearchResult:
        """Recursive octave-descent retrieval over the cone hierarchy (RLM-style)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return DeepSearchResult((), (), ())
        r = self._settings.recursive
        engine = RecursiveAnswerEngine(
            self._pipeline, max_depth=r.max_depth, max_breadth=r.max_breadth,
            beam_k=r.beam_k, min_aperture_stop=r.min_aperture_stop,
        )
        result: RecursiveResult = engine.answer(query)
        return DeepSearchResult(
            texts=tuple(result.evidence_texts[:k]),
            evidence=tuple(str(n) for n in result.evidence_node_ids[:k]),
            trace=tuple(tuple(t) for t in result.trace),
        )

    def explain_retrieval(self, query: str, k: int = 5) -> list[RetrievalStep]:
        """Per-hit trace of WHY it surfaced: score, containment to the top node, tension."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return []
        ranked = self._ranked(query, k)
        if not ranked:
            return []
        engine = self._pipeline.engine
        top = ranked[0][0]
        steps: list[RetrievalStep] = []
        for node, score, _epc in ranked:
            steps.append(RetrievalStep(
                text=self._primary_text(node), score=float(score), node_id=str(node.id),
                octave=int(node.prefix),
                containment_to_top=float(engine.contains(top, node)),
                tension=float(engine.tension(top, node)) if node.id != top.id else 0.0,
            ))
        return steps

    def sparse_search(self, query: str, k: int = 5, sparsity: float = 0.9) -> list:
        """NLA sparse search: prune low-activation candidates, return top-k SparseSearchResult."""
        import numpy as _np
        fetch_k = max(k, int(k / (1.0 - sparsity + 1e-9)))
        hits = self.search(query, k=fetch_k)
        if not hits:
            return []
        scores = _np.array([h.score for h in hits])
        threshold = float(_np.percentile(scores, sparsity * 100)) if len(scores) > 1 else 0.0
        results = []
        for h in hits:
            sparse_score = h.score if h.score >= threshold else 0.0
            results.append(SparseSearchResult(h, sparse_score))
        results.sort(key=lambda r: r.sparse_score, reverse=True)
        return results[:k]

    def optimal_octave(self, query: str, entropy_budget: float = 1.5, k: int = 5) -> int:
        """Smallest octave whose knn-scored retrieval entropy stays within entropy_budget."""
        import math as _math
        import numpy as _np
        if self._pipeline is None:
            return 64
        enc = self._pipeline._encoder
        store = self._pipeline.store
        q_vec = enc.encode([query])[0]
        octaves = [int(d) for d in enc.dims]
        best_octave = octaves[0] if octaves else 64
        best_energy = float("inf")
        for idx, octave in enumerate(octaves):
            prefix = Prefix(octave)
            scored = store.knn_scored(q_vec[:prefix], k, prefix)
            if not scored:
                continue
            scores = _np.array([s for _, s in scored], dtype=float)
            scores = scores / (scores.sum() + 1e-9)
            entropy = -float(_np.sum(scores * _np.log(scores + 1e-9))) / (_math.log(len(scores) + 1) + 1e-9)
            energy = idx * 0.2
            if entropy <= entropy_budget and energy < best_energy:
                best_energy = energy
                best_octave = octave
        return best_octave

    def decompose_query(self, query: str, reflect_fn=None) -> "QueryDecomposition":
        """Split compound query; assign each sub-query to best octave."""
        import re as _re
        if reflect_fn is not None:
            parts = reflect_fn(query)
            if not isinstance(parts, list):
                parts = [query]
        else:
            parts = _re.split(r'\s+(?:and|but|versus|vs|or)\s+', query, flags=_re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [query]
        compound_score = 0.0 if len(parts) <= 1 else min(1.0, len(parts) / 4.0)
        octave_assignments = []
        for p in parts:
            try:
                octave_assignments.append(self.best_octave(p, query))
            except Exception:
                octave_assignments.append(256)
        return QueryDecomposition(tuple(parts), tuple(octave_assignments), compound_score)

    def search_with_reflection(self, query: str, k: int = 5, reflect_fn=None) -> dict:
        """Agentic inference: on low confidence, optionally reflect and retry with refined query."""
        initial = self.search(query, k)
        if not initial:
            return {"original": [], "reflected": [], "query_used": query, "reflected_query": None}
        top = initial[0]
        if top.uncertainty_score <= 0.5:
            return {"original": initial, "reflected": [], "query_used": query, "reflected_query": None}
        if reflect_fn is not None:
            refined = reflect_fn(query)
        else:
            refined = query
        reflected = self.search(refined, k) if refined != query else []
        return {
            "original": initial,
            "reflected": reflected,
            "query_used": query,
            "reflected_query": refined,
        }

    def reflect_directive(self, query: str) -> "Directive":
        """Instruction-emitting reflection: return the Directive for the agent, no llm_fn call."""
        from .research_loop import ResearchLoop
        loop = ResearchLoop(self, self._settings)
        hits = self.search(query, k=5)
        text = loop.instructions["observe"]
        return Directive(stage="observe", instruction_text=text, target_query=query,
                         context=tuple(h.text for h in hits[:3]),
                         expected="supported texts and a support_signal in [0,1]")

    def agentic_reflect(self, query: str, llm_fn=None, max_rounds: int = 3,
                        reflect_strategy: str = "rephrase") -> list:
        """Iterative reflection loop: search, observe, refine query up to max_rounds.

        reflect_strategy: 'rephrase' (tail of observation), 'decompose' (sub-query split),
        'expand' (append top hit text).  Stops early if uncertainty_score < 0.2.
        """
        from .kb_types import ReflectStep
        summarizer = self._summarizer
        steps: list[ReflectStep] = []
        current_query = query
        prev_hit_ids: list = []
        for r in range(max_rounds):
            hits = self.search(current_query, k=5)
            hit_ids = [h.node_id for h in hits]
            if llm_fn is not None:
                context = {"query": current_query, "hits": [h.text for h in hits],
                           "strategy": reflect_strategy, "round": r}
                observation = llm_fn(**context)
            else:
                observation = summarizer.summarize(current_query, [h.text for h in hits])
            step = ReflectStep(round=r, query=current_query, hits=hits, observation=observation)
            steps.append(step)
            # stop if top hit is confident
            if hits and hits[0].uncertainty_score < 0.2:
                break
            if hit_ids == prev_hit_ids:
                break
            prev_hit_ids = hit_ids
            obs_words = str(observation).split()
            if reflect_strategy == "decompose":
                parts = str(observation).split(",")
                current_query = parts[0].strip() if parts else current_query
            elif reflect_strategy == "expand":
                top_text = hits[0].text if hits else ""
                current_query = f"{current_query} {top_text}"[:200]
            else:  # rephrase: tail of observation
                current_query = " ".join(obs_words[-3:]) if len(obs_words) >= 3 else (obs_words[-1] if obs_words else current_query)
        return steps

    def categorical_parent_score(self, query: str, k: int = 5) -> list:
        """Score parent nodes by cosine similarity of their summarized label to query."""
        import numpy as _np
        if self._pipeline is None or not query:
            return []
        store_nodes = self._pipeline.store.nodes_by_id()
        enc = self._pipeline._encoder
        q_vec = _np.array(enc.encode([query])[0], dtype=float)
        qn = _np.linalg.norm(q_vec)
        q_unit = q_vec / (qn + 1e-9)
        parent_nodes = [(nid, node) for nid, node in store_nodes.items() if node.members]
        if not parent_nodes:
            return []
        summaries = [self._summarizer.summarize(str(nid), self._member_texts(node)) for nid, node in parent_nodes]
        sv_batch = _np.array(enc.encode(summaries), dtype=float)
        results = []
        for (nid, _node), summary, sv in zip(parent_nodes, summaries, sv_batch):
            sn = _np.linalg.norm(sv)
            sv_unit = sv / (sn + 1e-9)
            sim = float(_np.dot(q_unit, sv_unit))
            results.append(CategoricalParentHit(node_id=str(nid), summary=summary, embedding_sim=sim))
        results.sort(key=lambda h: h.embedding_sim, reverse=True)
        return results[:k]

    def hybrid_score(self, query: str, texts: list, llm_fn=None) -> list:
        """Rerank texts by w*SBERT_cosine + (1-w)*LLM_score (w in AgentSettings); returns (text, score) pairs."""
        import numpy as _np
        if self._pipeline is None or not texts:
            return [(t, 0.0) for t in texts]
        enc = self._pipeline._encoder
        w = self._settings.agent.hybrid_score_cosine_weight
        q_vec = _np.array(enc.encode([query])[0], dtype=float)
        qn = _np.linalg.norm(q_vec) + 1e-9
        q_unit = q_vec / qn
        text_vecs = _np.array(enc.encode(texts), dtype=float)
        results = []
        for t, tv in zip(texts, text_vecs):
            tn = _np.linalg.norm(tv) + 1e-9
            cos_score = float(_np.dot(q_unit, tv / tn))
            llm_score = 0.5  # neutral default
            if llm_fn is not None:
                try:
                    raw = llm_fn(query=query, text=t)
                    llm_score = float(raw) if raw is not None else 0.5
                except Exception:
                    pass
            combined = w * cos_score + (1.0 - w) * llm_score
            results.append((t, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
