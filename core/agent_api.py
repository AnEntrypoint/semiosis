"""High-level KnowledgeBase API for agents -- hides cone/manifold internals."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .context_pack import ContextPack, ContextPackBuilder, ContextPackConfig
from .interfaces import Prefix, phrase_to_text_index
from .pipeline import KnowledgePipeline
from .recursive import RecursiveAnswerEngine, RecursiveResult
from .semiotic_memory import SemioticMemory
from .settings import Settings


class QueryPriority(Enum):
    HIGH = "high"    # deep multi-octave, tight MMR, higher latency budget
    MEDIUM = "medium"  # default balanced
    LOW = "low"      # coarse single-octave, fast


class FailureMode(Enum):
    NONE = "none"
    OUTSIDE_CONE = "outside_cone"        # query lies outside all node cones
    BOUNDARY_AMBIGUOUS = "boundary_ambiguous"  # high tension between top candidates
    OVER_COMPRESSED = "over_compressed"  # excessive merges at last consolidation
    OCTAVE_MISMATCH = "octave_mismatch"  # high entropy_divergence across octaves


@dataclass(frozen=True, slots=True)
class SearchHit:
    text: str
    score: float
    node_id: str
    octave: int
    members: tuple[str, ...] = ()
    aperture: float = 0.0           # cone half-aperture; low=tight/confident, high=broad
    local_entropy: float = 0.0      # Shannon entropy of member distances from centroid
    evidence_path_count: int = 1    # octaves this node was retrieved from (consensus)
    uncertainty_score: float = 0.0  # 1 - normalized score; higher = less confident


@dataclass(frozen=True, slots=True)
class ConsolidateReport:
    changed: bool
    nodes_before: int
    nodes_after: int
    merges: int
    aperture_updates: int
    dispel_count: int


@dataclass(frozen=True, slots=True)
class FlowNeighbor:
    text: str
    gradient: float
    direction: str


@dataclass(frozen=True, slots=True)
class TensionPair:
    text_a: str
    text_b: str
    tension: float
    kind: str


@dataclass(frozen=True, slots=True)
class DeepSearchResult:
    texts: tuple[str, ...]
    evidence: tuple[str, ...]
    trace: tuple[tuple, ...]


@dataclass(frozen=True, slots=True)
class CompressResult:
    texts: tuple[str, ...]
    energy_reduction: float


@dataclass(frozen=True, slots=True)
class DiagnoseReport:
    nodes: int
    octaves: int
    texts: int
    facts: int
    mean_aperture: float
    mean_tension: float
    total_energy: float
    redundant_pairs: int
    retrieval_entropy: float = 0.0
    entropy_divergence: float = 0.0
    failure_mode: FailureMode = FailureMode.NONE
    recovery_suggestions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    text: str
    score: float
    node_id: str
    octave: int
    containment_to_top: float
    tension: float


@dataclass(frozen=True, slots=True)
class SemanticDirection:
    from_node: str
    to_node: str
    octave: int
    direction_vec: tuple[float, ...]
    magnitude: float
    cosine_alignment: float


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    node_id: str
    octave: int
    distance_from_prev: float
    direction_vec: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SemanticTrajectory:
    steps: tuple[TrajectoryStep, ...]
    total_distance: float
    coherence_score: float
    energy_cost: float


@dataclass(frozen=True, slots=True)
class DirectionSearchResult:
    hits: tuple["SearchHit", ...]
    alpha: float
    alignment: float


@dataclass(frozen=True, slots=True)
class CompressedHierarchy:
    """Info-bottleneck pruned node set."""
    retained_nodes: tuple
    dropped_nodes: tuple
    info_retained_ratio: float


@dataclass(frozen=True, slots=True)
class RecursiveAnswerResult:
    """Result of recursive LLM-driven octave descent."""
    answer_nodes: tuple
    depth_reached: int
    energy_total: float
    sub_queries: tuple


@dataclass(frozen=True, slots=True)
class ManifoldComplexity:
    """Intrinsic dimensionality of query neighborhood."""
    intrinsic_dim: float
    suggested_octave: int
    complexity_label: str


@dataclass(frozen=True, slots=True)
class FoldBudgetResult:
    """Energy-aware greedy candidate selection under token budget."""
    included: tuple
    excluded: tuple
    tokens_used: int
    energy_cost: float


@dataclass(frozen=True, slots=True)
class SparseSearchResult:
    """SearchHit with sparse mask weight."""
    hit: SearchHit
    sparse_score: float


@dataclass(frozen=True, slots=True)
class IngestStreamResult:
    """Result of incremental streaming ingest."""
    ingested_count: int
    new_nodes: int
    rebalanced: bool
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ContrastiveDirection:
    """Direction vector representing what separates two concepts."""
    direction_vec: tuple
    contrast_score: float
    octave: int


@dataclass(frozen=True, slots=True)
class QueryDecomposition:
    """Compound query split into sub-queries with octave assignments."""
    sub_queries: tuple
    octave_assignments: tuple
    compound_score: float


@dataclass(frozen=True, slots=True)
class AttentionScore:
    """NLA-style attention weight for a node given a query."""
    node_id: str
    weight: float
    octave: int
    temperature: float


@dataclass(frozen=True, slots=True)
class AnalogyResult:
    """word2vec-style A:B::C:X analogy result."""
    hits: tuple
    direction_used: tuple
    analogy_score: float


@dataclass(frozen=True, slots=True)
class ConceptBoundary:
    """Hyperplane separating two concept clusters."""
    midpoint: tuple
    normal_vec: tuple
    margin: float
    octave: int


@dataclass(frozen=True, slots=True)
class DispelReport:
    """Result of entropy-triggered dispel."""
    dispelled_ids: tuple
    entropy_before: float
    entropy_after: float


class SemanticDirectionError(ValueError):
    pass


try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]


class StubSummarizer:
    """Summarizer that combines first two member texts; no LLM required; used by default."""

    def summarize(self, node_id: str, member_texts: list[str]) -> str:
        if not member_texts:
            return f"category:{node_id}"
        preview = "; ".join(member_texts[:2])
        return f"Category covering: {preview}"


class KnowledgeBase:
    """Ingest texts, search, navigate meaning flow, learn from outcomes, persist across sessions."""

    def __init__(self, settings: Settings | None = None, summarizer=None) -> None:
        self._settings = settings or Settings()
        self._pipeline: KnowledgePipeline | None = None
        self._texts: list[str] = []
        self._memory = SemioticMemory(None, [], self._settings)
        self._usage: dict[str, int] = {}
        self._metrics: dict[str, int] = {
            "queries": 0, "ingests": 0, "record_outcomes": 0, "consolidations": 0,
        }
        self._summarizer = summarizer or StubSummarizer()

    def ingest(self, texts: list[str]) -> None:
        """Add texts; reuses cached embeddings incrementally when an agent grows the KB in a loop."""
        if not texts:
            return
        self._texts.extend(texts)
        self._metrics["ingests"] += 1
        if self._pipeline is not None and self._settings.agent.incremental_ingest:
            self._pipeline.ingest(texts)
        else:
            self._pipeline = KnowledgePipeline(self._texts, self._settings)
        self._memory._pipeline = self._pipeline
        self._memory._texts = list(self._texts)

    # --- helpers -------------------------------------------------------------
    def _member_texts(self, node) -> list[str]:
        out: list[str] = []
        for m in node.members:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None and self._texts[idx] not in out:
                out.append(self._texts[idx])
        return out

    def _primary_text(self, node) -> str:
        mt = self._member_texts(node)
        return mt[0] if mt else (node.digest or node.label or "")

    def _node_uses(self, node) -> int:
        return sum(self._usage.get(t, 0) for t in self._member_texts(node))

    def _resolve_top(self, query: str):
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        ids = self._pipeline.query.knn(q_vec[:prefix], k=1, prefix=prefix)
        return self._pipeline.store.get(ids[0]) if ids else None

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
        cands = []
        for nid, base in scored.items():
            node = store.get(nid)
            if not node.members:
                continue
            cands.append((node, base + a.usage_weight * math.log1p(self._node_uses(node)), octave_hits.get(nid, 1)))
        cands.sort(key=lambda t: t[1], reverse=True)
        lam = (0.3 if p == QueryPriority.HIGH else (0.7 if p == QueryPriority.LOW else a.mmr_lambda))
        return self._mmr(cands, k, lam)

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
                div = max((engine.overlap_score(s, node) for s, _, _e in selected), default=0.0)
                mmr = lam * rel - (1.0 - lam) * div
                if mmr > best_mmr:
                    best, best_mmr, best_i = (node, rel, epc), mmr, i
            if best is None:
                break
            pool.pop(best_i)
            seen_text.add(self._primary_text(best[0]))
            selected.append(best)
        return selected

    # --- retrieval -----------------------------------------------------------
    def search(self, query: str, k: int = 5, priority: QueryPriority = QueryPriority.MEDIUM) -> list[SearchHit]:
        """Return up to k diversified, scored hits with confidence signals for agent self-regulation."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return []
        self._metrics["queries"] += 1
        engine = self._pipeline.engine
        hits: list[SearchHit] = []
        top_score: float | None = None
        for node, score, epc in self._ranked(query, k, priority):
            if top_score is None:
                top_score = score
            members = self._member_texts(node)
            ap = float(getattr(node, "aperture", 0.0))
            try:
                import numpy as np
                store = self._pipeline.store
                enc = self._pipeline._encoder
                q_vec = enc.encode([query])[0]
                vecs = np.stack([q_vec for _ in node.members]) if node.members else np.zeros((1, len(q_vec)))
                entropy = engine._member_entropy(vecs)
            except Exception:
                entropy = 0.0
            norm_score = float(score) / max(top_score, 1e-9) if top_score else 0.0
            hits.append(SearchHit(
                text=members[0] if members else self._primary_text(node),
                score=float(score), node_id=str(node.id), octave=int(node.prefix),
                members=tuple(members),
                aperture=ap,
                local_entropy=float(entropy),
                evidence_path_count=int(epc),
                uncertainty_score=float(1.0 - norm_score),
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

    def build_context_pack(self, query: str, max_tokens: int, counter=None) -> ContextPack:
        """Budget-bounded, redundancy-free context pack mitigating context rot."""
        if not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an int")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self._pipeline is None or not query:
            return ContextPack()
        c = self._settings.context
        cfg = ContextPackConfig(
            max_tokens=max_tokens, overlap_threshold=c.overlap_threshold,
            distance_summary_threshold=c.distance_summary_threshold,
            max_members_per_node=c.max_members_per_node, reserve_tokens=c.reserve_tokens,
            max_dedup_candidates=c.max_dedup_candidates,
        )
        return ContextPackBuilder(self._pipeline, self._texts, cfg, counter).build(query, max_tokens)

    # --- learning loop -------------------------------------------------------
    def record_outcome(self, query: str, useful_texts: list[str],
                       useless_texts: list[str] | None = None) -> dict[str, int]:
        """Feed back which retrieved texts proved useful; usage counts steer future ranking."""
        self._metrics["record_outcomes"] += 1
        known = set(self._texts)
        applied = 0
        for t in useful_texts:
            if t in known:
                self._usage[t] = self._usage.get(t, 0) + 1
                applied += 1
        for t in (useless_texts or []):
            if t in known and self._usage.get(t, 0) > 0:
                self._usage[t] -= 1
        return {"applied": applied, "ignored": len(useful_texts) - applied}

    def consolidate(self) -> ConsolidateReport:
        """Self-improve the KB: scan tension, merge redundant pairs; idempotent."""
        if self._pipeline is None:
            return ConsolidateReport(changed=False, nodes_before=0, nodes_after=0, merges=0, aperture_updates=0, dispel_count=0)
        self._metrics["consolidations"] += 1
        engine = self._pipeline.engine
        store = self._pipeline.store
        nodes = store.all_nodes()
        nodes_before = len(nodes)
        scan = engine.tension_scan(nodes, top_n=len(nodes))
        thr = self._settings.agent.consolidate_tension
        plan = [row for row in engine.dispel_plan(scan)
                if any(s[0] == row[1] and s[1] == row[2] and s[2] >= thr for s in scan)]
        merges = 0
        aperture_updates = 0
        for op, a_id, b_id in plan:
            if op == "merge":
                try:
                    merged = engine.merge_nodes(store.get(a_id), store.get(b_id))
                    store.upsert(merged)
                    merges += 1
                except KeyError:
                    continue
        nodes_after = len(store.all_nodes())
        return ConsolidateReport(
            changed=merges > 0,
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            merges=merges,
            aperture_updates=aperture_updates,
            dispel_count=len(plan),
        )

    def diagnose(self) -> DiagnoseReport:
        """Health snapshot an agent reads to decide when to consolidate or diversify ingest."""
        if self._pipeline is None:
            return DiagnoseReport(0, 0, len(self._texts), len(self._memory.facts()),
                                  0.0, 0.0, 0.0, 0)
        engine = self._pipeline.engine
        nodes = [n for n in self._pipeline.store.all_nodes() if n.members]
        octaves = len({n.prefix for n in nodes})
        mean_ap = sum(n.aperture for n in nodes) / len(nodes) if nodes else 0.0
        scan = engine.tension_scan(nodes, top_n=len(nodes))
        mean_t = sum(t for _, _, t, _ in scan) / len(scan) if scan else 0.0
        redundant = sum(1 for _, _, _, kind in scan if kind in ("redundancy", "contradiction"))
        energy = engine.context_energy(nodes[:32])
        # entropy divergence: variance of per-octave aperture means
        octave_aps: dict[int, list[float]] = {}
        for n in nodes:
            octave_aps.setdefault(int(n.prefix), []).append(n.aperture)
        oct_means = [sum(v) / len(v) for v in octave_aps.values()]
        global_mean = sum(oct_means) / len(oct_means) if oct_means else 0.0
        entropy_div = math.sqrt(sum((m - global_mean) ** 2 for m in oct_means) / max(len(oct_means), 1))
        # detect failure mode
        failure = FailureMode.NONE
        suggestions: list[str] = []
        if mean_ap > 1.2:
            failure = FailureMode.OUTSIDE_CONE
            suggestions.append("ingest more focused texts to tighten cone apertures")
        elif mean_t > 0.7 and redundant > len(nodes) // 4:
            failure = FailureMode.BOUNDARY_AMBIGUOUS
            suggestions.append("call consolidate() to resolve boundary tension between overlapping cones")
        elif octaves > 0 and len(nodes) / max(octaves, 1) < 2:
            failure = FailureMode.OVER_COMPRESSED
            suggestions.append("ingest more diverse texts across octaves to restore hierarchy depth")
        elif entropy_div > 0.4:
            failure = FailureMode.OCTAVE_MISMATCH
            suggestions.append("check for missing octave levels; use deep_search() for cross-octave queries")
        return DiagnoseReport(
            nodes=len(nodes), octaves=octaves, texts=len(self._texts),
            facts=len(self._memory.facts()), mean_aperture=float(mean_ap),
            mean_tension=float(mean_t), total_energy=float(energy), redundant_pairs=redundant,
            retrieval_entropy=float(mean_ap),
            entropy_divergence=float(entropy_div),
            failure_mode=failure,
            recovery_suggestions=tuple(suggestions),
        )

    def metrics(self) -> dict[str, int]:
        """Usage counters for agent monitoring."""
        m = dict(self._metrics)
        m.update({"nodes": len(self._pipeline.store.all_nodes()) if self._pipeline else 0,
                  "n_texts": len(self._texts), "n_facts": len(self._memory.facts())})
        return m

    def compress_hierarchy(self, query: str, max_nodes: int = 10) -> "CompressedHierarchy":
        """Retain highest-relevance nodes under info-bottleneck criterion."""
        import numpy as _np
        nodes = list(self._pipeline.store._nodes.values()) if hasattr(self._pipeline.store, '_nodes') else []
        if not nodes:
            return CompressedHierarchy((), (), 1.0)
        q_vec = self._pipeline._encoder.encode([query])[0]
        first_centroid = next((n.centroid for n in nodes if n.centroid), None)
        p = len(first_centroid) if first_centroid else min(256, len(q_vec))
        p = min(p, len(q_vec))
        q_slice = _np.array(q_vec[:p], dtype=float)
        q_slice = q_slice / (_np.linalg.norm(q_slice) + 1e-9)
        scored = []
        for node in nodes:
            c = _np.array(list(node.centroid)[:p], dtype=float) if node.centroid else _np.zeros(p)
            cn = _np.linalg.norm(c)
            c = c / (cn + 1e-9)
            scored.append((float(_np.dot(q_slice, c)), node.id))
        scored.sort(reverse=True)
        retained = tuple(nid for _, nid in scored[:max_nodes])
        dropped = tuple(nid for _, nid in scored[max_nodes:])
        ratio = len(retained) / (len(retained) + len(dropped)) if (retained or dropped) else 1.0
        return CompressedHierarchy(retained, dropped, ratio)

    def sense_complexity(self, query: str, k: int = 10) -> "ManifoldComplexity":
        """Estimate intrinsic dim of query neighborhood via TwoNN."""
        from .manifold_ops import twonn_intrinsic_dim
        hits = self.search(query, k=k)
        if len(hits) < 3:
            return ManifoldComplexity(1.0, 64, "constant")
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        vecs = []
        for h in hits[:k]:
            if h.node_id in store_nodes:
                c = list(store_nodes[h.node_id].centroid or [])
                if c:
                    vecs.append(c)
        if len(vecs) < 3:
            return ManifoldComplexity(1.0, 64, "constant")
        min_len = min(len(v) for v in vecs)
        vecs = [v[:min_len] for v in vecs]
        dim = twonn_intrinsic_dim(vecs)
        if dim < 1.5:
            label, octave = "constant", 64
        elif dim < 2.5:
            label, octave = "linear", 128
        elif dim < 4.0:
            label, octave = "quadratic", 256
        else:
            label, octave = "exponential", 512
        return ManifoldComplexity(dim, octave, label)

    def fold_budget(self, query: str, max_tokens: int, candidates: list) -> "FoldBudgetResult":
        """Greedy token-budget fold: select highest-relevance candidates under token limit."""
        import numpy as _np
        q_vec = _np.array(self._pipeline._encoder.encode([query])[0], dtype=float)
        q_vec = q_vec / (_np.linalg.norm(q_vec) + 1e-9)
        scored = []
        for text in candidates:
            v = _np.array(self._pipeline._encoder.encode([text])[0], dtype=float)
            vn = _np.linalg.norm(v)
            v = v / (vn + 1e-9)
            scored.append((float(_np.dot(q_vec, v)), text))
        scored.sort(reverse=True)
        included, excluded = [], []
        tokens_used = 0
        energy = 0.0
        for score, text in scored:
            tok = max(1, len(text.split()) * 4 // 3)
            if tokens_used + tok <= max_tokens:
                included.append(text)
                tokens_used += tok
                energy += 1.0 - score
            else:
                excluded.append(text)
        return FoldBudgetResult(tuple(included), tuple(excluded), tokens_used, energy)

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

    def optimal_octave(self, query: str, entropy_budget: float = 1.5) -> int:
        """Find octave minimizing energy_cost subject to retrieval entropy <= entropy_budget."""
        import math as _math
        import numpy as _np
        octaves = [64, 128, 256, 512, 1024]
        best_octave = 64
        best_energy = float("inf")
        for idx, octave in enumerate(octaves):
            hits = self.search(query, k=5)
            if not hits:
                continue
            scores = _np.array([h.score for h in hits])
            scores = scores / (scores.sum() + 1e-9)
            entropy = -float(_np.sum(scores * _np.log(scores + 1e-9))) / (_math.log(len(scores) + 1) + 1e-9)
            energy = idx * 0.2
            if entropy <= entropy_budget and energy < best_energy:
                best_energy = energy
                best_octave = octave
        return best_octave

    def information_content(self, node_id: str) -> float:
        """IC = -log2(aperture/pi); high IC = specific concept."""
        import math as _math
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        node = store_nodes.get(node_id)
        if node is None:
            return 0.0
        aperture = getattr(node, 'aperture', 0.5) or 0.5
        return max(0.0, -_math.log2(max(aperture, 1e-9) / _math.pi))

    def ingest_stream(self, texts, threshold: float = 0.7, rebalance_threshold: int = 20) -> "IngestStreamResult":
        """Incremental ingest: add each text and optionally rebalance."""
        import time as _time
        t0 = _time.monotonic()
        before = len(self._pipeline.store._nodes) if hasattr(self._pipeline.store, '_nodes') else 0
        count = 0
        for text in texts:
            self.ingest([text])
            count += 1
        after = len(self._pipeline.store._nodes) if hasattr(self._pipeline.store, '_nodes') else 0
        new_nodes = max(0, after - before)
        rebalanced = False
        if new_nodes > rebalance_threshold:
            self._pipeline.build(list(self._pipeline._vec_cache.keys()) if hasattr(self._pipeline, '_vec_cache') else [])
            rebalanced = True
        elapsed = (_time.monotonic() - t0) * 1000.0
        return IngestStreamResult(count, new_nodes, rebalanced, elapsed)

    def contrastive_direction(self, text_a: str, text_b: str, octave: int | None = None) -> "ContrastiveDirection":
        """Direction = normalize(embed(a) - embed(b)); contrast_score = norm of difference."""
        import numpy as _np
        va = _np.array(self._pipeline._encoder.encode([text_a])[0], dtype=float)
        vb = _np.array(self._pipeline._encoder.encode([text_b])[0], dtype=float)
        if octave is not None:
            va, vb = va[:octave], vb[:octave]
        diff = va - vb
        score = float(_np.linalg.norm(diff))
        direction = diff / (score + 1e-9)
        return ContrastiveDirection(tuple(direction.tolist()), score, octave or len(va))

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

    def attention_score(self, node_id: str, query: str, temperature: float = 1.0) -> "AttentionScore":
        """NLA-style scaled dot-product attention weight over all nodes."""
        import numpy as _np
        import math as _math
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        if not store_nodes:
            return AttentionScore(node_id, 0.0, 64, temperature)
        q_vec = _np.array(self._pipeline._encoder.encode([query])[0], dtype=float)
        p = 64
        q_slice = q_vec[:p] / (_np.linalg.norm(q_vec[:p]) + 1e-9)
        raw_scores = {}
        for nid, node in store_nodes.items():
            c_list = list(node.centroid or [])[:p]
            if not c_list:
                continue
            c = _np.array(c_list, dtype=float)
            if len(c) < p:
                c = _np.pad(c, (0, p - len(c)))
            cn = _np.linalg.norm(c)
            c = c / (cn + 1e-9)
            raw_scores[nid] = float(_np.dot(q_slice, c)) / (_math.sqrt(p) * temperature)
        max_s = max(raw_scores.values(), default=0.0)
        exp_scores = {nid: _math.exp(s - max_s) for nid, s in raw_scores.items()}
        total = sum(exp_scores.values()) + 1e-9
        weight = exp_scores.get(node_id, 0.0) / total
        octave = 64
        return AttentionScore(node_id, weight, octave, temperature)

    def find_analogy(self, text_a: str, text_b: str, text_c: str, k: int = 5) -> "AnalogyResult":
        """word2vec analogy: embed(c) + (embed(b) - embed(a)) -> nearest nodes."""
        import numpy as _np
        enc = self._pipeline._encoder.encode
        va = _np.array(enc([text_a])[0], dtype=float)
        vb = _np.array(enc([text_b])[0], dtype=float)
        vc = _np.array(enc([text_c])[0], dtype=float)
        direction = vb - va
        hits = self.direction_search(text_c, direction.tolist(), k=k)
        analogy_score = float(hits[0].alignment) if hits else 0.0
        raw_hits = []
        for r in hits:
            raw_hits.extend(list(r.hits))
        return AnalogyResult(tuple(raw_hits[:k]), tuple(direction.tolist()), analogy_score)

    def concept_boundary(self, node_id_a: str, node_id_b: str, octave: int | None = None) -> "ConceptBoundary":
        """Compute decision boundary between two concept nodes."""
        import numpy as _np
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        node_a = store_nodes.get(node_id_a)
        node_b = store_nodes.get(node_id_b)
        p = octave or 256
        def get_centroid(node):
            if node is None or not node.centroid:
                return _np.zeros(p)
            c = _np.array(list(node.centroid)[:p], dtype=float)
            return c / (_np.linalg.norm(c) + 1e-9)
        ca = get_centroid(node_a)
        cb = get_centroid(node_b)
        midpoint = (ca + cb) / 2.0
        normal = cb - ca
        margin = float(_np.linalg.norm(normal)) / 2.0
        nn = _np.linalg.norm(normal)
        normal_unit = normal / (nn + 1e-9)
        return ConceptBoundary(tuple(midpoint.tolist()), tuple(normal_unit.tolist()), margin, p)

    def entropy_dispel(self, entropy_ceiling: float = 2.0) -> "DispelReport":
        """Auto-dispel nodes with entropy proxy exceeding ceiling."""
        import math as _math
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        if not store_nodes:
            return DispelReport((), 0.0, 0.0)
        def node_entropy(node):
            aperture = getattr(node, 'aperture', 0.5) or 0.5
            n = max(1, len(getattr(node, 'members', []) or []))
            return _math.log(1.0 + aperture * n)
        entropies = {nid: node_entropy(n) for nid, n in store_nodes.items()}
        before = sum(entropies.values()) / (len(entropies) + 1e-9)
        to_dispel = [nid for nid, e in entropies.items() if e > entropy_ceiling]
        if to_dispel:
            try:
                self._pipeline.engine.dispel_plan(to_dispel)
            except Exception:
                pass
        after_entropies = {nid: node_entropy(n) for nid, n in store_nodes.items() if nid not in to_dispel}
        after = sum(after_entropies.values()) / (len(after_entropies) + 1e-9) if after_entropies else 0.0
        return DispelReport(tuple(to_dispel), before, after)

    def build_digest_chain(self, summarizer=None) -> dict:
        """Bottom-up hierarchy digest chain: leaf->parent using summarizer."""
        store_nodes = self._pipeline.store._nodes if hasattr(self._pipeline.store, '_nodes') else {}
        if not store_nodes:
            return {}
        digests = {}
        for nid, node in store_nodes.items():
            members = list(getattr(node, 'members', []) or [])
            if members:
                digests[nid] = members[0][:100] if members[0] else nid
            else:
                digests[nid] = nid
        for nid, node in store_nodes.items():
            digest_attr = getattr(node, 'digest', None)
            if digest_attr:
                if summarizer is not None:
                    members = list(getattr(node, 'members', []) or [])
                    try:
                        digests[nid] = summarizer.summarize(nid, members)
                    except Exception:
                        digests[nid] = digest_attr
                else:
                    digests[nid] = digest_attr
        return digests

    def compute_transition_matrix(self, node_ids: list) -> dict:
        """Transition matrix P(octave_j | octave_i) from node octave labels."""
        import re as _re
        octaves = [64, 128, 256, 512, 1024]
        counts = {}
        for o1 in octaves:
            for o2 in octaves:
                counts[(o1, o2)] = 0.0
        for nid in node_ids:
            m = _re.search(r'@(\d+)', nid)
            if m:
                o = int(m.group(1))
                if o in octaves:
                    counts[(o, o)] += 1.0
        matrix = {}
        for o1 in octaves:
            row_sum = sum(counts[(o1, o2)] for o2 in octaves) + 1e-9
            for o2 in octaves:
                matrix[(o1, o2)] = counts[(o1, o2)] / row_sum
        return matrix

    # --- persistence ---------------------------------------------------------
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
    def load(cls, path: "str | os.PathLike[str]", settings: Settings | None = None) -> "KnowledgeBase":
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

    # --- meaning flow and tension --------------------------------------------
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
        return self._memory.assemble_context(query or "", budget_tokens)

    def navigate(self, focus_query: str, k: int = 5) -> list[FlowNeighbor]:
        """Neighbor cones ranked by entailment gradient with up/down direction labels."""
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
        out: list[FlowNeighbor] = []
        for nid, weight, direction in engine.flow_neighbors(focus, nodes, k):
            out.append(FlowNeighbor(self._primary_text(self._pipeline.store.get(nid)),
                                    float(weight), direction))
        return out

    def scan_tension(self, top_n: int = 10) -> list[TensionPair]:
        """Surface the worst redundancy/contradiction pairs as human-readable text pairs."""
        if self._pipeline is None:
            return []
        engine = self._pipeline.engine
        store = self._pipeline.store
        out: list[TensionPair] = []
        for a_id, b_id, tension, kind in engine.tension_scan(store.all_nodes(), top_n=top_n):
            out.append(TensionPair(self._primary_text(store.get(a_id)),
                                   self._primary_text(store.get(b_id)), float(tension), kind))
        return out

    def compress_context(self, query: str, k: int) -> CompressResult:
        """Pick k energy-minimizing representatives over a candidate pool; report energy reduction."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return CompressResult((), 0.0)
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        ids = self._pipeline.query.knn(q_vec[:prefix], k=max(k * 4, k), prefix=prefix)
        pool = [n for n in (self._pipeline.store.get(i) for i in ids) if n.members]
        engine = self._pipeline.engine
        reps, coverage = engine.select_representatives(pool, k)
        base = engine.select_representatives(pool, 1)[1]
        return CompressResult(tuple(self._primary_text(n) for n in reps),
                              float(max(0.0, base - coverage)))

    # --- hierarchy / containment ---------------------------------------------
    def explain_hierarchy(self, query: str) -> dict[str, Any]:
        """Return cone hierarchy info for the top node matching query."""
        if self._pipeline is None or not query:
            return {}
        node = self._resolve_top(query)
        if node is None:
            return {}
        return {
            "node_id": str(node.id), "aperture": node.aperture,
            "members": [str(m) for m in node.members], "label": node.label,
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

    def semantic_distance(self, text_a: str, text_b: str, octave: int | None = None,
                          use_hyperbolic: bool = False) -> "float | dict[int, float]":
        """Cosine or hyperbolic geodesic distance between two texts at one or all octave prefixes."""
        if self._pipeline is None:
            return 0.0 if octave is not None else {}
        enc = self._pipeline._encoder
        va = enc.encode([text_a])[0]
        vb = enc.encode([text_b])[0]
        import numpy as np
        prefixes = [int(octave)] if octave is not None else [int(d) for d in enc.dims]
        results: dict[int, float] = {}
        for p in prefixes:
            a = va[:p].astype(np.float64)
            b = vb[:p].astype(np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                results[p] = 1.0
                continue
            a, b = a / na, b / nb
            if use_hyperbolic:
                # Lorentz lift: x=(sqrt(2),v); Minkowski <xa,xb>_L = -2+dot(a,b)
                # geodesic = arccosh(-<xa,xb>_L) = arccosh(2-dot); requires arg>=1
                dot_ab = float(np.clip(np.dot(a, b), -1.0, 1.0))
                results[p] = float(np.arccosh(max(2.0 - dot_ab, 1.0)))
            else:
                cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
                results[p] = float(1.0 - cos)
        return results[prefixes[0]] if octave is not None else results

    def compute_direction(self, node_id_a: str, node_id_b: str, octave: int | None = None) -> SemanticDirection:
        """Direction vector from node A centroid to node B centroid in octave subspace."""
        if self._pipeline is None:
            raise SemanticDirectionError("KB not initialized")
        enc = self._pipeline._encoder
        store = self._pipeline.store
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        import numpy as np
        def _centroid(nid: str) -> np.ndarray:
            node = store.get(nid)
            if node.centroid is not None:
                return np.array(node.centroid[:prefix], dtype=np.float64)
            texts = self._member_texts(node)
            if not texts:
                raise SemanticDirectionError(f"node {nid} has no members in this octave")
            vecs = enc.encode(texts)[:, :prefix].astype(np.float64)
            return vecs.mean(axis=0)
        ca = _centroid(node_id_a)
        cb = _centroid(node_id_b)
        diff = cb - ca
        mag = float(np.linalg.norm(diff))
        if mag < 1e-9:
            raise SemanticDirectionError(f"zero direction: nodes {node_id_a} and {node_id_b} are identical in octave {prefix}")
        direction = diff / mag
        na = np.linalg.norm(ca)
        nb = np.linalg.norm(cb)
        cos = float(np.dot(ca / max(na, 1e-9), cb / max(nb, 1e-9)))
        return SemanticDirection(
            from_node=node_id_a, to_node=node_id_b, octave=prefix,
            direction_vec=tuple(float(x) for x in direction),
            magnitude=mag, cosine_alignment=cos,
        )

    def best_octave(self, text_a: str, text_b: str) -> int:
        """Return the octave prefix where distance signal is sharpest (max second derivative)."""
        dists = self.semantic_distance(text_a, text_b)
        if not isinstance(dists, dict) or len(dists) < 3:
            if isinstance(dists, dict):
                return next(iter(dists)) if dists else 64
            return 64
        import math
        prefixes = sorted(dists.keys())
        log_p = [math.log(p) for p in prefixes]
        d = [dists[p] for p in prefixes]
        best_p, best_dd = prefixes[1], -1.0
        for i in range(1, len(prefixes) - 1):
            dd = abs((d[i+1] - d[i]) / max(log_p[i+1] - log_p[i], 1e-9)
                     - (d[i] - d[i-1]) / max(log_p[i] - log_p[i-1], 1e-9))
            if dd > best_dd:
                best_dd, best_p = dd, prefixes[i]
        return best_p

    def direction_search(self, anchor_text: str, direction_vec: "tuple[float, ...] | list[float]",
                         k: int = 5, octave: int | None = None) -> list[DirectionSearchResult]:
        """Find concepts in a given semantic direction from anchor; returns hits per alpha step."""
        if self._pipeline is None or not anchor_text:
            return []
        import numpy as np
        enc = self._pipeline._encoder
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        av = enc.encode([anchor_text])[0][:prefix].astype(np.float64)
        dv = np.array(direction_vec[:prefix], dtype=np.float64)
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-9:
            return []
        dv = dv / dv_norm
        results = []
        for alpha in (0.1, 0.5, 1.0, 2.0):
            probe = av + alpha * dv
            pn = np.linalg.norm(probe)
            if pn > 1e-9:
                probe = probe / pn
            probe32 = probe.astype(np.float32)
            store = self._pipeline.store
            ids = store.knn(probe32, k, Prefix(prefix))
            hits = []
            for nid in ids:
                node = store.get(nid)
                texts = self._member_texts(node)
                if not texts:
                    continue
                nv = enc.encode([texts[0]])[0][:prefix].astype(np.float64)
                nn = np.linalg.norm(nv)
                alignment = float(np.dot(nv / max(nn, 1e-9), dv)) if nn > 1e-9 else 0.0
                hits.append(SearchHit(
                    text=texts[0], score=alignment, node_id=str(node.id),
                    octave=prefix, members=tuple(texts),
                    aperture=float(getattr(node, 'aperture', 0.0)),
                    local_entropy=0.0, evidence_path_count=1,
                    uncertainty_score=float(max(0.0, 1.0 - alignment)),
                ))
            if hits:
                avg_align = sum(h.score for h in hits) / len(hits)
                results.append(DirectionSearchResult(hits=tuple(hits), alpha=alpha, alignment=avg_align))
        return results

    def fold_directions(self, node_id: str, octave: int | None = None) -> list[dict]:
        """Direction vectors from a parent node to each child node; maps 'downward intuition'."""
        if self._pipeline is None:
            return []
        store = self._pipeline.store
        parent = store.get(node_id)
        enc = self._pipeline._encoder
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        import numpy as np
        def _centroid_vec(node) -> "np.ndarray | None":
            if node.centroid is not None:
                return np.array(node.centroid[:prefix], dtype=np.float64)
            texts = self._member_texts(node)
            if not texts:
                return None
            return enc.encode(texts)[:, :prefix].astype(np.float64).mean(axis=0)
        pc = _centroid_vec(parent)
        if pc is None:
            return []
        all_nodes = [n for n in store.all_nodes() if n.prefix == Prefix(prefix) and n.id != parent.id and n.members]
        results = []
        for child in all_nodes:
            cc = _centroid_vec(child)
            if cc is None:
                continue
            diff = cc - pc
            mag = float(np.linalg.norm(diff))
            if mag < 1e-9:
                continue
            direction = diff / mag
            label = self._summarizer.summarize(str(child.id), self._member_texts(child))
            results.append({
                "child_id": str(child.id),
                "direction_vec": tuple(float(x) for x in direction),
                "magnitude": mag,
                "semantic_label": label,
            })
        results.sort(key=lambda r: -r["magnitude"])
        return results

    def search_with_reflection(self, query: str, k: int = 5,
                               reflect_fn=None) -> dict:
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

    def compute_trajectory(self, query: str, answer_node_id: str,
                           octave: int | None = None) -> SemanticTrajectory:
        """Trace the octave-descent path from query to answer_node as a SemanticTrajectory."""
        if self._pipeline is None:
            return SemanticTrajectory(steps=(), total_distance=0.0, coherence_score=0.0, energy_cost=0.0)
        import numpy as np
        enc = self._pipeline._encoder
        store = self._pipeline.store
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        q_vec = enc.encode([query])[0][:prefix].astype(np.float64)
        try:
            answer = store.get(answer_node_id)
        except KeyError:
            return SemanticTrajectory(steps=(), total_distance=0.0, coherence_score=0.0, energy_cost=0.0)
        # build path: query -> intermediate nodes at each octave -> answer
        steps: list[TrajectoryStep] = []
        prev_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
        all_dims = [int(d) for d in enc.dims if int(d) <= prefix]
        for dim in all_dims:
            ids = store.knn(q_vec[:dim].astype(np.float32), k=1, prefix=Prefix(dim))
            if not ids:
                continue
            node = store.get(ids[0])
            texts = self._member_texts(node)
            if not texts:
                continue
            nv = enc.encode([texts[0]])[0][:dim].astype(np.float64)
            nn = float(np.linalg.norm(nv))
            nv_n = nv / max(nn, 1e-9)
            cos = float(np.clip(np.dot(prev_vec[:dim], nv_n), -1.0, 1.0))
            dist = float(1.0 - cos)
            diff = nv_n - prev_vec[:dim]
            dm = float(np.linalg.norm(diff))
            direction = tuple(float(x) for x in (diff / dm if dm > 1e-9 else diff))
            steps.append(TrajectoryStep(
                node_id=str(node.id), octave=dim,
                distance_from_prev=dist, direction_vec=direction,
            ))
            prev_vec = nv_n
        total = sum(s.distance_from_prev for s in steps)
        import math
        if len(steps) > 1:
            dists = [s.distance_from_prev for s in steps]
            mean_d = total / len(dists)
            var = sum((d - mean_d) ** 2 for d in dists) / len(dists)
            entropy = math.sqrt(var) / max(mean_d, 1e-9)
            coherence = max(0.0, 1.0 - entropy / math.log(len(steps) + 1))
        else:
            coherence = 1.0
        energy = sum(s.distance_from_prev * (2 ** i) for i, s in enumerate(steps))
        return SemanticTrajectory(
            steps=tuple(steps), total_distance=total,
            coherence_score=coherence, energy_cost=energy,
        )
