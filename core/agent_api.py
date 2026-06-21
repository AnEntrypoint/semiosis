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


class KnowledgeBase:
    """Ingest texts, search, navigate meaning flow, learn from outcomes, persist across sessions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._pipeline: KnowledgePipeline | None = None
        self._texts: list[str] = []
        self._memory = SemioticMemory(None, [], self._settings)
        self._usage: dict[str, int] = {}
        self._metrics: dict[str, int] = {
            "queries": 0, "ingests": 0, "record_outcomes": 0, "consolidations": 0,
        }

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
