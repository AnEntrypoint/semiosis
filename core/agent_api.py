"""High-level KnowledgeBase API for agents -- hides cone/manifold internals."""
from __future__ import annotations

import threading
from typing import Any

from .interfaces import Prefix, phrase_to_text_index
from .kb_diagnostics import DiagnosticsMixin
from .kb_geometry import GeometryMixin
from .kb_hierarchy import HierarchyMixin
from .kb_persistence import PersistenceMixin
from .kb_retrieval import RetrievalMixin
from .lexical_index import BM25Index
from .kb_types import (  # noqa: F401 -- re-exported for backwards compat
    QueryPriority, FailureMode,
    SearchHit, ConsolidateReport, FlowNeighbor, TensionPair, DeepSearchResult,
    CompressResult, DiagnoseReport, RetrievalStep, SemanticDirection, TrajectoryStep,
    SemanticTrajectory, DirectionSearchResult, CompressedHierarchy, RecursiveAnswerResult,
    ManifoldComplexity, FoldBudgetResult, SparseSearchResult, IngestStreamResult,
    ContrastiveDirection, QueryDecomposition, AttentionScore, AnalogyResult,
    ConceptBoundary, DispelReport, ReflectStep, CategoricalParentHit, EnergyStep,
    SemanticDirectionError, Directive, Observation, Hypothesis, ResearchStep, ResearchResult,
)
from .pipeline import KnowledgePipeline
from .semiotic_memory import SemioticMemory
from .settings import Settings


class StubSummarizer:
    """Summarizer that combines first two member texts; no LLM required; used by default."""

    def summarize(self, node_id: str, member_texts: list[str]) -> str:
        if not member_texts:
            return f"category:{node_id}"
        preview = "; ".join(member_texts[:2])
        return f"Category covering: {preview}"


class KnowledgeBase(
    RetrievalMixin, GeometryMixin, HierarchyMixin, DiagnosticsMixin, PersistenceMixin,
):
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
        self._bm25 = BM25Index(k1=self._settings.store.bm25_k1, b=self._settings.store.bm25_b)
        self._lock = threading.RLock()

    def ingest(self, texts: list[str]) -> None:
        """Add texts; new texts route to nearest leaves, full rebuild only past the tension threshold."""
        if not texts:
            return
        with self._lock:
            start_idx = len(self._texts)
            self._texts.extend(texts)
            for i, t in enumerate(texts):
                self._bm25.add(str(start_idx + i), t)
            self._metrics["ingests"] += 1
            if self._pipeline is not None and self._settings.agent.incremental_ingest:
                self._pipeline.ingest(texts)
            else:
                self._pipeline = KnowledgePipeline(self._texts, self._settings)
            self._memory._pipeline = self._pipeline
            self._memory._texts = list(self._texts)

    def ingest_stream(self, texts) -> "IngestStreamResult":
        """Batch ingest via leaf routing; reports node growth and whether a structural rebuild fired."""
        import time as _time
        texts = list(texts)
        t0 = _time.monotonic()
        with self._lock:
            before = len(self._pipeline.store.all_nodes()) if self._pipeline is not None else 0
            rebuilds_before = self._pipeline.rebuild_count if self._pipeline is not None else 0
            self.ingest(texts)
            after = len(self._pipeline.store.all_nodes()) if self._pipeline is not None else 0
            rebuilds_after = self._pipeline.rebuild_count if self._pipeline is not None else 0
        elapsed = (_time.monotonic() - t0) * 1000.0
        return IngestStreamResult(
            len(texts), max(0, after - before), rebuilds_after > rebuilds_before, elapsed)

    # --- helpers shared across every mixin ------------------------------------
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

    # --- memory: pinned facts + layered recall --------------------------------
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

    # --- context packing (delegates to ContextPackBuilder) --------------------
    def build_context_pack(self, query: str, max_tokens: int, counter=None):
        """Budget-bounded, redundancy-free context pack mitigating context rot."""
        from .context_pack import ContextPack, ContextPackBuilder, ContextPackConfig
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
