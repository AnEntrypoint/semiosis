"""Token-budgeted context-pack builder; mitigates context rot via dedup and semiotic distancing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .interfaces import NodeId, Prefix, phrase_to_text_index


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Dependency-free token estimate at ~4 chars per token; override via Settings for a real tokenizer."""

    def count(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True, slots=True)
class ContextEntry:
    node_id: NodeId
    text: str
    tokens: int
    relevance: float
    is_summary: bool = False
    represented: tuple[NodeId, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextPack:
    entries: tuple[ContextEntry, ...] = ()
    total_tokens: int = 0
    dropped_ids: tuple[NodeId, ...] = ()
    truncated: bool = False
    degraded: bool = False
    low_confidence: bool = False

    def render(self, separator: str = "\n\n") -> str:
        """Join entries into one prompt string with per-entry provenance markers."""
        parts = []
        for e in self.entries:
            tag = "summary" if e.is_summary else "fact"
            parts.append(f"[{tag} {e.node_id}] {e.text}")
        return separator.join(parts)


@dataclass
class ContextPackConfig:
    max_tokens: int = 2048
    overlap_threshold: float = 0.5
    distance_summary_threshold: float = 0.0
    max_members_per_node: int = 4
    reserve_tokens: int = 64
    max_dedup_candidates: int = 256
    entropy_weight: float = 0.0


class ContextPackBuilder:
    """Assemble a budget-bounded, redundancy-free context pack over a fitted cone store."""

    def __init__(self, pipeline, texts: Sequence[str], config: ContextPackConfig,
                 counter: "TokenCounter | None" = None) -> None:
        self._pipeline = pipeline
        self._texts = list(texts)
        self._cfg = config
        self._counter = counter or HeuristicTokenCounter()

    def _degraded(self) -> bool:
        """True iff the real encoder failed to load (see pipeline.encoder_fallback_reason for why)."""
        enc = getattr(self._pipeline, "_encoder", None)
        return type(enc).__name__ == "RandomEncoder"

    def _node_text(self, node) -> str:
        """Resolve up to max_members_per_node distinct member texts for a node."""
        seen: list[str] = []
        for m in node.members:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is None:
                continue
            t = self._texts[idx]
            if t not in seen:
                seen.append(t)
            if len(seen) >= self._cfg.max_members_per_node:
                break
        return " ".join(seen)

    def _summary_text(self, node) -> str:
        """Cheap stand-in for a distant cone: its digest, label, or a bounded cluster marker."""
        if node.digest:
            return node.digest
        if node.label:
            return str(node.label)
        head = self._node_text(node).split(" ")[:8]
        return f"[cluster of {len(node.members)}: {' '.join(head)}]"

    def build(self, query: str, max_tokens: "int | None" = None) -> ContextPack:
        cfg = self._cfg
        budget = cfg.max_tokens if max_tokens is None else max_tokens
        pipeline = self._pipeline
        if pipeline is None or not self._texts:
            return ContextPack()
        if budget <= 0:
            return ContextPack(truncated=True)

        enc = pipeline._encoder
        prefix = Prefix(enc.dims[0])
        q_vec = enc.encode([query])[0]
        over_fetch = min(len(pipeline.store.all_nodes()),
                         max(cfg.max_dedup_candidates, 1))
        ids = list(pipeline.query.knn(q_vec[:prefix], k=over_fetch, prefix=prefix))
        if not ids:
            return ContextPack(truncated=False, degraded=self._degraded())

        engine = pipeline.engine
        store = pipeline.store
        nodes = [store.get(i) for i in ids]
        n = len(nodes)
        ranked = [(node, 1.0 - i / n) for i, node in enumerate(nodes)]

        kept: list = []
        dropped: list[NodeId] = []
        for node, rel in ranked:
            redundant = any(engine.overlap_score(k, node) > cfg.overlap_threshold for k, _ in kept)
            if redundant and kept:
                dropped.append(node.id)
                continue
            kept.append((node, rel))

        top_node = kept[0][0] if kept else None
        entries: list[ContextEntry] = []
        for node, rel in kept:
            distant = (
                top_node is not None
                and node.id != top_node.id
                and engine.overlap_score(top_node, node) < cfg.distance_summary_threshold
            )
            if distant:
                text = self._summary_text(node)
                entries.append(ContextEntry(node.id, text, self._counter.count(text), rel,
                                            is_summary=True, represented=tuple(
                                                [node.id])))
            else:
                text = self._node_text(node)
                if not text:
                    continue
                entries.append(ContextEntry(node.id, text, self._counter.count(text), rel))

        return self._pack(entries, budget, dropped)

    def _pack(self, entries: Sequence[ContextEntry], budget: int,
              dropped: Sequence[NodeId]) -> ContextPack:
        cfg = self._cfg
        free = max(0, budget - cfg.reserve_tokens)
        out: list[ContextEntry] = []
        seen_text: set[str] = set()
        total = 0
        truncated = False
        dropped = list(dropped)
        for e in sorted(entries, key=lambda x: (-x.relevance, str(x.node_id))):
            if e.text in seen_text:
                continue
            if total + e.tokens <= free:
                out.append(e)
                seen_text.add(e.text)
                total += e.tokens
            elif not out and e.tokens > free:
                trunc = self._truncate(e.text, free)
                tok = self._counter.count(trunc)
                out.append(ContextEntry(e.node_id, trunc, tok, e.relevance, e.is_summary, e.represented))
                total += tok
                truncated = True
                break
            else:
                dropped.append(e.node_id)
                truncated = True
        return ContextPack(
            entries=tuple(out),
            total_tokens=total,
            dropped_ids=tuple(dropped),
            truncated=truncated,
            degraded=self._degraded(),
            low_confidence=not out,
        )

    def _truncate(self, text: str, budget: int) -> str:
        """Binary-search the longest text prefix whose token count fits the budget."""
        if budget <= 0:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._counter.count(text[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]
