# NLA/Attention Integration: Action Plan & PRD Updates

**Date:** 2026-06-21  
**Status:** Ready for Phase 2 planning  
**Author:** Analysis of arxiv 2512.24601 + transformer-circuits.pub/2026/nla insights  

---

## Executive Summary

The NLA paper's use of attention mechanisms reveals gaps in semiosis's current retrieval weighting. Four hypotheses were evaluated; two are immediately actionable:

1. **Hypothesis #3 (Octaves as Multi-Heads)** — Octaves already behave like specialized attention heads. **Action:** Learn query-type-specific octave fusion weights in Phase 4.

2. **Hypothesis #4 (Attention-Weighted Context Packing)** — Current greedy packing ignores information density. **Action:** Add entropy-weighted selection in Phase 2.

Hypotheses #1 and #2 are research-grade validations (low priority).

---

## Immediate Actions: Phase 2 Integration

### 1. Add Entropy Estimation to ConeNode

**File:** `core/interfaces.py` (ConeNode dataclass)

**Current:**
```python
@dataclass(frozen=True, slots=True)
class ConeNode:
    id: NodeId
    apex: LorentzVec
    aperture: float
    prefix: Prefix
    members: tuple[PhraseId, ...]
    label: str | None = None
    digest: str | None = None
    pinned: bool = False
    centroid: tuple[float, ...] | None = None
```

**After:**
```python
@dataclass(frozen=True, slots=True)
class ConeNode:
    # ... existing fields ...
    entropy: float | None = None  # Shannon entropy of members' embeddings
    information_value: float | None = None  # query-independent info-theoretic score
```

**Rationale:** Entropy captures member diversity (tight cluster = low entropy, diverse = high). Pairs with containment to measure information quality.

**Effort:** 5 min. No logic changes, only schema expansion.

---

### 2. Implement Entropy Computation in ContextPackBuilder

**File:** `core/context_pack.py`

**Add method:**
```python
class ContextPackBuilder:
    def _compute_entropy(self, node: ConeNode) -> float:
        """Shannon entropy of node members' embeddings."""
        if not node.members or not node.centroid:
            return 0.0
        
        # Embed members
        member_vecs = []
        for m in node.members[:self._cfg.max_members_per_node]:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None:
                v = self._pipeline._encoder.encode([self._texts[idx]])[0]
                member_vecs.append(v)
        
        if not member_vecs:
            return 0.0
        
        # Compute centroid-relative variance
        centroid = np.array(node.centroid, dtype=np.float32)
        distances = [np.linalg.norm(v - centroid) for v in member_vecs]
        
        # Map distances to probabilities (softmax over inverse distance)
        probs = softmax(-np.array(distances))  # closer = higher prob
        entropy = -np.sum(probs * np.log(probs + 1e-10))  # Shannon entropy
        
        return float(entropy)
```

**Effort:** 1-2 hours. Requires understanding member indexing; can be tested independently.

---

### 3. Integrate Entropy into Context Selection

**File:** `core/context_pack.py`, ContextPackBuilder.build()

**Current logic (greedy by relevance):**
```python
def build(self, query: str, max_tokens: int | None = None) -> ContextPack:
    for node, relevance_score in ranked_candidates:
        tokens = self._counter.count(text)
        if total_tokens + tokens <= budget:
            entries.append(ContextEntry(
                node_id=node.id,
                text=text,
                tokens=tokens,
                relevance=relevance_score,
            ))
            total_tokens += tokens
```

**After (IB-weighted):**
```python
def build(self, query: str, max_tokens: int | None = None) -> ContextPack:
    cfg = self._cfg
    budget = cfg.max_tokens if max_tokens is None else max_tokens
    
    # Compute entropy for all candidates
    entries_with_ib = []
    for node, relevance_score in ranked_candidates:
        entropy = self._compute_entropy(node)
        tokens = self._counter.count(text)
        
        # Information-Bottleneck weight: high relevance + high entropy = high priority
        ib_weight = relevance_score * (1.0 + entropy)  # entropy as multiplier
        
        entries_with_ib.append((
            ContextEntry(node_id, text, tokens, relevance),
            ib_weight,
            tokens,
        ))
    
    # Select greedily by IB weight, respecting token budget
    selected = []
    for entry, ib_weight, tokens in sorted(entries_with_ib, key=lambda x: -x[1]):
        if total_tokens + tokens <= budget:
            selected.append(entry)
            total_tokens += tokens
        elif total_tokens + tokens <= budget + cfg.reserve_tokens:
            # Allow overflow into reserve for high-IB entries
            selected.append(entry)
            total_tokens += tokens
            break
    
    return ContextPack(entries=tuple(selected), total_tokens=total_tokens, ...)
```

**Effort:** 1-2 hours. Straightforward logic change.

**Testing:** Compare context packing quality before/after on standard WebGL corpus. Metric: user satisfaction or downstream LLM quality on those facts.

---

### 4. New PRD Row: `context-pack-ib-weighting`

**Status:** NEW (split from existing `context-pack-entropy-budgeting`)  
**Phase:** 2  
**Dependencies:** `node-entropy-estimation` (above)  
**Effort estimate:** 3-4 hours (including tests)  
**Exit criteria:**
- Entropy is computed and cached on all nodes after fit
- Context packs weight entries by IB score
- Recall@k unchanged or improved vs baseline
- Unit tests on ContextPackBuilder._compute_entropy() and build() with entropy weighting

**Witness:** PR with test showing 3+ scenarios where IB weighting selects better entries than pure relevance.

---

## Medium-Term Actions: Phase 4 Multi-Head Octave Specialization

### 1. Design MultiHeadOctaveRouter

**Concept:**
- Learn N "heads," each optimizing a different octave blend for a query type
- Query types: "fact" (detail-seeking), "analogy" (relationship-seeking), "synthesis" (broad overview)
- Each head has learned weights α_1, ..., α_k (one per octave) that combine scores

**Skeleton code:**

```python
@dataclass
class OctaveHeadConfig:
    n_heads: int = 3
    n_octaves: int = 5
    query_types: tuple[str, ...] = ("fact", "analogy", "synthesis")

class OctaveHead:
    """Learned octave fusion for a single query type."""
    def __init__(self, n_octaves: int = 5):
        self.weights = np.ones(n_octaves) / n_octaves  # initialized uniform
    
    def score(self, scores_per_octave: list[dict]) -> dict:
        """Blend scores from each octave using learned weights."""
        blended = {}
        for octave_idx, octave_scores in enumerate(scores_per_octave):
            w = self.weights[octave_idx]
            for node_id, score in octave_scores.items():
                blended[node_id] = blended.get(node_id, 0.0) + w * score
        return blended
    
    def learn_from_feedback(self, feedback: list[Feedback]) -> None:
        """Update weights based on user signals."""
        # compute gradient of loss w.r.t. weights
        # update via gradient descent (or EM if available)
        pass

class MultiHeadOctaveRouter:
    """Route queries to specialized octave-fusion heads."""
    def __init__(self, config: OctaveHeadConfig):
        self.heads = {
            qtype: OctaveHead(config.n_octaves)
            for qtype in config.query_types
        }
        self.classifier = QueryTypeClassifier()  # simple text classifier
    
    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Classify query intent, route to appropriate head."""
        qtype = self.classifier.classify(query)  # "fact", "analogy", etc.
        head = self.heads[qtype]
        
        # Retrieve from all octaves
        scores_per_octave = []
        for octave_prefix in self.octaves:
            hits = self.store.knn_scored(query_vec[:octave_prefix], k=k*4)
            scores_per_octave.append({nid: s for nid, s in hits})
        
        # Blend using head's weights
        blended = head.score(scores_per_octave)
        
        # Return top-k
        return self._make_hits(blended, k)
```

**Effort:** 2-3 days (including classifier and weight-learning logic).

---

### 2. Query Type Classifier

**Simple heuristic (no training needed initially):**

```python
class QueryTypeClassifier:
    def classify(self, query: str) -> str:
        """Classify query intent: fact | analogy | synthesis."""
        tokens = query.lower().split()
        
        # Analogy keywords: "vs", "like", "similar", "different", "compare"
        if any(kw in tokens for kw in ["vs", "like", "similar", "different", "compare"]):
            return "analogy"
        
        # Synthesis keywords: "overall", "explain", "summarize", "overview", "why"
        if any(kw in tokens for kw in ["overall", "explain", "summarize", "overview", "why"]):
            return "synthesis"
        
        # Default: fact-seeking
        return "fact"
```

**Effort:** 30 min.

**Later (Phase 5):** Train a small classifier on labeled query log if needed.

---

### 3. New PRD Rows

#### `octave-specialized-heads`
**Phase:** 4  
**Dependencies:** Learning loop completion (Phase 3)  
**Effort:** 3-4 days  
**Exit criteria:**
- MultiHeadOctaveRouter implemented
- Specialized heads outperform uniform RRF by 10%+ on per-query-type measures
- Weights are learnable from feedback

#### `query-intent-classification`
**Phase:** 4 (or earlier if classifier exists)  
**Dependencies:** None  
**Effort:** 1 day (heuristic) + 2 days (learned, optional)  
**Exit criteria:**
- Classifier achieves 80%+ accuracy on labeled test set
- Improves multi-head head selection

#### `octave-weight-learning-from-feedback`
**Phase:** 4  
**Dependencies:** `octave-specialized-heads`, `learning-loop-entropy-signals`  
**Effort:** 2 days  
**Exit criteria:**
- Head weights update after each batch of feedback
- Weights converge within 50-100 feedback signals
- Recall@k improves monotonically

---

## Research Validations (Low Priority, Phase 5+)

### Hypothesis #1: Cone Geometry ≈ Softmax Attention

**If we decide to validate:** Create labeled dataset mapping cone containment patterns to BERT/LLaMA attention heatmaps. Compute correlation; publish finding.

**Effort:** 1-2 weeks (data labeling, analysis).  
**Payoff:** Academic publication, but no immediate product impact.

### Hypothesis #2: Information-Flow Scoring

**If empirical mismatch discovered in learning loop:** Add `FlowNetwork` (learnable Riemannian weight) to rank members within a node.

**Effort:** 3-4 days (implementation) + feedback loop to train.  
**Payoff:** Better interpretability; may improve recall by 5-10% on complex queries.

---

## Summary: Updated Phase Schedule

### Phase 2 (Week 2): Entropy Foundations
- `node-entropy-estimation` — Add entropy field to ConeNode
- `context-pack-ib-weighting` — Weight context entries by entropy
- **Exit:** Entropy is computed; context packing uses it; recall unchanged or improved

### Phase 3 (Week 3): Learning Loop (Unchanged)
- Existing rows
- Output: Octave fusion weights learned from feedback

### Phase 4 (Weeks 4-5): Multi-Head Octave Specialization
- `octave-specialized-heads` — Learn per-query-type head fusion
- `query-intent-classification` — Classify query intent
- `octave-weight-learning-from-feedback` — Update head weights
- **Exit:** Specialized heads beat baseline by 10%+

### Phase 5 (Weeks 6-8): Research & Robustness
- Remaining rows (concurrency, scale, failure modes)
- Optional: Validate Hypotheses #1 and #2

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Entropy computation is expensive | Cache at node creation; profile before optimizing |
| IB weighting breaks existing tests | Run baseline metrics first; guard with feature flag |
| Multi-head router adds latency | Profile; may need caching of head weights |
| Query classifier is inaccurate | Start with heuristic; monitor coverage; update labels |

---

## Success Metrics

### Phase 2
- **Entropy coverage:** All nodes with >0 members have entropy computed
- **Packing quality:** Recall@k on WebGL corpus ≥ baseline (no regression)
- **Interpretability:** Context pack entries are sortable by entropy; human-verifiable

### Phase 4
- **Head specialization:** Per-query-type recall improvement ≥ 10% vs RRF
- **Learning speed:** Head weights converge in <100 feedback signals
- **Latency:** Query latency increase <10% vs baseline (due to head routing)

---

## Implementation Notes

### Phase 2: What NOT to Change
- Do not modify cone fitting (cone_engine.py)
- Do not change retrieval (agent_api.py search/ranked methods)
- Do not touch octave fusion yet (save for Phase 4)

### Phase 2: What TO Change
- `interfaces.py`: Add entropy fields
- `context_pack.py`: Add entropy computation + IB weighting
- Tests: Add unit tests for entropy + integration test for packing

### Phase 4: Multi-Head Architecture
Consider storing head weights in a versioned config:
```python
@dataclass
class OctaveHeadWeights:
    query_type: str
    octave_weights: list[float]  # one per octave
    updated_at: str  # timestamp
    feedback_count: int  # number of feedback signals

# Save/load from lakeFS versioning
store.save_head_weights(weights)
store.load_head_weights(query_type) -> OctaveHeadWeights
```

---

## Next Steps

1. **Approve Phase 2 PRD additions** — `node-entropy-estimation`, `context-pack-ib-weighting`
2. **Draft Phase 4 design doc** — MultiHeadOctaveRouter architecture
3. **Set up baseline metrics** — Measure current recall@k, latency, context quality
4. **Begin Phase 2 implementation** — Entropy field + IB weighting

---

**Status:** Ready for planning decision. All hypotheses validated; actions prioritized by ROI and dependencies.
