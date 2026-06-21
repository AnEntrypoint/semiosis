# NLA/Attention Analysis: Complete Summary

**Date:** 2026-06-21  
**Scope:** How does the Neural Logic Architecture (NLA) paper's use of attention mechanisms apply to semiosis cone geometry?

---

## Quick Start

Three new documents added to `docs/`:

1. **`nla-attention-hypothesis.md`** (12 KB) — Full technical analysis of 4 hypotheses
2. **`nla-attention-action-plan.md`** (8 KB) — Prioritized action items for Phase 2-4
3. **`phase2-entropy-implementation.md`** (10 KB) — Concrete code walkthrough for Phase 2
4. **`README-nla-analysis.md`** (this file) — Navigation guide

---

## The Four Hypotheses

| # | Question | Evidence | Verdict | Action |
|---|----------|----------|---------|--------|
| **1** | Does cone containment approximate softmax attention? | Geometric, not probabilistic | Partial ✓ | Research-grade validation (Phase 5+) |
| **2** | Should we add explicit information-flow scoring? | Plausible but unvalidated | Speculative | No; wait for empirical mismatch (Phase 5) |
| **3** | Can octaves implement multi-head-like structures? | Yes; dimensional hierarchy matches | **Strong ✓** | **Yes; design Phase 4** |
| **4** | Should context-packing use attention-like weighting? | Yes; information-bottleneck theory | **Strong ✓** | **Yes; implement Phase 2** |

**TL;DR:** Hypotheses 3 & 4 are actionable now; 1 & 2 are research-grade.

---

## What Changed

### Hypothesis #3: Octaves as Multi-Heads ✓

**Finding:** Semiosis already has a multi-head-like structure via Matryoshka octaves. Each octave (64D, 128D, 256D, 512D, 1024D) encodes different granularities — just like attention heads specialize on different interaction patterns.

**Gap:** Octave fusion is currently uniform RRF (reciprocal rank fusion). Can be improved by learning query-type-specific weights.

**Action:** 
- Phase 3: Learning loop trains octave weights from feedback
- Phase 4: Deploy learned weights with query-type classification

**Expected impact:** 10-15% recall improvement on complex queries.

---

### Hypothesis #4: Attention-Weighted Context Packing ✓

**Finding:** Current context packing is greedy-by-relevance. Ignores information density (member diversity). Information-bottleneck theory says we should weight entries by both relevance AND entropy.

**Gap:** No entropy metric in current system.

**Action:** 
- Phase 2: Add entropy computation to ConeNode
- Phase 2: Implement IB-weighted context selection
- Phase 4: Train critic network to predict user satisfaction

**Expected impact:** 5-10% better context quality; faster LLM inference on same knowledge.

---

## Implementation Roadmap

### Phase 2: Entropy Foundations (Ready to start)

**Files to modify:**
- `core/interfaces.py` — Add entropy fields to ConeNode
- `core/context_pack.py` — Implement entropy computation + IB weighting
- `core/test_*.py` — Add unit and integration tests

**PRD rows:**
- `node-entropy-estimation` (new)
- `context-pack-ib-weighting` (new, split from existing row)

**Effort:** 3-4 hours

**Success metric:** Recall@k unchanged or improved; entropy correlates with member diversity.

**Code entry points:**
- `EntropyEstimator.shannon_entropy()` — Compute entropy from distances
- `ContextPackBuilder._compute_node_entropy()` — Per-node entropy
- `ContextPackBuilder.build(..., use_ib_weighting=True)` — IB-weighted selection

---

### Phase 3: Learning Loop (Existing plan + entropy signals)

Use entropy signals from Phase 2 to train octave weights:
- High-entropy queries → favor fine octaves (detail-seeking)
- Low-entropy queries → favor coarse octaves (broad overview)

**PRD rows:** Existing 4 rows unchanged.

**Interdependency:** Phase 2 entropy enables Phase 3 learning signals.

---

### Phase 4: Multi-Head Octave Specialization (Design ready)

**New class:** `MultiHeadOctaveRouter`
- N heads (e.g., 3: fact-seeking, analogy, synthesis)
- Each head has learned weights α_1, ..., α_k (per octave)
- Simple query classifier routes query to appropriate head

**Files:**
- `core/agent_api.py` or new `core/router.py` — MultiHeadOctaveRouter
- `core/classifier.py` (new) — QueryTypeClassifier

**Effort:** 2-3 days (skeleton ready in action plan doc).

**Success metric:** Per-query-type recall improvement ≥ 10%.

---

### Phase 5+: Research Validations (Optional)

If empirical results warrant:

- **Hypothesis #1:** Label cone containment patterns vs BERT/LLaMA attention heatmaps; compute correlation
- **Hypothesis #2:** Implement FlowNetwork (learnable information-flow scorer); compare vs baseline

---

## Key Insights from Papers

### NLA Paper (transformer-circuits.pub/2026/nla)

**Main idea:** Attention mechanisms are routing primitives that guide information flow through a network.

**Semiosis connection:** Cone containment acts like a geometric attention mask (select children inside cone; penalize outside).

**Gap:** Cones don't have backprop-mediated learning; attention weights do. Can bridge with learned head weights (Phase 4).

---

### Recursive Language Models Paper (arxiv 2512.24601)

**Main idea:** Hierarchical structures enable multi-scale reasoning; coarse reasoning before fine.

**Semiosis connection:** Octave descent (coarse → fine) is exactly RLM's hierarchical decomposition.

**Gap:** Current octave fusion doesn't adapt per query type. Multi-head specialization (Phase 4) fixes this.

---

### Information-Bottleneck Theory (Tishby & Zamir)

**Main idea:** Compress information while preserving query relevance; trade off information loss vs utility.

**Semiosis connection:** Context packing is information bottleneck applied to facts. Entropy weighting operationalizes the theory.

**Gap:** No explicit information-value metric. Phase 2 adds it via entropy.

---

## FAQ

### Q: Are these changes mandatory for correctness?

**A:** No. Semiosis is correct today. These are incremental improvements:
- Phase 2: Better context quality (5-10% edge)
- Phase 4: Better routing for complex queries (10-15% edge)

Both are wins but not critical path.

---

### Q: Why wait until Phase 4 for multi-head octaves?

**A:** Phase 3 learning loop must finish first. Head weights are learned from feedback (octave-weighting row in Phase 3). Phase 4 deploys those weights with intent classification.

---

### Q: Do we need to understand NLA/attention to implement Phase 2?

**A:** No. Phase 2 is purely information-theoretic: entropy computation + weighted selection. No attention mechanisms involved. Self-contained.

---

### Q: What if entropy weighting degrades retrieval quality?

**A:** Phase 2 is designed conservatively:
- use_ib_weighting parameter (default True, can be disabled)
- Tests compare vs baseline (no regression allowed)
- Feature flag in settings to roll back if needed

If empirical eval shows regression, revert to greedy-by-relevance.

---

### Q: Can we do Phase 4 without Phase 2?

**A:** Technically yes, but Phase 2 is a fast prerequisite (3-4 hours). Multi-head router (Phase 4) benefits from entropy signals (Phase 2). Recommend doing both.

---

## Document Map

| Document | Purpose | Audience |
|----------|---------|----------|
| `nla-attention-hypothesis.md` | Deep technical analysis of all 4 hypotheses | Research, architects |
| `nla-attention-action-plan.md` | Prioritized actions + PRD updates | Engineering leads |
| `phase2-entropy-implementation.md` | Step-by-step code walkthrough | Implementers |
| `README-nla-analysis.md` (this) | Navigation + quick reference | Everyone |

---

## Success Metrics (End of Phase 4)

1. **Phase 2 (Entropy):**
   - Entropy is computed and cached on all ConeNodes
   - Context packs weight entries by IB score
   - Recall@k unchanged or improved
   - Entropy correlates with member diversity (Pearson ρ > 0.7)

2. **Phase 3 (Learning Loop):**
   - Octave weights converge from feedback
   - Recall improves 5-10% vs baseline

3. **Phase 4 (Multi-Head):**
   - Specialized heads beat uniform RRF by ≥10% per query type
   - Query classifier achieves ≥80% accuracy
   - Latency overhead <10%

---

## Risk Summary

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Entropy computation expensive | Low | Cache at fit time; profile first |
| IB weighting breaks tests | Low | Run baseline; use feature flag |
| Learning loop doesn't converge | Medium | Conservative step sizes; regularize |
| Multi-head classifier is inaccurate | Medium | Start with heuristic; monitor labels |
| Phase 3 delays Phase 4 | Low | Phase 4 skeleton ready now; block on Phase 3 only for weight learning |

---

## Next Steps

1. **Read the three analysis documents** (in order of detail):
   - Start: `README-nla-analysis.md` (you are here)
   - Deep dive: `nla-attention-hypothesis.md`
   - Implementation: `nla-attention-action-plan.md` + `phase2-entropy-implementation.md`

2. **Planning decision:**
   - Approve Phase 2 PRD additions
   - Approve Phase 4 design sketch

3. **Phase 2 kickoff:**
   - Assign implementer
   - Create GitHub issues for Steps 1-5
   - Set success metrics and baseline measurements

---

## References

- **Core codebase:** `core/cone_engine.py`, `core/context_pack.py`, `core/agent_api.py`
- **Configuration:** `core/settings.py`
- **Tests:** `core/test_manifold_invariants.py`, `core/test_encoder.py`
- **Papers:** arxiv 2512.24601 (RLM), transformer-circuits.pub/2026/nla, arxiv 2205.13147 (Matryoshka)
- **Existing PRD:** `docs/paper-insights-summary.md` (50-row roadmap)

---

**Status:** Analysis complete. Ready for engineering team review and decision.

**Questions?** See `nla-attention-hypothesis.md` (lines 450-550) for detailed FAQ; `nla-attention-action-plan.md` for risk tables.
