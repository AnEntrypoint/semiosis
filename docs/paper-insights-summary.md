# Paper-Derived Insights: Implementation Roadmap

**Date:** 2026-06-21
**Source Papers:**
1. arxiv 2512.24601 (Recursive Language Models on Hierarchical Structures)
2. transformer-circuits.pub/2026/nla (Neural Logic Architecture)
3. Original Inspiration: https://claude.ai/share/2f993e0a-246d-472d-af4c-4397df09b57a

## Executive Summary

This document maps 50 PRD rows generated from the above papers into semiosis, a hyperbolic entailment-cone semantic knowledge base. The papers inspire three core insights:

1. **Information-Bottleneck Folding** — Compress information hierarchically while preserving query-relevance.
2. **Hierarchical Meaning-Making** — Different abstraction levels reveal different aspects; meaning emerges from structure.
3. **Energy/Entropy Optimization** — Manage computational cost and information density jointly across scales.

## Concept-to-PRD Mapping

### 1. Information-Bottleneck Principle

**Paper Insight:** RLM and NLA compress information across hierarchical levels. Information-bottleneck (IB) theory formalizes this: keep only what is relevant to the query.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| cross-octave-information-fusion | Fuse octaves via IB (compress while preserving query relevance) | pending |
| entropy-weighted-retrieval | Weight retrieved nodes by entropy to reward dense clusters | pending |
| context-pack-entropy-budgeting | Allocate tokens by information-value, not just count | pending |
| distillation-of-summaries-via-information-bottleneck | Query-adaptive summaries via IB principle | pending |

**Test Surface:** `pytest core/test_information_geometry.py::test_ib_compression`

---

### 2. Hierarchical Folding & Dimensional Reduction

**Paper Insight:** Hierarchies emerge from dimensional structure. Lower dimensions capture coarse abstractions; higher dimensions capture details. Folding means collapsing high-dimensional structure into lower-dimensional representations.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| octave-boundary-detection | Auto-detect octave boundaries via information-density gaps | pending |
| cone-aperture-auto-tuning | Aperture scales with member entropy (tight=coherent, wide=diverse) | pending |
| embedding-subspace-alignment | Verify Matryoshka structure via canonical correlation | pending |
| sampling-strategy-for-large-stores | Stratified sampling enables multi-scale learning on 100k+ nodes | pending |

**Test Surface:** `pytest core/test_manifold_invariants.py::test_octave_hierarchy`

---

### 3. Multi-Scale Reasoning & Query Decomposition

**Paper Insight:** Hierarchical structures enable queries to operate at different scales. A query about "texture optimization" fits at middle octave; decompose into detail octaves for specifics, coarse octaves for context.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| hierarchical-query-decomposition | Decompose queries using octave structure as guide | pending |
| query-intent-latent-recovery | Reveal user's implicit reasoning level from query embedding | pending |
| query-decomposition-via-latent-semantics | Learn system's own ontology via factor analysis | pending |
| octave-latency-profiling | Profile latency per octave; compute max_depth for latency budget | pending |

**Test Surface:** `/explain-hierarchy endpoint; trace.length == max_depth`

---

### 4. Energy & Computational Efficiency

**Paper Insight:** Papers emphasize efficient information processing. Folding reduces computation: coarse octaves are cheaper; fine octaves are expensive. Budget computational energy.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| energy-tracking-and-budgeting | Track & budget computational energy across pipeline | pending |
| hierarchical-cache-strategy | Cache octave-aware (summaries stay 10x longer) | pending |
| octave-latency-profiling | Profile latency per octave; guide depth budgeting | pending |
| implicit-feedback-octave-weighting | Learn which octaves work best for which queries | pending |

**Test Surface:** `search(query, query_priority='low')` returns partial results within energy budget.

---

### 5. Semiotic Structure & Meaning-Making

**Paper Insight:** Meaning is not intrinsic; it emerges from hierarchical structure. Different levels of abstraction create different meanings.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| semiotic-meaning-extraction | ConeNode holds (content/meaning/form) triple | pending |
| emergence-detection-via-clustering | Detect emergent abstraction levels at cluster-stability transitions | pending |
| semantic-drift-detection | Track meaning_vector changes; signal when cluster concept evolves | pending |
| tension-as-information-flow | Reframe cone geometry as information loss/routing | pending |

**Test Surface:** `meaning_vector` most-similar to high-value members.

---

### 6. Learning from Structure & Feedback

**Paper Insight:** Hierarchies are learnable and adaptable. System discovers good reasoning levels through feedback.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| learning-loop-entropy-signals | Flag queries with entropy divergence from training | pending |
| implicit-feedback-octave-weighting | Learn which octaves work best for which query kinds | pending |
| multi-scale-feature-learning | Learn feature importance weights per octave scale | pending |
| hierarchical-relevance-feedback-loop | Update octave apertures based on user signals | pending |
| cross-domain-transfer-learning | Transfer learned strategies across domains | pending |

**Test Surface:** Learning loop outcomes show 20% MRR improvement after 50 feedback signals.

---

### 7. Robustness & Edge Cases

**Paper Insight:** Hierarchical systems have failure modes (boundary ambiguity, over-compression, octave mismatch). Robustness means anticipating and recovering from all modes.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| failure-mode-taxonomy | Enumerate 5 failure modes + remediation strategies | pending |
| meta-reasoning-cone-probe | Explain retrieval failures (outside_all/boundary/multi_match) | pending |
| cone-collapse-detection | Detect & prevent aperture collapse | pending |
| graceful-degradation-on-partial-failures | Tiered fallback when parts unavailable | pending |
| consistency-checks-post-mutation | Verify acyclicity, transitivity, reachability | pending |
| concurrent-access-and-thread-safety | MVCC model; atomic writes; non-blocking reads | pending |
| query-injection-and-adversarial-inputs | Sanitize inputs; prevent embedding degeneration | pending |
| numeric-stability-underflow-overflow | Log-space entropy; clipped distances | pending |
| extremely-large-clusters-memory-safety | Cap cluster size; sample for stats; lazy load | pending |

**Test Surface:** Stress tests with adversarial inputs; ThreadSanitizer clean.

---

### 8. Diagnostic & Observability

**Paper Insight:** Understanding requires visibility. Expose system reasoning so humans can debug and improve it.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| api-endpoint-explain-hierarchy | /explain-hierarchy endpoint with full traversal trace | pending |
| information-density-heatmap | Visualize compression quality across hierarchy | pending |
| learning-loop-outcome-distribution-tracking | Track outcomes per (query_kind, octave, memory_layer) | pending |
| uncertainty-quantification-in-retrieval | Return uncertainty per result for downstream handling | pending |

**Test Surface:** Web UI renders hierarchy traces; heatmap identifies 3+ compression-quality regions.

---

### 9. Research Integration & Meta-Learning

**Paper Insight:** System should learn from reading new research. Auto-ingest papers, extract insights, propose improvements.

**Semiosis Implementation:**

| PRD Row | Feature | Status |
|---------|---------|--------|
| paper-integration-pipeline | Subscribe to arxiv; auto-summarize papers; propose PRD rows | pending |
| research-integration-document | docs/paper-integration.md mapping papers -> rows -> code | pending |
| comprehensive-paper-insights-summary | Master roadmap (this document) | in_progress |

**Test Surface:** Successfully ingest test paper; propose 3+ relevant PRD rows.

---

## Implementation Phases

### Phase 1: High-Value, Low-Effort (Week 1)
- entropy-weighted-retrieval
- cone-aperture-auto-tuning
- transitive-containment-closure
- uncertainty-quantification-in-retrieval

**Exit Criteria:** All 4 rows resolved; retrieval entropy decreases; tests pass.

### Phase 2: Semiotic Core (Week 2)
- semiotic-meaning-extraction
- tension-as-information-flow
- semantic-drift-detection
- emergence-detection-via-clustering

**Exit Criteria:** ConeNode has meaning_vector; tension correlates with information loss; tests pass.

### Phase 3: Learning Loop (Week 3)
- learning-loop-entropy-signals
- implicit-feedback-octave-weighting
- multi-scale-feature-learning
- hierarchical-relevance-feedback-loop

**Exit Criteria:** Learning loop converges; octave weighting improves retrieval 15%+.

### Phase 4: Advanced (Weeks 4-6)
- hierarchical-query-decomposition
- query-decomposition-via-latent-semantics
- cross-octave-information-fusion
- evidence-aggregation-credibility-weighting
- paper-integration-pipeline

**Exit Criteria:** Query-adaptive summaries work; octave fusion beats RRF by 30%; arxiv pipeline ingests papers.

### Phase 5: Robustness & Scale (Weeks 7-8)
- Remaining 15 rows (edge cases, concurrency, large-store handling, adversarial tests)

**Exit Criteria:** ThreadSanitizer clean; 100k nodes fit in memory; all failure modes handled.

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Octave boundaries misaligned with natural structure | Validate via CCA; heatmap visualization |
| Energy budgeting too aggressive (loses retrieval quality) | Tune via A/B test; track recall@k vs energy |
| Learning loop oscillates (unstable aperture updates) | Use conservative step sizes; regularize updates |
| Meaning vectors drift from semantics | Manual validation on WebGL domain; correlation metric |
| Concurrent updates corrupt hierarchy | MVCC testing; ThreadSanitizer required |

---

## Success Metrics

1. **Retrieval Quality** — Recall@1 / MRR on WebGL facts; target: no regression from baseline.
2. **Energy Efficiency** — Energy-per-query reduced 30%+ on low-priority queries (vs baseline full depth).
3. **Meaningfulness** — Meaning vectors correlate 0.85+ with manual domain semantics.
4. **Learning Speed** — Octave weighting converges in <50 feedback signals.
5. **Robustness** — All 5 failure modes handled; 0 crashes on adversarial inputs.
6. **Scale** — Fit 100k nodes in <2GB RAM; retrieve in <100ms per query.

---

## References

- **AGENTS.md** — Hard project rules; no Unicode decorative symbols; semiotic naming conventions.
- **CLAUDE.md** — Module layout; build order; stability invariants.
- **core/cone_engine.py** — Hyperbolic manifold primitives; tension/flow/energy operations.
- **core/semiotic_memory.py** — 4-layer memory (facts/summaries/working/session).
- **core/recursive.py** — Octave-based coarse-to-fine retrieval.
- **docs/agent-guide.md** — Agent API; retrieval interface.

---

## Next Steps

1. **Validate PRD Completeness** — Cross-reference rows against paper insights; ensure "every possible aspect" is covered.
2. **Prioritize Implementation** — Phase 1 (Week 1) focused on high-confidence, high-value rows.
3. **Establish Test Baselines** — WebGL facts + retrieval metrics as golden baseline.
4. **Begin Phase 1 Implementation** — Start with entropy-weighted-retrieval (3-line change + test).

---

**Status:** PLAN complete; ready for EXECUTE. Total PRD rows: 162 (112 existing + 50 paper-derived). All rows are actionable and witnessed.
