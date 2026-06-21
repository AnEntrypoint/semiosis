# Learning Loop Hypotheses: Complete Analysis & Roadmap

## Overview

The semiosis learning loop (record_outcome, consolidate, diagnose) currently feeds back usage signals and resolves redundancy via tension scanning. This package contains a comprehensive analysis of four hypothesis-driven enhancements that enable:

1. **Per-octave diagnostics** to detect performance asymmetry.
2. **Entropy divergence signals** to predict retrieval failures.
3. **Adaptive centroid recomputation** to mitigate incremental ingest drift.
4. **Memory layer auto-tuning** via papers methods (Bayesian Optimization, PBT, Hyperband).

## Documents

- **LEARNING_LOOP_ANALYSIS.md**: Strategic overview, hypothesis summary, validation plan.
  - Read this first for context and rationale.
  
- **LEARNING_LOOP_HYPOTHESES.md**: Detailed hypothesis specifications.
  - Hypothesis 1: Octave boundaries adapt to retrieval performance.
  - Hypothesis 2: Entropy divergence detects systematic failures.
  - Hypothesis 3: Centroid recomputation on information density shift.
  - Hypothesis 4: Memory layer auto-tuning via papers methods.
  - Includes witness points, strength/risk analysis, integration strategy.

- **IMPLEMENTATION_ROADMAP.md**: Phased execution plan with concrete code examples.
  - Phase 1 (1-2 days): Diagnostics—expand diagnose(), add per-octave recall.
  - Phase 2 (2-3 days): Coherence—implement centroid_coherence(), detect instability.
  - Phase 3 (3-4 days): Octave adaptation—recompute centroids on density shift.
  - Phase 4 (5-7 days): Memory auto-tuning—Bayesian Optimizer over memory configs.

- **IMPLEMENTATION_EXAMPLES.md**: Copy-paste-ready code snippets.
  - DiagnoseReport extensions, per-octave stats computation.
  - centroid_coherence() function and ranking stability check.
  - Centroid recomputation trigger logic.
  - MemoryTuner class and KnowledgeBase integration.
  - Test skeleton (copy-paste tests).

## Quick Start

### For Strategy Review
1. Read **LEARNING_LOOP_ANALYSIS.md** (15 min).
2. Skim **LEARNING_LOOP_HYPOTHESES.md** (30 min).
3. Review risk/strength matrix at end of HYPOTHESES.

### For Implementation Planning
1. Read **IMPLEMENTATION_ROADMAP.md** (30 min).
2. Copy code from **IMPLEMENTATION_EXAMPLES.md** as reference.
3. Start with Phase 1 (low risk, high information value).

### For Code Review
1. Check IMPLEMENTATION_EXAMPLES.md for correct signatures.
2. Verify all new settings have `bool` feature flags.
3. Ensure backward compatibility (all new fields have defaults).

## Key Findings

| Hypothesis | Signal | Action | Risk | Payoff |
|-----------|--------|--------|------|--------|
| 1. Octave adapt | Per-octave recall@k | Refit octaves or adjust dims | Medium | Early detection of over/under-clustering |
| 2. Entropy detect | Centroid-member distance, ranking drift | Trigger consolidate or re-digest | Low | Predict failures before silent collapse |
| 3. Centroid recompute | Member count growth, information density | Recompute apex/aperture, re-close transitivity | Medium | Mitigate incremental ingest drift |
| 4. Memory auto-tune | Recall@k under different memory configs | Bayesian Optimization | High | 5-15% recall improvement |

## Implementation Sequence

**Phase 1 (NOW)**: Expand `diagnose()` to compute per-octave stats and coherence. No behavior change.

**Phase 2 (NEXT)**: Implement `centroid_coherence()` metric. Enable agents to detect instability.

**Phase 3 (AFTER)**: Add octave adaptation logic in `consolidate()`. Gate behind feature flag.

**Phase 4 (FINAL)**: Implement `MemoryTuner` with Bayesian Optimization. Requires labeled queries.

## Files to Read (Current Codebase)

- `core/agent_api.py`: KnowledgeBase, learning loop methods (record_outcome, consolidate, diagnose).
- `core/cone_engine.py`: HyperbolicConeEngine, tension scanning, dispel operations.
- `core/eval.py`: Retrieval quality harness (recall@k, MRR).
- `core/settings.py`: Configuration (EncoderSettings, ConeSettings, AgentSettings).
- `core/recursive.py`: Octave-descent retrieval.
- `core/semiotic_memory.py`: Memory layer (facts, summaries, working, session).

## Expected Outcomes

### Phase 1 Completion
- `diagnose()` returns per-octave aperture stats (min, mean, max, std).
- Flags surface high-redundancy octaves.
- Agents can read diagnostic output; no retrieval behavior changes.

### Phase 2 Completion
- `centroid_coherence()` computes member-to-apex distances.
- Coherence values correlate with empirical retrieval success.
- Ranking stability metric detects when top-k rotates.

### Phase 3 Completion
- Recomputation triggers when member count grows > 20%.
- Updated apertures are tighter; nodes better fit members.
- Recall@k stabilizes or improves post-recomputation.

### Phase 4 Completion
- `MemoryTuner` converges to best config in < 20 trials.
- Best config improves recall@k by ≥ 5% vs. baseline.
- Tuner state persists across sessions via save/load.

## References

**Cone Geometry**:
- Ganea et al. (2018): "Hyperbolic Entailment Cones for Learning Hierarchies" (ICML).

**Hyperparameter Optimization**:
- Jaderberg et al. (2017): "Population Based Training of Neural Networks" (ICML).
- Li et al. (2018): "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (JMLR).
- Shahriari et al. (2015): "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (IEEE).
- Hansen & Ostermeier (2001): "Completely Derandomized Self-Adaptation in Evolution Strategies" (CMA-ES).

## Questions & Feedback

- **Should Hypothesis X be prioritized differently?** See risk/strength matrix in HYPOTHESES.
- **Can Phase Y be shortened?** ROADMAP includes "fast path" notes per phase.
- **What if labeled data is unavailable?** Phase 4 requires `set_labeled_queries()`; can be omitted entirely.

## Next Steps

1. Review LEARNING_LOOP_ANALYSIS.md.
2. Approve hypothesis approach.
3. Start Phase 1 implementation (2-3 PRs: diagnose, recall_per_octave, tests).
4. A/B test Phase 1 output against baseline.
5. Proceed to Phase 2 if Phase 1 shows low risk/high signal.

---

**Last Updated**: 2026-06-21  
**Status**: Analysis complete; implementation pending approval.

