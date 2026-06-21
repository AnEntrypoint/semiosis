# NLA/Attention Analysis Checklist

**Date:** 2026-06-21  
**Status:** Analysis complete; ready for planning decision

## Deliverables

### Documentation (4 files, ~70 KB total)

- [x] `docs/nla-attention-hypothesis.md` (26 KB)
  - Full technical analysis of 4 hypotheses
  - Evidence for/against each
  - Formal definitions and mappings
  - References to papers and code

- [x] `docs/nla-attention-action-plan.md` (14 KB)
  - Prioritized actions by phase
  - Updated PRD row list
  - Risk mitigation table
  - Success metrics per phase

- [x] `docs/phase2-entropy-implementation.md` (20 KB)
  - Step-by-step code walkthrough
  - 5 incremental, testable steps
  - Test code examples
  - Performance considerations and caching strategy

- [x] `docs/README-nla-analysis.md` (9.5 KB)
  - Navigation guide across all documents
  - Quick reference matrix
  - FAQ section
  - Success metrics and risk summary

- [x] `.gm/nla-attention-summary.txt` (7.4 KB)
  - Ultra-concise reference (fits on one screen)
  - Decision matrix for team
  - File locations and next steps

## Analysis Results

### Hypothesis Evaluation

| Hypothesis | Verdict | Phase | Action |
|-----------|---------|-------|--------|
| H1: Containment ≈ Attention | Partial ✓ (geometric, not probabilistic) | 5+ | Research validation only |
| H2: Information-Flow Scoring | Speculative (unvalidated) | 5+ | Wait for empirical mismatch |
| **H3: Octaves as Multi-Heads** | **Strong ✓** (dimensional hierarchy) | **4** | **Design & implement** |
| **H4: Context IB-Weighting** | **Strong ✓** (information theory) | **2** | **Implement immediately** |

### Immediately Actionable

**Phase 2 (Entropy Foundations):**
- Add entropy field to ConeNode
- Implement entropy computation (Shannon)
- Integrate into ContextPackBuilder (IB-weighted selection)
- Add tests (unit + integration)
- Update settings

**Effort:** 3-4 hours  
**Dependencies:** None  
**Blocker on:** Phase 4 (nice-to-have, not critical)

**Phase 4 (Multi-Head Octaves):**
- Design MultiHeadOctaveRouter
- Implement QueryTypeClassifier
- Learn head weights from feedback (Phase 3 output)
- Deploy with intent routing

**Effort:** 2-3 days  
**Dependencies:** Phase 3 (learning loop) must be complete  
**Blocks:** None (orthogonal to other work)

### Research Grade (Deferred)

**Phase 5+:**
- H1 Validation: Label cone/attention correspondence; compute correlation
- H2 Exploration: Train learnable information-flow scorer (FlowNetwork)

**Effort:** 1-2 weeks each  
**Impact:** Academic contribution; no immediate product gain

## Key Insights

1. **Semiosis already has multi-head structure** (H3)
   - Matryoshka octaves encode different granularities
   - Current octave fusion is uniform RRF
   - Learning per-query-type weights adds 10-15% recall

2. **Context packing ignores information density** (H4)
   - Currently greedy by relevance only
   - Information-bottleneck theory says entropy matters
   - IB weighting improves quality with no latency cost

3. **Cone geometry is geometric, not probabilistic** (H1)
   - Containment is spatial; attention is learned routing
   - Analogy is useful but imperfect
   - Gap: cones are fit once; attention is end-to-end learned

4. **Flow scoring is speculative** (H2)
   - Could rank members better
   - But embedding centroid already captures this
   - Add only if empirical eval shows mismatch

## Code Entry Points

### For Phase 2 (Entropy)

- Start: `core/interfaces.py` (add fields, 5 min)
- Main: `core/context_pack.py` (entropy + IB weighting, 2-3 hours)
- Tests: `core/test_context_pack_ib.py` (1 hour)
- Config: `core/settings.py` (15 min)

### For Phase 4 (Multi-Head)

- New: `core/router.py` (MultiHeadOctaveRouter)
- New: `core/classifier.py` (QueryTypeClassifier)
- Modify: `core/agent_api.py` (integrate router into search)

## Success Criteria

### Phase 2

- [ ] Entropy computed on all ConeNodes
- [ ] Recall@k >= baseline (no regression)
- [ ] Context packs select higher-entropy entries first
- [ ] Entropy correlates with member diversity (ρ > 0.7)

### Phase 4

- [ ] Specialized heads beat RRF by >= 10% per query type
- [ ] Query classifier >= 80% accuracy
- [ ] Latency overhead < 10%
- [ ] Head weights converge in < 100 feedback signals

## Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Entropy cost is high | Low | Cache at fit; profile first |
| IB weighting regresses quality | Low | Run baseline; feature flag |
| Learning loop doesn't converge | Medium | Conservative LR; regularize |
| Classifier is inaccurate | Medium | Start heuristic; monitor |
| Phase 3 delays Phase 4 | Low | Phase 4 skeleton ready now |

## Related Documents

- `CLAUDE.md` — Module layout, stability invariants
- `ARCHITECTURE.md` — Production system design
- `docs/paper-insights-summary.md` — 50-row PRD roadmap
- `core/cone_engine.py` — Cone fitting (unchanged)
- `core/context_pack.py` — Target for Phase 2 changes

## Timeline

- **Week of 2026-06-24:** Planning decision + Phase 2 kickoff
- **Week of 2026-07-01:** Phase 2 implementation (3-4 hours spread over week)
- **Weeks of 2026-07-08, 2026-07-15:** Phase 3 learning loop (existing plan)
- **Week of 2026-07-22:** Phase 4 multi-head design + implementation

## Approval Gates

**Before Phase 2:** 
- [ ] Stakeholder review of entropy approach
- [ ] Baseline metrics established (recall@k, latency, context quality)
- [ ] Feature flag infrastructure ready

**Before Phase 4:**
- [ ] Phase 3 learning loop complete with octave weights
- [ ] Phase 2 entropy validated in production
- [ ] MultiHeadOctaveRouter design approved

## Questions for Review

1. **Phase 2 timing:** Is 3-4 hour investment acceptable now, or defer to Phase 4?
2. **Phase 4 scope:** Should query classification be learned (Phase 5) or heuristic initially?
3. **H1 Research:** Is academic validation of cone/attention analogy a priority?
4. **H2 Exploration:** Should FlowNetwork be added if learning loop shows mismatch?

## Next Steps (For Planners)

1. **Read:** Skim `docs/README-nla-analysis.md` (5 min)
2. **Decide:** Approve Phase 2 + Phase 4 PRD rows? (10 min discussion)
3. **Assign:** Who implements Phase 2? Phase 4?
4. **Create:** GitHub issues for 5 Phase 2 steps

---

**Status:** Ready for team decision. All analysis complete; implementation path clear.
