---
title: Adversarial Robustness PRD Rows - New Discoveries
date: 2026-06-21
focus: 8 new PRD rows derived from adversarial-robustness-queries analysis
---

# Adversarial Robustness PRD Rows (New Discoveries)

Based on detailed analysis of arxiv 2512.24601 (RLM) + information-theoretic perspective, these 8 rows extend and ground the `adversarial-robustness-queries` row (already completed) with concrete, phased implementations.

---

## Row 1: adversarial-detection-embedding-collapse

**ID:** `adversarial-detection-embedding-collapse`

**Title:** Detect embedding collapse attacks via norm anomaly + member entropy

**Description:** 
Embedding collapse occurs when query embedding aligns with cluster centroid, losing discrimination. Agent should detect this by monitoring:
1. `norm_ratio = centroid_norm / query_norm` (should be < 0.85 for specific queries)
2. `member_entropy = shannon_entropy(distances_of_members_from_centroid)` (should be > 0.3 * baseline)
3. Cross-check: if norm_ratio high AND member_entropy low AND reported_confidence high => **collapse attack**

**Implementation:** 
- Add to `agent_api.py KnowledgeBase.search()`: compute norm_ratio and member_entropy before retrieval
- Return `collapse_risk_score` in SearchHit metadata
- If collapse_risk > 0.7: degrade confidence to 0.5, use coarser octave
- Test on WebGL corpus: benign queries score < 0.3; hand-crafted attacks score > 0.7

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.1 (embedding norm anomaly metric)
- `docs/adversarial-pattern-recognition-guide.md` Part 1, Pattern 1 (null-distinguisher)
- RLM paper Section 4, Observation 3 (confidence degrades with complexity)

**Effort:** Low (2-3 days; uses existing cone_engine entropy calculation)

**Priority:** High (first attack vector to implement)

---

## Row 2: adversarial-detection-semantic-poisoning

**ID:** `adversarial-detection-semantic-poisoning`

**Title:** Detect semantic poisoning attacks via contradiction entropy

**Description:**
Semantic poisoning injects contradictory facts into retrieved set. Agent should detect this by:
1. Computing meaning vectors for all retrieved facts
2. Measuring pairwise agreement between meaning vectors
3. Computing `contradiction_entropy = shannon_entropy(agreement_scores)`
4. Comparing to baseline: if entropy > 1.5 * baseline => **poisoning attack**

**Implementation:**
- Add to `agent_api.py KnowledgeBase.search()`: compute meaning_vector for each fact
- Compute pairwise agreement (cosine similarity of meaning vectors)
- Return `semantic_contradiction_score` in SearchHit metadata
- If contradiction > 0.4 OR entropy > 1.5 * baseline: reduce confidence, ask user to clarify goal priority
- Test: benign queries score < 0.4; poisoning attacks score > 0.7

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.1 (semantic contradiction entropy metric)
- `docs/adversarial-pattern-recognition-guide.md` Part 1, Pattern 2 (contradiction maximizer)
- RLM paper Appendix A (negative results: non-deterministic outputs, syntax errors reveal system instability)

**Effort:** Low (2-3 days; requires meaning_vector computation, already available in ConeNode)

**Priority:** High (second attack vector; directly observable in retrieval results)

---

## Row 3: adversarial-detection-circular-containment

**ID:** `adversarial-detection-circular-containment`

**Title:** Detect circular containment (self-referential containment cycles)

**Description:**
Circular queries exploit the containment graph to create self-loops: query -> parent -> ... -> query. Agent should detect via:
1. DFS on containment_graph starting from query node
2. Track visited nodes; if node visited twice => cycle detected
3. Limit search depth to prevent detection infinite loop
4. Return `has_cycle` and `cycle_length` in metadata

**Implementation:**
- Add to `core/recursive.py RecursiveAnswerEngine.descend()`: cycle detection before containment traversal
- Use standard DFS with visited set; max_depth=5 to prevent infinite detection
- If cycle detected: stop traversal, mark result as incomplete, suggest clarification
- Test: acyclic containment chains pass; hand-crafted cycles are detected

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.1 (transitive closure cycle detection)
- `docs/adversarial-pattern-recognition-guide.md` Part 1, Pattern 3 (circular reference)
- RLM paper Section 2, Algorithm 1 (state management must terminate; state.Final must be set)

**Effort:** Very Low (1 day; standard graph algorithm, < 20 lines of code)

**Priority:** High (simple to implement; high impact)

---

## Row 4: adversarial-detection-boundary-ambiguity

**ID:** `adversarial-detection-boundary-ambiguity`

**Title:** Detect boundary ambiguity attacks via octave disagreement

**Description:**
Boundary queries are equally contained in multiple clusters at different octaves. Agent should detect via:
1. Retrieve at all octaves (64, 32, 16, 8)
2. For each octave, find max containment score (best cluster match)
3. Compute `octave_disagreement = entropy(scores_across_octaves)`
4. If disagreement > 1.3 * baseline AND max_score < 0.7 for any octave => **boundary attack**

**Implementation:**
- Modify `core/recursive.py RecursiveAnswerEngine.descend()`: return scores_per_octave not just best match
- In `agent_api.py search()`: compute octave_disagreement_score
- If boundary_risk > 0.7: widen apertures (use coarser clusters), ask user which interpretation
- Test: benign queries show high agreement across octaves; boundary queries show entropy spike

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.1 (octave disagreement signal)
- `docs/adversarial-pattern-recognition-guide.md` Part 1, Pattern 4 (boundary dweller)
- RLM paper Section 5 (final-answer detection is brittle; same issue on octave boundaries)

**Effort:** Low (2 days; requires returning multi-octave scores, aggregating them)

**Priority:** High (reveals ambiguity, often dismissed as "query clarity" issue)

---

## Row 5: adversarial-detection-confidence-miscalibration

**ID:** `adversarial-detection-confidence-miscalibration`

**Title:** Detect confidence miscalibration via expected-vs-reported divergence

**Description:**
Miscalibration attacks report high confidence despite low information density. Agent should:
1. Compute `expected_confidence = f(query_complexity, context_richness, octave_agreement)`
2. Compute `reported_confidence` from retrieval scores
3. If `|reported - expected| > 0.2` => **miscalibration risk**
4. Validate against test set: if accuracy_gap > 0.3 => confirm attack

**Implementation:**
- Add to `core/recursive.py`: compute_expected_confidence(query, octaves) function
- Learn calibration curve from training queries (complexity, context, agreement -> accuracy)
- In `agent_api.py search()`: compute confidence_miscalibration_score
- If miscalibration > 0.2: adjust reported_confidence to expected_confidence
- Track confidence-accuracy on validation set; flag divergence for offline analysis
- Test: benign queries calibrated (gap < 0.1); miscalibration attacks detected (gap > 0.3)

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.1 (confidence accuracy gap metric)
- `docs/adversarial-pattern-recognition-guide.md` Part 1, Pattern 5 (false confidence trap)
- RLM paper Section 4, Observation 3 (confidence must degrade with complexity)
- rlm-agent-reasoning-extraction.md: "Honest uncertainty calibration lets downstream systems adjust"

**Effort:** Medium (3-4 days; requires learning calibration curve, confidence tracking)

**Priority:** High (confidence miscalibration has downstream impact on system reliability)

---

## Row 6: adversarial-risk-score-aggregation

**ID:** `adversarial-risk-score-aggregation`

**Title:** Aggregate five detection scores into single adversarial_risk_score [0,1]

**Description:**
Combine all five attack vectors into a weighted score:
- `risk = 0.20 * collapse_risk + 0.25 * semantic_entropy + 0.20 * has_cycle + 0.20 * disagreement + 0.15 * confidence_gap`
- Implement 4-level degradation (Level 1-4 confidence reduction, aperture widening, user clarification)
- Implement 4-level logging (low/medium/high/critical incidents)

**Implementation:**
- Add to `agent_api.py`: compute_adversarial_risk_score(query, retrieval_result)
- In `search()` main loop: compute risk_score after retrieval
- If risk > 0.3: log incident + trigger degradation handler
- If risk > 0.5: ask user clarification
- If risk > 0.7: reduce confidence to < 0.5, use coarser octave
- If risk > 0.9: reject or escalate to human
- Test: 100 benign queries (avg risk < 0.25); 50 adversarial queries (avg risk > 0.75)

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 2.2 (aggregate risk score formula)
- `docs/adversarial-robustness-strategy.md` Section 3.1 (degradation strategies, 4 levels)
- All five attack patterns (Patterns 1-5 grounded in Rows 1-5)

**Effort:** Low (1 day; pure aggregation logic)

**Priority:** Critical (necessary to tie all detections together)

---

## Row 7: adversarial-incident-logging-analytics

**ID:** `adversarial-incident-logging-analytics`

**Title:** Log all adversarial incidents for offline analysis and threshold tuning

**Description:**
Every suspicious query (risk_score > 0.3) is logged with:
- `timestamp, user, query, risk_score, detection_vector (collapse/contradiction/cycle/disagreement/miscalibration)`
- `action_taken (degrade/clarify/reject/escalate)`
- `outcome (user_accepted / user_clarified / system_corrected / human_reviewed)`

Enable offline analysis to:
1. Identify new attack patterns (false positives, emerging adversarial strategies)
2. Retune thresholds based on false-positive rate and impact
3. Learn user clarification acceptance rates
4. Validate that degradation prevents downstream errors

**Implementation:**
- Add to `agent_api.py`: AdversarialIncident dataclass
- In `search()`: log all incidents with risk_score > 0.3
- Persist to `.gm/adversarial-incidents.jsonl` (one JSON per line)
- Create `docs/adversarial-analysis-guide.md` with analysis queries
- Weekly report: false-positive rate, emerging patterns, threshold recommendations
- Test: log 100+ incidents; analyze to validate detection effectiveness

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 3.2 (logging and feedback loop)
- RLM paper Section 5 (trajectory analysis enables debugging failures)

**Effort:** Low (1 day; logging infrastructure exists in .gm/)

**Priority:** Medium (not blocking functionality, but critical for long-term robustness)

---

## Row 8: adversarial-confidence-calibration-loop

**ID:** `adversarial-confidence-calibration-loop`

**Title:** Continuously recalibrate confidence-accuracy relationship from validation outcomes

**Description:**
Machine the confidence calibration to prevent miscalibration attacks:
1. Maintain validation set of queries with known-correct answers
2. Periodically test: query -> agent_result -> compare_to_known_answer
3. Update calibration curve: `expected_confidence = f(complexity, context, agreement)`
4. If actual_accuracy diverges from reported_confidence, flag in logs
5. Re-tune confidence computation to match observed accuracy

**Implementation:**
- Add to `core/recursive.py`: ConfidenceCalibrator class
- Maintain golden validation set (WebGL facts with high confidence)
- Weekly job: test on validation set, compute actual accuracy per complexity/context/agreement bucket
- Update calibration curve via regression or piecewise-linear fit
- In `search()`: use current calibration curve to compute expected_confidence
- Alert: if recalibration changes thresholds > 5%, investigate for new attack pattern
- Test: confidence-accuracy correlation improves from 0.8 to 0.95

**Witness Path:**
- `docs/adversarial-robustness-strategy.md` Section 4.5 (confidence miscalibration detection)
- `docs/adversarial-pattern-recognition-guide.md` Part 2 (information-theoretic intuition)
- RLM paper Section 4, Observation 5 (hints improve confidence 38.7% -> 65.6%; calibration is learnable)

**Effort:** Medium (2-3 days; regression + validation infrastructure)

**Priority:** Medium (high impact for robustness; not blocking initial deployment)

---

## Summary Table

| ID | Title | Attack Vector | Effort | Priority | Status |
|---|---|---|---|---|---|
| adversarial-detection-embedding-collapse | Detect norm anomalies | Embedding Collapse | Low | High | New |
| adversarial-detection-semantic-poisoning | Detect contradiction entropy | Semantic Poisoning | Low | High | New |
| adversarial-detection-circular-containment | Detect cycles in containment | Hierarchical Confusion | Very Low | High | New |
| adversarial-detection-boundary-ambiguity | Detect octave disagreement | Boundary Exploitation | Low | High | New |
| adversarial-detection-confidence-miscalibration | Detect confidence divergence | Miscalibration | Medium | High | New |
| adversarial-risk-score-aggregation | Aggregate all detections | (Integration) | Low | Critical | New |
| adversarial-incident-logging-analytics | Log & analyze incidents | (Ops) | Low | Medium | New |
| adversarial-confidence-calibration-loop | Retune confidence model | (Maintenance) | Medium | Medium | New |

---

## Implementation Roadmap

### Phase 1: Immediate (Week 1)
1. **adversarial-detection-embedding-collapse** — 2 days
2. **adversarial-detection-semantic-poisoning** — 2 days
3. **adversarial-detection-circular-containment** — 1 day

**Exit Criteria:** Three detectors working; tests pass on benign + adversarial sets.

### Phase 2: Integration (Week 2)
1. **adversarial-risk-score-aggregation** — 1 day
2. **adversarial-detection-boundary-ambiguity** — 2 days
3. **adversarial-detection-confidence-miscalibration** — 3 days

**Exit Criteria:** All five detectors integrated; aggregate risk_score working.

### Phase 3: Operations (Week 3)
1. **adversarial-incident-logging-analytics** — 1 day
2. **adversarial-confidence-calibration-loop** — 2 days
3. Validation & threshold tuning — 2 days

**Exit Criteria:** Logging operational; calibration loop learning from data; false-positive rate < 5%.

---

## Testing Strategy

### Unit Tests (Per Row)
Each detector has unit tests:
- `test_detect_embedding_collapse()` — norm_ratio, member_entropy calculations
- `test_detect_semantic_poisoning()` — contradiction_entropy, agreement_scores
- `test_detect_circular_containment()` — cycle detection on graphs with/without cycles
- `test_detect_boundary_ambiguity()` — octave disagreement entropy
- `test_detect_confidence_miscalibration()` — expected vs reported confidence

### Integration Tests
- `test_adversarial_risk_aggregation()` — all five vectors combined correctly
- `test_degradation_strategies()` — correct confidence reduction at each level
- `test_incident_logging()` — incidents logged correctly, parseable

### Adversarial Test Suite
Hand-crafted adversarial queries targeting each vector:
- 10 embedding-collapse queries (vague, norm-collapsed)
- 10 semantic-poisoning queries (contradictory goals)
- 10 circular queries (self-referential containment)
- 10 boundary queries (ambiguous, octave-disagreement)
- 10 miscalibration queries (simple but wrong-domain)

Expected: all 50 adversarial queries detected with risk_score > 0.7.

### Benign Test Suite
100+ benign WebGL queries from golden corpus:
Expected: all benign queries score < 0.3, pass through normally.

---

## Success Metrics

1. **Detection Rate:** >= 90% of adversarial queries detected (risk_score > 0.7)
2. **False Positive Rate:** <= 5% of benign queries flagged (risk_score > 0.3)
3. **Degradation Effectiveness:** When adversarial detected, downstream errors reduced 50%+
4. **Calibration:** Confidence-accuracy correlation >= 0.9
5. **User Experience:** Clarification requests help users refine ambiguous queries; satisfaction >= 85%

---

## References

**Paper Sources:**
- RLM (arxiv 2512.24601): Section 4 (confidence), Appendix A (negative results)
- NLA: Information-bottleneck principle, robustness patterns

**Semiosis Code:**
- `core/cone_engine.py`: `contains()`, `_member_entropy()`
- `core/agent_api.py`: SearchHit, search pipeline
- `core/recursive.py`: RecursiveAnswerEngine

**Documentation:**
- `docs/adversarial-robustness-strategy.md` — full formal treatment
- `docs/adversarial-pattern-recognition-guide.md` — agent guidance + decision trees

