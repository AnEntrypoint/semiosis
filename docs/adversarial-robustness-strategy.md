---
title: Adversarial Robustness Strategy - Information-Theoretic Perspective
date: 2026-06-21
focus: Map RLM/NLA paper concepts to adversarial attack patterns; propose detection strategies via information theory
---

# Adversarial Robustness Strategy: Paper-Derived Detection Patterns

## Executive Summary

Adversarial attacks on hierarchical knowledge systems exploit:
1. **Embedding Collapse** — queries that cause query embedding norm to match corpus centroid (no distinction)
2. **Semantic Poisoning** — contradictory claims that maximize entropy in retrieved clusters
3. **Hierarchical Confusion** — circular queries that create self-loops in containment graph
4. **Boundary Exploitation** — queries on cluster boundaries that are ambiguous across octaves
5. **Confidence Manipulation** — queries designed to appear high-confidence while being incorrect

The information-theoretic view from RLM/NLA papers reveals: **adversarial success is detectable as anomalous information density and abnormal confidence-entropy relationships**.

This document formalizes detection strategies using (1) entropy thresholds, (2) norm-based anomaly detection, (3) transitive-closure cycle checking, (4) confidence-accuracy miscalibration metrics, and (5) octave-disagreement signals.

---

## 1. PAPER-DERIVED ADVERSARIAL PATTERNS

### 1.1 Embedding Collapse (Information Loss Attack)

**Paper Evidence:**
- RLM Section 4: Confidence degrades with task complexity; agents detect degradation via metadata
- NLA papers: Information-bottleneck principle — information loss is detectable via entropy
- cone_engine.py `_member_entropy()`: Shannon entropy of member distances from centroid

**Attack Mechanism:**
An adversary constructs a query `q_adv` such that:
- Embedding norm `||q_adv|| ≈ ||centroid_cluster||` (query aligns with cluster center)
- Query appears to match cluster well, but is semantically unrelated
- Example: WebGL query "GPU optimization" designed to match rendering cluster even though asking about ML

**Information-Theoretic Signal:**
- Normal query: embedding entropy = high (discriminative across members)
- Attack query: embedding entropy ≈ 0 (aligns with centroid, loses member diversity)
- Metric: `collapse_risk = entropy(query_embedding_vs_members) / baseline_entropy`
- **Detection Threshold:** collapse_risk < 0.3 => flag as suspicious

**RLM Connection:**
RLM agents recognize this via metadata (Section 5): "output truncation, incomplete states, non-deterministic execution" reveal uncertainty. Semiosis equivalent: when query embedding norm exactly matches centroid norm, entropy collapses to zero — metadata signal.

---

### 1.2 Semantic Poisoning (Contradiction Attack)

**Paper Evidence:**
- RLM Appendix A: Negative results document "syntax errors and non-deterministic outputs"
- NLA: Information density anomalies reveal structural breaks
- cone_engine.py `_information_loss()`: Compute information loss in parent-child pair

**Attack Mechanism:**
An adversary constructs a query with contradictory semantic intent:
- Query: "How to speed up rendering AND reduce GPU memory AND minimize power consumption while maximizing visual quality?"
- Semantically: these goals conflict (speed vs quality, memory vs power)
- Adversary expects conflicting retrieved facts to be returned simultaneously

**Information-Theoretic Signal:**
- Normal query: retrieved facts have low-entropy agreement on intent (coherent)
- Attack query: retrieved facts have high entropy — cluster disagreement on which goal to optimize
- Metric: `semantic_entropy = entropy(meaning_vectors_of_retrieved_facts)`
- Metric: `contradiction_score = number_of_mutually_exclusive_facts_retrieved / total_facts`
- **Detection Threshold:** semantic_entropy > octave_baseline * 2.0 OR contradiction_score > 0.5 => flag

**RLM Connection:**
RLM agents detect syntax errors (Appendix A) — malformed queries cause execution errors. Semiosis equivalent: contradictory queries cause information explosion in retrieval, similar to how unstructured input causes RLM syntax errors.

---

### 1.3 Hierarchical Confusion (Circular Query Attack)

**Paper Evidence:**
- RLM Section 2: "Recursive sub-calling and state management" require acyclic chains
- cone_engine.py `contains()`: Transitive containment via manifold geometry
- interfaces.py: Missing cycle-detection in Query Protocol

**Attack Mechanism:**
An adversary constructs a query that creates a self-loop in the containment graph:
- Query at octave=64: "What is the relationship between X and Y?"
- Semiosis retrieves parent octave=32: "X and Y are related through Q"
- Query then asks: "What is Q?" — which re-retrieves the same parent
- Creates circular dependency: Q -> X/Y -> Q

**Information-Theoretic Signal:**
- Normal query: containment chain forms a DAG (directed acyclic graph)
- Attack query: containment chain contains cycle
- Metric: `transitive_closure(containment_graph)` detects self-loops
- Metric: `cycle_depth = distance from query to self in containment graph`
- **Detection Threshold:** cycle_depth >= 1 => flag as circular

**RLM Connection:**
RLM state management (Section 2, Algorithm 1): "state <- InitREPL(...); while True do..." — agents must detect loop termination. Semiosis equivalent: containment graph must be acyclic. Detect circular queries via standard cycle-detection (DFS with visited set).

---

### 1.4 Boundary Exploitation (Ambiguity Attack)

**Paper Evidence:**
- RLM Section 5: "Trajectory analysis shows agents work with partial/ambiguous information"
- cone_engine.py: Aperture width determines cluster inclusion
- cone_engine.py `contains()`: Returns float (margin) — positive=inside, near-zero=boundary

**Attack Mechanism:**
An adversary constructs a query on a cluster boundary:
- Cluster A: "WebGL performance optimization"
- Cluster B: "JavaScript execution speed"
- Query q_adv: "Optimize JavaScript rendering" — equally contained in both clusters
- Retrieves facts from both; downstream agent uncertain which cluster owns the answer

**Information-Theoretic Signal:**
- Normal query: clear containment in one octave (high confidence)
- Attack query: marginal containment in multiple octaves (low confidence, high disagreement)
- Metric: `boundary_risk = number_of_octaves_with_containment_margin_in_[0.1, 0.5]`
- Metric: `octave_disagreement = entropy(scores_across_octaves) / baseline_octave_entropy`
- **Detection Threshold:** boundary_risk > 2 OR octave_disagreement > 1.5 => flag

**RLM Connection:**
RLM agents handle ambiguity (Section 5, Appendix A): "final-answer detection is brittle" — same issue applies to octave disagreement. Detect via entropy of octave scores.

---

### 1.5 Confidence Manipulation (Miscalibration Attack)

**Paper Evidence:**
- RLM Section 4, Observation 3: "Confidence degrades with task complexity"
- RLM Section 4, Observation 5: "Hints improve confidence 38.7% -> 65.6%"
- rlm-agent-reasoning-extraction.md: "Honest uncertainty calibration lets downstream systems adjust"

**Attack Mechanism:**
An adversary constructs a query that appears high-confidence but is incorrect:
- Query: O(1) complexity (simple lookup) — agent assigns confidence 0.95
- But fact retrieved is from wrong cluster / context
- Downstream system auto-accepts high-confidence result, propagating error

**Information-Theoretic Signal:**
- Normal query: confidence = f(complexity, context_richness) and calibrated to accuracy
- Attack query: confidence high despite low information density or disagreement
- Metric: `confidence_accuracy_gap = reported_confidence - actual_accuracy_on_validation_set`
- Metric: `context_richness = entropy(exemplars) + entropy(similar_queries) + entropy(cluster_metadata)`
- Metric: `expected_confidence = f(query_complexity, context_richness)` (calibration curve)
- **Detection Threshold:** confidence_accuracy_gap > 0.3 OR confidence > expected_confidence + 0.2 => flag

**RLM Connection:**
RLM paper (Section 4) emphasizes honest confidence calibration. Semiosis agents should track confidence accuracy on validation set and detect when reported confidence diverges from observed accuracy.

---

## 2. INFORMATION-THEORETIC DETECTION FRAMEWORK

### 2.1 Core Metrics

#### Information Density Anomaly

```python
def compute_information_density(query_embedding, retrieved_nodes, octave):
    """
    Information density = entropy of meaning vectors across retrieved nodes.
    High density = coherent cluster (normal).
    Low density = collapsed / redundant results (anomaly).
    """
    meaning_vectors = [node.meaning_vector for node in retrieved_nodes]
    pairwise_distances = compute_pairwise_distances(meaning_vectors)
    entropy = shannon_entropy(normalize(pairwise_distances))
    baseline = octave.baseline_information_density  # Learned from history
    return entropy / baseline
```

#### Embedding Norm Anomaly

```python
def compute_embedding_norm_anomaly(query_embedding, centroid_embedding):
    """
    Embedding collapse risk: how close is query to centroid?
    Low risk: query is different from centroid.
    High risk: query aligns with centroid (no discrimination).
    """
    query_norm = np.linalg.norm(query_embedding)
    centroid_norm = np.linalg.norm(centroid_embedding)
    
    # Query should have higher norm than centroid (specificity > generality)
    norm_ratio = centroid_norm / query_norm if query_norm > 0 else float('inf')
    
    # If ratio > 0.9, query is not specific enough
    collapse_risk = 1.0 - min(norm_ratio, 1.0)
    return collapse_risk
```

#### Octave Disagreement Signal

```python
def compute_octave_disagreement(query, retrieval_scores_per_octave):
    """
    Octave disagreement: do different octaves agree on relevance?
    High agreement = normal (query clarity).
    Low agreement = boundary/ambiguous (anomaly).
    """
    scores_per_octave = [scores[query] for scores in retrieval_scores_per_octave]
    
    # Entropy across octave scores
    normalized_scores = normalize(scores_per_octave)
    disagreement = shannon_entropy(normalized_scores)
    
    # Baseline: learned from benign queries
    baseline = octave_agreement_baseline
    return disagreement / baseline
```

#### Semantic Contradiction Entropy

```python
def compute_semantic_contradiction(retrieved_facts):
    """
    Contradiction entropy: how mutually exclusive are the facts?
    Low entropy = coherent narrative (normal).
    High entropy = contradictory/conflicting facts (anomaly).
    """
    # For each fact, compute "agreement" with others
    fact_pairs = [(f1, f2) for f1, f2 in combinations(facts, 2)]
    
    contradiction_scores = []
    for f1, f2 in fact_pairs:
        # Does f1 contradict f2? (binary or continuous score)
        contradiction = detect_semantic_opposition(f1, f2)
        contradiction_scores.append(contradiction)
    
    entropy = shannon_entropy(normalize(contradiction_scores))
    baseline = contradiction_entropy_baseline
    return entropy / baseline
```

#### Transitive Closure Cycle Detection

```python
def detect_circular_containment(query, containment_graph, max_depth=5):
    """
    Does the containment chain from query ever loop back to itself?
    Normal: DAG (no cycles).
    Anomaly: contains cycle => circular dependency.
    """
    visited = set()
    stack = [query]
    
    while stack and len(visited) <= max_depth:
        current = stack.pop()
        if current in visited:
            return True  # Cycle detected
        visited.add(current)
        
        # Follow containment edges
        for parent in containment_graph.get_parents(current):
            stack.append(parent)
    
    return False
```

### 2.2 Aggregate Adversarial Score

```python
def compute_adversarial_risk_score(query, retrieval_result, octaves):
    """
    Aggregate risk score from all five attack vectors.
    Combines normalized scores into single risk metric [0, 1].
    """
    
    # 1. Embedding collapse
    collapse_risk = compute_embedding_norm_anomaly(
        query.embedding, octaves[0].centroid
    )
    
    # 2. Semantic poisoning
    semantic_entropy = compute_semantic_contradiction(
        retrieval_result.facts
    )
    
    # 3. Circular query
    has_cycle = detect_circular_containment(query, containment_graph)
    
    # 4. Boundary ambiguity
    disagreement = compute_octave_disagreement(
        query, retrieval_result.scores_per_octave
    )
    
    # 5. Confidence miscalibration
    confidence_gap = (
        retrieval_result.reported_confidence - 
        expected_confidence_from_complexity(query)
    )
    
    # Weighted combination
    risk_score = (
        0.20 * collapse_risk +
        0.25 * semantic_entropy +
        0.20 * (1.0 if has_cycle else 0.0) +
        0.20 * disagreement +
        0.15 * max(0, confidence_gap)
    )
    
    return risk_score
```

---

## 3. PROPOSED ADVERSARIAL DETECTION ARCHITECTURE

### 3.1 Detection Pipeline

```
User Query
    |
    v
[Input Validation]  <- Detect malformed/injection attacks
    |
    v
[Embedding Phase]
    |
    +---> [Norm Anomaly Check] <- embedding_collapse_risk
    |
    v
[Retrieval Phase]
    |
    +---> [Octave Disagreement] <- boundary_exploitation_risk
    |
    v
[Result Assembly]
    |
    +---> [Semantic Contradiction Check] <- semantic_poisoning_risk
    |
    +---> [Circular Containment Check] <- hierarchical_confusion_risk
    |
    +---> [Confidence Calibration Check] <- miscalibration_risk
    |
    v
[Risk Aggregation] -> adversarial_risk_score
    |
    v
IF risk_score > threshold:
    DEGRADE confidence
    REQUEST clarification from user
    LOG suspicious pattern
    FALLBACK to broader context
ELSE:
    RETURN normal result
```

### 3.2 Degradation Strategies on Detection

**Level 1 (Low Risk, score 0.3-0.5):**
- Reduce reported confidence by 20-30%
- Add disclaimer: "Some ambiguity detected in retrieved context"
- Continue normal retrieval

**Level 2 (Medium Risk, score 0.5-0.7):**
- Reduce confidence by 40-50%
- Widen apertures (use coarser octaves for broader context)
- Add warning: "Retrieved facts show some disagreement; interpretation may be ambiguous"
- Request user clarification: "Did you mean X or Y?"

**Level 3 (High Risk, score 0.7-0.9):**
- Reduce confidence to < 0.4 (low-confidence result)
- Fall back to depth=1 (coarsest octave only)
- Log suspicious pattern for offline analysis
- Return result with explicit caveat: "High uncertainty; recommend manual review"

**Level 4 (Critical Risk, score > 0.9):**
- Reject query or return "Unable to process — ambiguous or contradictory"
- Log incident and recommended threshold adjustments
- Escalate to human reviewer
- Offer fallback: "Would you like to refine your query?"

---

## 4. RLM PAPER GROUNDING

### 4.1 Embedding Collapse & RLM Section 4

**RLM Observation 3:** "Confidence degrades with task complexity"

**Semiosis Mapping:** Query complexity is revealed by embedding norm anomaly:
- O(1) queries: specific embedding, high norm difference from centroid => confidence 0.95
- O(n²) queries: diffuse embedding, low norm difference from centroid => confidence 0.5

**Detection:** When embedding norm approaches centroid norm, query appears O(n²) even if claimed O(1) => adversarial.

### 4.2 Semantic Poisoning & RLM Appendix A

**RLM Negative Results:** "Syntax errors and non-deterministic outputs"

**Semiosis Mapping:** Contradictory facts are like syntax errors in semantic space:
- RLM agents detect syntax via error output
- Semiosis agents detect contradiction via entropy of meaning vectors

**Detection:** When retrieved facts have high semantic entropy (conflicting), treat similarly to RLM's error detection.

### 4.3 Circular Queries & RLM Algorithm 1

**RLM Design:** State loop must terminate; agents must detect when state.Final is set

**Semiosis Mapping:** Containment chain must form DAG; agents must detect cycles

**Detection:** Standard graph cycle detection (DFS) on containment relationships.

### 4.4 Boundary Exploitation & RLM Section 5

**RLM Observation 5:** "Final-answer detection is brittle"

**Semiosis Mapping:** Octave boundary ambiguity is like RLM's final-answer ambiguity:
- RLM agents unsure when to stop => high error rates near decision boundary
- Semiosis agents unsure which octave to trust => high error rates at cluster boundaries

**Detection:** Entropy of scores across octaves signals boundary ambiguity.

### 4.5 Confidence Miscalibration & RLM Section 4

**RLM Key Finding:** "Extended reasoning with hints improves confidence 38.7% -> 65.6%"

**Semiosis Mapping:** Confidence should scale with context_richness and task_complexity:
- `expected_confidence = f(query_complexity, context_richness, octave_agreement)`
- When reported_confidence diverges from expected_confidence => miscalibration attack

**Detection:** Track confidence accuracy on validation set; flag divergence.

---

## 5. INFORMATION-THEORETIC INTUITION

### Why Information Theory Detects Adversarial Queries?

**Fundamental Principle:** Adversarial attacks sacrifice information coherence to exploit system weaknesses.

1. **Embedding Collapse:** Loss of discrimination in embedding space => entropy collapse in member distances
2. **Semantic Poisoning:** Contradictory claims => entropy spike in fact meanings
3. **Circular Queries:** Self-referential loops => detectable cycles in containment graph
4. **Boundary Exploitation:** Ambiguous intent => entropy in octave agreement scores
5. **Miscalibration:** False confidence => divergence between expected and observed accuracy

**All attacks leave information-theoretic signatures** because they violate the implicit assumptions of coherent, non-contradictory, and acyclic knowledge structures.

### Information Density as Defense

```
Normal System State:
    Query Entropy        HIGH (discriminative)
    Octave Agreement     HIGH (consistent)
    Fact Contradiction   LOW (coherent)
    Containment Cycles   ZERO (acyclic)
    Confidence Accuracy  CALIBRATED (high correlation)

Adversarial Attack State:
    Query Entropy        LOW (collapsed)
    Octave Agreement     LOW (boundary ambiguity)
    Fact Contradiction   HIGH (poisoning)
    Containment Cycles   YES (circular)
    Confidence Accuracy  UNCALIBRATED (divergence)
```

Detection exploits this by monitoring each dimension and flagging when any dimension becomes anomalous.

---

## 6. IMPLEMENTATION ROADMAP

### Phase 1: Metrics & Detection (Week 1)

1. Implement `compute_information_density()`
2. Implement `compute_embedding_norm_anomaly()`
3. Implement `compute_octave_disagreement()`
4. Implement `compute_semantic_contradiction()` (skeleton; requires NLP classifier)
5. Implement `detect_circular_containment()`
6. Tests: pass benign queries with risk_score < 0.3

### Phase 2: Aggregation & Thresholds (Week 2)

1. Implement `compute_adversarial_risk_score()` (weighted combination)
2. Learn thresholds from golden dataset (WebGL facts)
3. Implement confidence degradation logic (4-level fallback)
4. Tests: adversarial queries (hand-crafted) score > 0.7; benign queries score < 0.3

### Phase 3: Integration & Logging (Week 3)

1. Hook adversarial detector into `agent_api.py search()` pipeline
2. Add logging for suspicious patterns (for offline analysis)
3. Implement user-facing warnings/disclaimers
4. Tests: end-to-end on mixed adversarial + benign query set

### Phase 4: Refinement & Validation (Week 4)

1. Calibrate thresholds on larger dataset
2. Add confidence-accuracy tracking loop
3. Generate false-positive/false-negative analysis
4. Tests: 90%+ precision, <10% false-positive rate on benign

---

## 7. SUCCESS CRITERIA

1. **Detection Rate:** Adversarial queries score > 0.7 with >90% precision
2. **False Positive Rate:** Benign queries score < 0.3 with <5% false-positive rate
3. **Degradation Effectiveness:** When adversarial detected, system confidence reduced 40%+ and results marked with caveat
4. **User Experience:** Clarification requests help users refine ambiguous queries; satisfaction >85%
5. **Logging Quality:** Suspicious patterns logged enable offline analysis of new attack vectors

---

## 8. RESEARCH REFERENCES

**Paper:**
- Recursive Language Models (Zhang, Kraska, Khattab, arxiv 2512.24601)
  - Section 4: Confidence calibration via task complexity
  - Appendix A: Negative results and failure modes
  - Algorithm 1: State management and loop termination

**Information Theory:**
- Shannon Entropy: Quantify information density and agreement
- Information Bottleneck: Relevant compression preserves task-specific information
- Mutual Information: Octave agreement signals

**Semiosis Architecture:**
- core/cone_engine.py: `contains()` and `_member_entropy()`
- core/agent_api.py: Search result structure; confidence fields
- core/recursive.py: Octave traversal and depth selection

---

## 9. NEXT STEPS

1. Implement Phase 1 metrics (aim for end of week 1)
2. Create synthetic adversarial test set (hand-crafted attacks targeting each vector)
3. Validate detection on benign WebGL corpus
4. Integrate into agent_api.py search() before Week 2 merge
5. Track confidence-accuracy divergence over time to detect new attack patterns

