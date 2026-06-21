---
title: Adversarial Pattern Recognition Guide for Agents
date: 2026-06-21
focus: Concrete patterns agents should recognize; how information-theoretic view helps; detection workflows
---

# Adversarial Pattern Recognition: A Guide for Semiosis Agents

## Overview

This document translates adversarial attacks into **concrete, recognizable patterns** that agents can learn and detect. Rather than abstract theory, it provides pattern examples, decision trees, and information-theoretic explanations grounded in RLM/NLA papers.

---

## PART 1: Five Adversarial Attack Patterns & Recognition

### Pattern 1: The Null-Distinguisher Attack (Embedding Collapse)

**What it looks like:**
- User query: "Optimize rendering performance"
- System embedding for "Optimize rendering performance" has norm ≈ corpus centroid norm
- Retrieval returns facts from everywhere (no specificity)
- Downstream agent thinks: "confidence=0.95 because query is simple" but result is actually incoherent

**Why it works:**
- Agent trusts embedding norm as a proxy for query specificity
- Adversary constructs query that appears simple but semantically underconstrained

**Information-Theoretic Signal:**
```
NORMAL QUERY:
  query_embedding_norm = 1.5
  cluster_centroid_norm = 0.8
  norm_ratio = 0.53 (query specific; ratio < 0.8)
  => expected_specificity = HIGH
  
ATTACK QUERY:
  query_embedding_norm = 0.8
  cluster_centroid_norm = 0.8
  norm_ratio = 1.0 (query collapsed to centroid)
  => expected_specificity = LOW
  => BUT agent_reports_confidence = 0.95 (mismatch!)
  
DETECTION: norm_ratio > 0.85 AND confidence > 0.7 => FLAG
```

**Agent Recognition Checklist:**
- [ ] Query embedding norm is within 0.15 of centroid norm?
- [ ] Reported confidence is high (>0.7) despite low norm differentiation?
- [ ] Retrieved facts are scattered across multiple clusters?
- [ ] Meaning vectors of retrieved facts have low correlation?

**Degradation Action:**
- Reduce confidence to 0.5
- Retrieve from coarser octave (more context, less specificity)
- Report: "Query specificity unclear; retrieved facts diverse"

---

### Pattern 2: The Contradiction Maximizer (Semantic Poisoning)

**What it looks like:**
- User query: "Fast AND memory-efficient AND low-power AND high-quality rendering"
- These goals are mutually exclusive (speed vs quality, power vs performance)
- System retrieves facts supporting each goal separately
- When combined, facts form incoherent narrative

**Why it works:**
- Agent assumes retrieved facts are compatible (no mutual checking)
- Adversary crafts query that maximizes entropy in fact-space

**Information-Theoretic Signal:**
```
NORMAL QUERY:
  retrieved_facts = [fact_1: "use LOD", fact_2: "use occlusion culling"]
  
  semantic_vectors = [
    "reduce geometry_count" (both facts agree on this direction)
  ]
  
  entropy_of_agreement = LOW (facts align on meaning)
  contradiction_score = 0.1 (10% of facts contradict)
  
ATTACK QUERY:
  retrieved_facts = [
    fact_1: "maximize quality -> use high res textures",
    fact_2: "minimize power -> use low res textures",
    fact_3: "maximize speed -> batch draw calls",
    fact_4: "minimize memory -> unbatch draw calls"
  ]
  
  semantic_vectors = [
    "texture resolution" (contradictory advice),
    "draw call batching" (contradictory advice)
  ]
  
  entropy_of_agreement = HIGH (facts conflict on meaning)
  contradiction_score = 0.5 (50% of facts contradict)
  
DETECTION: contradiction_score > 0.4 OR entropy > baseline * 1.5 => FLAG
```

**Agent Recognition Checklist:**
- [ ] Query contains multiple conflicting optimization goals?
- [ ] Retrieved facts are about mutually exclusive strategies?
- [ ] Fact reasoning chains lead to opposite conclusions?
- [ ] Query uses AND to connect opposing demands?

**Degradation Action:**
- Ask user to clarify priority: "Is quality or speed more important?"
- Retrieve facts for only highest-priority goal
- Report: "Conflicting goals detected; prioritize one"

---

### Pattern 3: The Circular Reference (Hierarchical Confusion)

**What it looks like:**
- User query at octave=64: "What is query optimization?"
- System retrieves parent fact at octave=32: "Query optimization means applying heuristics from algorithm literature"
- User then asks: "What is algorithm literature optimization?"
- System retrieves parent fact: "Same as query optimization applied to algorithms"
- Loop detected: query -> fact -> new_query -> same_fact -> ...

**Why it works:**
- Agent does not track which facts were retrieved in prior steps
- Circular dependencies exploit the recursion mechanism itself

**Information-Theoretic Signal:**
```
NORMAL CONTAINMENT CHAIN:
  octave_64: "rendering performance"
  -> octave_32: "GPU utilization patterns"
  -> octave_16: "GPU architecture fundamentals"
  -> octave_8: "hardware constraints"
  
  DAG structure (no cycles); traversal terminates naturally

ATTACK CONTAINMENT CHAIN:
  octave_64: "What is X?"
  -> octave_32: "X is a type of Y"
  -> octave_16: "Y is a specialization of Z"
  -> octave_8: "Z is a type of X"
  
  CYCLE DETECTED: X -> Y -> Z -> X
  
DETECTION: DFS on containment_graph; if visit node twice => FLAG
```

**Agent Recognition Checklist:**
- [ ] Did this fact appear in an earlier retrieval step?
- [ ] Does the current fact reference concepts retrieved before?
- [ ] Is the containment chain length growing without convergence?
- [ ] Does following the "parent" link eventually point back to query?

**Degradation Action:**
- Stop retrieval; do not follow circular link
- Report: "Circular dependency detected; stopping traversal"
- Return partial result from non-circular path

---

### Pattern 4: The Boundary Dweller (Ambiguity Exploitation)

**What it looks like:**
- Query: "JavaScript rendering optimization"
- This is equally relevant to cluster A (JavaScript performance) and cluster B (WebGL rendering)
- Octave_64 says: "JS optimization" (contains in cluster A at margin 0.3)
- Octave_32 says: "WebGL optimization" (contains in cluster B at margin 0.35)
- Octave_16 says: "Hybrid JS/WebGL" (contains in both at margin 0.2 each)
- Agent unsure which cluster owns the query

**Why it works:**
- Agent assumes one cluster will dominate (high containment score)
- Boundary queries make multiple clusters equally plausible
- Adversary asks: which interpretation do you prefer?

**Information-Theoretic Signal:**
```
NORMAL QUERY (CLEAR CLUSTER):
  octave_64: score_in_cluster_A = 0.9, score_in_cluster_B = 0.1
  octave_32: score_in_cluster_A = 0.85, score_in_cluster_B = 0.15
  octave_16: score_in_cluster_A = 0.8, score_in_cluster_B = 0.2
  
  entropy_across_octaves(cluster_A) = 0.1 (high agreement)
  entropy_across_octaves(cluster_B) = 0.15 (high agreement)
  => CONSISTENT: cluster A dominates all octaves

ATTACK QUERY (BOUNDARY):
  octave_64: score_in_cluster_A = 0.5, score_in_cluster_B = 0.5
  octave_32: score_in_cluster_A = 0.3, score_in_cluster_B = 0.7
  octave_16: score_in_cluster_A = 0.6, score_in_cluster_B = 0.4
  
  entropy_across_octaves(cluster_A) = 0.5 (disagreement)
  entropy_across_octaves(cluster_B) = 0.5 (disagreement)
  => AMBIGUOUS: octaves cannot agree on best cluster
  
DETECTION: entropy > baseline * 1.3 OR max(scores) < 0.6 for any octave => FLAG
```

**Agent Recognition Checklist:**
- [ ] Different octaves prefer different clusters?
- [ ] No octave gives a high-confidence answer (all < 0.65)?
- [ ] Query is phrased at the intersection of multiple domains?
- [ ] Marginal containment scores (0.2-0.6) rather than clear inclusion/exclusion?

**Degradation Action:**
- Widen apertures (use coarser clusters to reduce ambiguity)
- Retrieve from consensus octave (where agreement is highest)
- Ask user: "Do you mean X (cluster A) or Y (cluster B)?"
- Report: "Query is ambiguous; multiple interpretations possible"

---

### Pattern 5: The False Confidence Trap (Miscalibration)

**What it looks like:**
- User query: "Combine deferred rendering with forward rendering in WebGL"
- Agent analyzes: O(1) complexity (a specific technique question)
- Agent reports: confidence = 0.95
- But facts retrieved are from wrong context (C++ graphics library, not WebGL)
- Agent never verified accuracy; just trusted complexity estimate

**Why it works:**
- Agent calibrates confidence based only on query complexity
- Adversary crafts a simple query about wrong domain (low complexity but high error risk)
- Confidence-accuracy relationship breaks down

**Information-Theoretic Signal:**
```
NORMAL QUERY:
  query_complexity = O(1) (simple lookup)
  context_richness = HIGH (exemplars available, prior queries similar)
  octave_agreement = HIGH (all octaves agree on answer)
  expected_confidence = 0.95 (based on calibration curve)
  actual_accuracy_on_validation = 0.93
  => CALIBRATED: confidence matches accuracy
  
ATTACK QUERY:
  query_complexity = O(1) (simple lookup - appears innocent)
  context_richness = LOW (no exemplars, domain unfamiliar)
  octave_agreement = MEDIUM (mixed answers from different octaves)
  expected_confidence = f(O(1), LOW, MEDIUM) = 0.65
  BUT agent_reports_confidence = 0.95 (based only on complexity!)
  actual_accuracy_on_validation = 0.35
  => MISCALIBRATED: confidence >> accuracy
  
DETECTION: |reported_confidence - expected_confidence| > 0.2 => FLAG
```

**Agent Recognition Checklist:**
- [ ] Confidence based only on query complexity, not context?
- [ ] Limited context (no exemplars, few prior queries of this type)?
- [ ] Disagreement between octaves (medium agreement, not high)?
- [ ] Retrieved facts from unexpected domains?

**Degradation Action:**
- Reduce confidence to match expected_confidence
- Add caveat: "Limited context for this query type"
- Retrieve from broader context (coarser octave)
- Suggest: "Would you like to provide examples of what you're looking for?"

---

## PART 2: Information-Theoretic View — Why These Patterns Work

### The Core Insight

Adversarial attacks **violate coherence assumptions** that make hierarchical systems efficient:

1. **Coherence Assumption:** Knowledge is organized so related facts cluster together (high entropy within cluster, low entropy across clusters)
2. **Acyclicity Assumption:** Containment relationships form a DAG (facts do not circularly reference each other)
3. **Calibration Assumption:** Agent confidence correlates with accuracy (confidence is well-trained)

Adversarial attacks target:
- **Collapse coherence** (Pattern 1: all facts collapse to centroid)
- **Introduce contradiction** (Pattern 2: contradictory facts in one cluster)
- **Break acyclicity** (Pattern 3: create cycles in containment)
- **Violate clarity** (Pattern 4: query at cluster boundary, no clear winner)
- **Miscalibrate confidence** (Pattern 5: confidence decoupled from accuracy)

### Information Density as a Guard

**Key Metric: Shannon Entropy**

In a normal system:
- Query embedding has high entropy relative to centroid (discriminative)
- Retrieved facts have low entropy (coherent cluster)
- Octave agreement has low entropy (consensus across scales)

In an adversarial system:
- Query embedding has low entropy relative to centroid (collapsed)
- Retrieved facts have high entropy (contradictory cluster)
- Octave agreement has high entropy (no consensus)

**Detection Strategy:** Monitor entropy across all dimensions. When entropy spikes in unexpected ways, the system is under attack.

### RLM Paper Connection

RLM agents detect similar anomalies via **metadata**:

```
RLM AGENT (Section 5, Appendix A):
  Uses metadata to detect failures:
    - stdout length (truncated = incomplete execution)
    - execution time (slow = inefficient decomposition)
    - error count (many = unstable algorithm)
  When metadata anomalous => adjust strategy (retry, fallback, simplify)

SEMIOSIS AGENT:
  Uses information-theoretic metadata to detect attacks:
    - embedding norm ratio (low = collapsed embedding)
    - fact entropy (high = contradictory retrieval)
    - octave disagreement (high = ambiguous query)
    - confidence-accuracy gap (high = miscalibrated)
  When metadata anomalous => adjust strategy (degrade, clarify, fallback)
```

The parallel is exact: **both systems detect failure via observable metadata, not just content**.

---

## PART 3: Agent Decision Trees

### Decision Tree 1: Embedding Collapse Detection

```
STEP 1: Compute Embedding Norms
  query_norm = ||embedding(user_query)||
  centroid_norm = ||embedding(cluster_centroid)||
  norm_ratio = centroid_norm / query_norm

STEP 2: Check Norm Ratio
  IF norm_ratio > 0.85:
    => RISK: Query may be collapsed
    => GOTO: STEP 3
  ELSE:
    => OK: Query is specific
    => RETURN: proceed_normal_retrieval()

STEP 3: Check Confidence
  reported_confidence = agent.compute_confidence(query)
  IF reported_confidence > 0.75:
    => DANGER: High confidence despite low specificity
    => GOTO: STEP 4
  ELSE:
    => OK: Confidence matches uncertainty
    => RETURN: proceed_normal_retrieval()

STEP 4: Verify with Member Entropy
  member_distances = [||member - centroid|| for member in cluster]
  member_entropy = shannon_entropy(normalize(member_distances))
  baseline_entropy = octave.learned_baseline
  IF member_entropy / baseline_entropy < 0.3:
    => CONFIRMED ATTACK: Embedding collapse
    => RETURN: degrade_confidence(0.5) + use_coarser_octave()
  ELSE:
    => FALSE ALARM: Entropy suggests coherent cluster
    => RETURN: proceed_normal_retrieval()
```

### Decision Tree 2: Semantic Contradiction Detection

```
STEP 1: Check Query Structure
  query_tokens = tokenize(user_query)
  conflicting_keywords = count_mutually_exclusive_pairs(query_tokens)
  
  IF conflicting_keywords >= 2:
    => RISK: Query may contain contradictions
    => GOTO: STEP 2
  ELSE:
    => OK: Query appears coherent
    => RETURN: proceed_normal_retrieval()

STEP 2: Retrieve Facts
  facts = retrieve_from_all_octaves(query)
  meaning_vectors = [fact.meaning_vector for fact in facts]

STEP 3: Compute Contradiction Entropy
  semantic_agreement = measure_pairwise_agreement(meaning_vectors)
  contradiction_entropy = shannon_entropy(semantic_agreement)
  baseline = octave.learned_contradiction_baseline
  
  IF contradiction_entropy > baseline * 1.5:
    => RISK: Retrieved facts are contradictory
    => GOTO: STEP 4
  ELSE:
    => OK: Facts are coherent
    => RETURN: return_normal_result(facts)

STEP 4: Check Contradiction Score
  contradiction_score = count_mutually_exclusive_facts(facts) / len(facts)
  IF contradiction_score > 0.4:
    => CONFIRMED ATTACK: Semantic poisoning
    => RETURN: ask_user_clarification() + reduce_confidence(0.4)
  ELSE:
    => FALSE ALARM: Not all facts are contradictory
    => RETURN: return_result_with_caveat("Some disagreement in retrieved facts")
```

### Decision Tree 3: Circular Containment Detection

```
STEP 1: Initialize Cycle Detection
  visited_nodes = set()
  current_node = user_query
  traversal_depth = 0
  max_depth = 5  (prevent infinite loops in detection itself)

STEP 2: Follow Containment Chain
  WHILE traversal_depth < max_depth:
    parent = containment_graph.get_parent(current_node)
    
    IF parent is None:
      => OK: Reached root
      => RETURN: traversal_is_acyclic()
    
    IF parent in visited_nodes:
      => CONFIRMED ATTACK: Circular containment
      => cycle_length = traversal_depth - visited_nodes[parent].depth
      => RETURN: degrade_confidence(0.3) + stop_traversal()
    
    visited_nodes.add(parent)
    current_node = parent
    traversal_depth += 1

STEP 3: Depth Limit Reached
  IF traversal_depth >= max_depth:
    => WARNING: Containment chain is very deep; risk of undetected cycle
    => RETURN: degrade_confidence(0.2) + warn_user("Unusually deep hierarchy")
```

### Decision Tree 4: Boundary Ambiguity Detection

```
STEP 1: Retrieve Across All Octaves
  octaves = [64, 32, 16, 8]
  scores = {}
  FOR octave in octaves:
    cluster_scores = retrieve(query, octave)
    scores[octave] = cluster_scores

STEP 2: Check for Clear Winner
  FOR octave in octaves:
    max_score = max(scores[octave])
    IF max_score > 0.7:
      => OK: This octave is confident
      => no_boundary_risk = True
      break
    ELSE:
      => RISK: This octave is uncertain
      => boundary_risk += 1

STEP 3: Compute Octave Disagreement
  IF boundary_risk >= 2:
    => RISK: Multiple octaves uncertain
    => scores_across_octaves = [max_score for scores in scores.values()]
    => disagreement_entropy = shannon_entropy(normalize(scores_across_octaves))
    => baseline = octave.learned_agreement_baseline
    
    IF disagreement_entropy > baseline * 1.3:
      => CONFIRMED ATTACK: Boundary ambiguity
      => RETURN: ask_user_clarification() + widen_apertures()
    ELSE:
      => FALSE ALARM: Disagreement within expected range
      => RETURN: return_best_guess() + reduce_confidence(0.2)
  ELSE:
    => OK: At least one octave is confident
    => RETURN: proceed_normal_retrieval()
```

### Decision Tree 5: Confidence Miscalibration Detection

```
STEP 1: Compute Expected Confidence
  query_complexity = estimate_complexity(query)
  context_richness = count_exemplars() + count_similar_prior_queries()
  octave_agreement = entropy(scores_across_octaves)
  
  expected_confidence = calibration_curve(
    query_complexity=query_complexity,
    context_richness=context_richness,
    octave_agreement=octave_agreement
  )

STEP 2: Retrieve and Score
  facts = retrieve(query)
  reported_confidence = agent.compute_confidence(facts)

STEP 3: Check for Miscalibration
  confidence_gap = abs(reported_confidence - expected_confidence)
  
  IF confidence_gap > 0.2:
    => RISK: Confidence diverges from expectation
    => GOTO: STEP 4
  ELSE:
    => OK: Confidence is calibrated
    => RETURN: return_normal_result(facts, reported_confidence)

STEP 4: Verify Accuracy (if validation set available)
  actual_accuracy = test_facts_against_validation_set(facts)
  accuracy_gap = abs(reported_confidence - actual_accuracy)
  
  IF accuracy_gap > 0.3:
    => CONFIRMED ATTACK: Miscalibration
    => RETURN: reduce_confidence(expected_confidence) + add_caveat()
  ELSE:
    => FALSE ALARM: Confidence gap may be due to noise
    => RETURN: return_result() + monitor_confidence_accuracy()
```

---

## PART 4: Practical Workflows for Agents

### Workflow 1: Safe Query Execution

```
def safe_search(query: str) -> SearchResult:
    # Phase 1: Initial detection
    embedding = encode(query)
    
    collapse_risk = detect_embedding_collapse(embedding)
    IF collapse_risk > 0.7:
        => return USER_ERROR("Query too vague; please be more specific")
    
    # Phase 2: Retrieve with monitoring
    facts = retrieve(query, monitor_for_attacks=True)
    
    contradiction_risk = detect_semantic_poisoning(facts)
    IF contradiction_risk > 0.7:
        => return CLARIFICATION_NEEDED("Your query has conflicting goals")
    
    # Phase 3: Traverse with cycle detection
    extended_context = expand_via_containment(facts)
    IF contains_cycle(extended_context):
        => return PARTIAL_RESULT(facts, caveat="Circular reference detected")
    
    # Phase 4: Octave agreement check
    disagreement_risk = detect_boundary_ambiguity(query, facts)
    IF disagreement_risk > 0.7:
        => return CLARIFICATION_NEEDED("Query is ambiguous; please refine")
    
    # Phase 5: Confidence calibration check
    confidence_gap = detect_miscalibration(facts)
    IF confidence_gap > 0.2:
        => facts.confidence = adjust_confidence(facts.confidence, gap)
    
    # Phase 6: Return with confidence
    IF aggregate_risk_score(facts) > 0.6:
        => return RESULT(facts, confidence=degraded, caveat=explanation)
    ELSE:
        => return RESULT(facts, confidence=reported, trace=reasoning)
```

### Workflow 2: Adversarial Incident Response

```
def handle_suspicious_query(query: str, risk_score: float):
    
    # Log the incident
    log_event({
        timestamp: now(),
        query: query,
        risk_score: risk_score,
        user: current_user(),
    })
    
    # Determine severity
    IF risk_score > 0.9:
        => LEVEL: CRITICAL
        => ACTION: escalate_to_human()
        => RETURN: USER_ERROR("Unable to process; please refine")
    
    ELIF risk_score > 0.7:
        => LEVEL: HIGH
        => ACTION: ask_user_clarification()
        => ACTION: retrieve_from_coarser_octave()
        => RETURN: PARTIAL_RESULT(facts, confidence=0.3)
    
    ELIF risk_score > 0.5:
        => LEVEL: MEDIUM
        => ACTION: add_caveat_to_result()
        => ACTION: widen_apertures()
        => RETURN: RESULT(facts, confidence=degraded)
    
    ELIF risk_score > 0.3:
        => LEVEL: LOW
        => ACTION: monitor_confidence_accuracy()
        => ACTION: continue_normal_retrieval()
        => RETURN: RESULT(facts, confidence=normal, flag=monitored)
    
    # Offline analysis
    IF risk_score > 0.5:
        => send_to_analytics("suspicious_pattern", query, risk_score)
        => trigger_threshold_update_review()
```

---

## PART 5: Success Indicators for Agents

### "I Successfully Detected Adversarial Query" When:

1. **Embedding Collapse:**
   - [ ] Query norm ratio > 0.85
   - [ ] Member entropy < 0.3 * baseline
   - [ ] Retrieved facts scatter across multiple clusters
   - [ ] Action taken: confidence reduced, coarser octave used

2. **Semantic Poisoning:**
   - [ ] Query contains 2+ conflicting keywords
   - [ ] Fact entropy > 1.5 * baseline
   - [ ] Contradiction score > 0.4
   - [ ] Action taken: user asked to clarify priorities

3. **Circular Containment:**
   - [ ] DFS on containment graph found cycle
   - [ ] Traversal depth hit limit before root
   - [ ] Parent of query points back to query (distance <= 3)
   - [ ] Action taken: traversal stopped, partial result returned

4. **Boundary Ambiguity:**
   - [ ] 2+ octaves with max_score < 0.7
   - [ ] Octave disagreement entropy > 1.3 * baseline
   - [ ] Multiple clusters equally plausible
   - [ ] Action taken: user asked which interpretation, apertures widened

5. **Confidence Miscalibration:**
   - [ ] Confidence gap > 0.2
   - [ ] Accuracy gap > 0.3 (when validated)
   - [ ] Low context richness but high reported confidence
   - [ ] Action taken: confidence reduced to expected level

---

## PART 6: Agent Learning Loop

As agents encounter adversarial queries, they learn and improve:

```
LEARNING LOOP:

1. OBSERVE: Agent retrieves facts; notices high risk_score
2. LOG: Store (query, risk_score, action_taken, outcome)
3. ANALYZE: Offline analysis identifies new attack pattern
4. UPDATE: Threshold or metric retuned based on analysis
5. VALIDATE: New pattern tested on held-out adversarial set
6. PROPAGATE: Updated detection logic deployed to all agents

Example:
  - Agent notices queries with "AND" + opposing goals always score high
  - Log shows contradiction_entropy >> baseline every time
  - Update: increase weight of contradiction_entropy in risk_score formula
  - Test: adversarial queries (semantic poisoning) detected 95%+ precision
  - Deploy: all agents now use updated formula
```

---

## Summary: Information Theory as Defense

Information theory provides a **principled foundation** for adversarial detection:

1. **Entropy is a universal anomaly signal** — attacks spike entropy in unexpected dimensions
2. **Coherence is an assumption we can monitor** — break coherence and entropy rises
3. **Metadata is more stable than content** — norm ratio, entropy are robust to input variations
4. **Calibration is learnable** — confidence-accuracy relationship can be continuously updated

By monitoring entropy across embedding space, fact coherence, octave agreement, containment cycles, and confidence calibration, agents can detect **all five attack patterns** reliably.

