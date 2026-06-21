# LEARNING AND ADAPTATION PATTERNS FROM RLM PAPER
## Extracted for Semiosis Integration

### Core Extraction: 13 Learning and Adaptation Concepts

From arxiv 2512.24601 (Recursive Language Models), the following patterns are directly applicable to semiosis's cone-based retrieval and octave hierarchy:

---

## 1. **Incremental Learning via Trajectory Filtering**

**RLM Evidence:**
- RLM-Qwen3-8B trained on only 1,000 filtered trajectories achieving 28.3% median improvement
- Simple SFT (Supervised Fine-Tuning) on curated dataset generalizes across diverse tasks
- Trajectories include both successful paths and error+recovery patterns

**Semiosis Application:**
- Octave apertures learned from outcome feedback on actual queries
- Track (query_kind, octave_path, result_quality) tuples in KnowledgeBase.record_outcome()
- Periodically batch feedback into aperture adjustments without full retraining
- Cone clustering becomes learnable: aperture tightness adjusts based on precision/recall

**Implementation Location:** `core/agent_api.py` → `record_outcome()` + `consolidate()` learning loop

---

## 2. **Online Learning via Execution Feedback**

**RLM Evidence:**
- REPL loop provides immediate binary feedback: code success/failure
- Exception messages and stdout length constraints guide learning
- Models learn error recovery patterns from execution environment

**Semiosis Application:**
- Retrieval failures (outside_all, boundary_ambiguous, over_compressed) generate feedback
- Extend scan_tension() to log failure modes as atoms in a sequence
- Lightweight learning loop detects failure-type patterns, adjusts geometry
- ContextPackBuilder learns which memory layers to prioritize per query type

**Implementation Location:** `core/cone_engine.py` → `scan_tension()` + `core/agent_api.py` → failure tracking

---

## 3. **Parameter Tuning via Complexity-Aware Adaptation**

**RLM Evidence:**
- Recursion depth learned implicitly per task complexity
- O(1) needle-in-haystack (constant depth), O(n) linear search (medium depth), O(n²) pair aggregation (deep recursion)
- Model learns cost-quality tradeoff: deeper = more accurate but more expensive

**Semiosis Application:**
- Query complexity estimated from embedding entropy and relevant document span
- Aperture size, octave depth, centroid recompute frequency become learnable parameters
- Low-entropy queries → tight apertures (fast); High-entropy → loose apertures or multi-octave decomposition
- Complexity signal stored with outcomes to train estimator

**Implementation Location:** `core/recursive.py` → `decompose_query()` with complexity parameterization

---

## 4. **Transfer Learning: Length Generalization**

**RLM Evidence:**
- Trained on 64k-token sequences; generalizes to 1M+ tokens (16x longer)
- Decomposition strategy (probe → decompose → recurse → stitch) transfers without retraining
- Learned algorithmic patterns scale independently of absolute input length

**Semiosis Application:**
- Apertures and memory layer strategies learned on small stores (1k nodes) transfer to large stores (100k+)
- Cone geometry learned with tight constraints transfers when entropy distributions shift
- Key insight: learn the decision algorithm, not the parameter values
- Policy-based learning: "if entropy > threshold, split; if aperture_consistency < 0.7, tighten"

**Implementation Location:** `core/interfaces.py` → `HierarchicalClusterer` abstract policy + `core/agent_api.py` → cross-domain transfer

---

## 5. **Self-Improvement via Decomposition**

**RLM Evidence:**
- Models learn to generate programs that recursively call themselves with transformed inputs
- Code generation is self-modification: input transformation + execution strategy adjustment
- Unbounded reasoning without weight updates via symbolic recursion

**Semiosis Application:**
- Queries decompose recursively: high-level → octave selection → centroid search → member ranking → memory layer selection
- Each layer generates an execution program: "search X with apertures Y; if ambiguous, decompose into sub-queries Z1, Z2"
- System learns decomposition structure from outcome feedback
- Enables non-monotonic reasoning: retrieve → evaluate → redecompose → retrieve again

**Implementation Location:** `core/recursive.py` → `RecursiveAnswerEngine` decomposition grammar

---

## 6. **Plasticity: Dynamic Strategy Selection**

**RLM Evidence:**
- Single 8B model handles 4 very different tasks (CodeQA, Research, OOLONG-Pairs, Long reasoning) without task-specific weights
- Adapts decomposition strategy dynamically based on input characteristics
- System prompts guide but don't constrain adaptation

**Semiosis Application:**
- Same cone structure handles dense (high-entropy) and sparse (low-entropy) domains
- Domain-agnostic principles: tight apertures for signal, loose for exploration
- Query intent inferred from embedding geometry, not explicit tagging
- Plasticity emerges from multi-scale pattern learning, not from domain-specific parameters

**Implementation Location:** `core/agent_api.py` → `Query.execute()` with dynamic aperture/depth selection

---

## 7. **Feedback Loops: Error Propagation and Recovery**

**RLM Evidence:**
- Syntax errors in generated code; failures in recursive calls; context overflow
- Initial decomposition critical; wrong early choices propagate through recursion tree
- Training includes 16% error+recovery patterns; system learns recovery strategies

**Semiosis Application:**
- Retrieval failures trigger adaptive recovery: re-query with loosened apertures, switch octaves, redecompose
- Store learns which recovery strategies work per failure mode and query type
- Failures become labeled examples in learning loop
- Failure-mode taxonomy: outside_all, boundary_ambiguous, centroid_not_found, over_compressed, octave_mismatch

**Implementation Location:** `core/agent_api.py` → `deep_search()` error handling + recovery policy learning

---

## 8. **Cost-Aware Learning: Token Budgeting**

**RLM Evidence:**
- Median RLM cost equals/beats base model cost despite sub-calls
- Model learns when to invoke sub-calls (expensive) vs inline computation (cheap)
- Cost awareness emerges from training on constrained inference budgets

**Semiosis Application:**
- Context packing learns token-value tradeoffs: summaries vs details, aperture tightness vs looseness
- Tight apertures = fewer members to encode (cheap) but less coverage
- Loose apertures = more members, better coverage but expensive
- Memory layer compression learns information worth keeping
- Octave-specific cache TTLs: summaries (high-value, low-cost) stay 10x longer than raw members

**Implementation Location:** `core/context_pack.py` → `ContextPackBuilder` + `core/semiotic_memory.py` layer-specific TTLs

---

## 9. **Learning to Decompose: Granularity Discovery**

**RLM Evidence:**
- Decomposition strategy not pre-programmed; learned from examples
- Discovers sub-task granularity, query slicing boundaries, result stitching strategy
- When to decompose (avoid overflow), how deep (cost vs quality), how to merge results

**Semiosis Application:**
- Aperture size and octave depth become learnable parameters
- Tight apertures = fine-grained decomposition (many small clusters)
- Loose apertures = coarse decomposition (few large clusters)
- System learns optimal granularity per query type from outcome feedback
- Transitive containment closure becomes learnable cache

**Implementation Location:** `core/recursive.py` → `decompose_query()` with granularity parameterization

---

## 10. **Multi-Scale Reasoning: Emergence of Hierarchy**

**RLM Evidence:**
- Recursion depths (0, 1, 2+) exhibit emergent hierarchy
- Depth 0: fast/shallow; Depth 1: balanced; Depth 2+: slow/deep
- Different task complexities benefit from different depths

**Semiosis Application:**
- Matryoshka octaves align with recursion depths
- Root octave for global queries, sub-octaves for focused, deepest for fine-grained
- System learns which depth minimizes expected cost per query type
- Multi-scale plasticity: same cone geometry (aperture, tension, flow) applies at all scales

**Implementation Location:** `core/cone_engine.py` → octave-aware depth selection

---

## 11. **Context Window Management: State Compression**

**RLM Evidence:**
- Learns output pacing: what to keep in state, what to summarize, what to discard
- Symbolic handles reference full text without copying into context
- Truncates long outputs to prevent overflow while preserving next-step information

**Semiosis Application:**
- SemioticMemory's 4 layers (facts, summaries, working, session) mirror output pacing
- System learns which layer for which query type
- Symbolic handles in ConeNode (centroid, aperture, member_ids) avoid copying full content
- Memory layer compression ratios: facts 1:100, summaries 1:10, working 1:2, session 1:1

**Implementation Location:** `core/semiotic_memory.py` → layer-specific compression rates learned per query type

---

## 12. **Adversarial Learning: Hard Example Mining**

**RLM Evidence:**
- Training uses 1,000 filtered (curated) trajectories, not random samples
- Filtering targets hard cases: syntax errors, edge cases, boundary conditions
- Hard example mining amplifies learning signal from rare but critical failures

**Semiosis Application:**
- Adversarial query stress testing becomes core learning mechanism
- Construct queries at cone boundaries, with conflicting cluster memberships, extreme entropy
- Learn aperture robustness from hard examples
- Active learning loop: flag uncertain retrievals, analyze failure mode, amplify in training

**Implementation Location:** `core/eval.py` → `adversarial_query_stress_test()` + learning weight amplification

---

## 13. **In-Context Learning: Few-Shot Plasticity**

**RLM Evidence:**
- In-context examples improve decomposition even if unrelated to task
- Single trajectory example changes model behavior without weight updates
- Few-shot plasticity: immediate behavioral change from context alone
- Shows models have learned decomposition meta-patterns that generalize across domains

**Semiosis Application:**
- System prompts in agent_api.py become powerful learning mechanisms
- Query-specific prompts guide decomposition without retraining
- Examples in prompts (successful retrieval traces) improve performance immediately
- Knowledge base learns which example traces transfer best across query types
- Few-shot learning enables domain adaptation without fine-tuning

**Implementation Location:** `core/agent_api.py` → system prompt engineering + example ranking

---

## Integration with Existing PRD

These 13 concepts map to Session 2 PRD rows and extend them:

**Already covered (Session 2):**
- Row 31: multi-scale-feature-learning → Concept 10
- Row 38: hierarchical-relevance-feedback-loop → Concept 7
- Row 39: cross-domain-transfer-learning → Concept 4
- Row 37: failure-mode-taxonomy → Concept 7
- Row 6: learning-loop-entropy-signals → Concept 2

**New from RLM (Session 3):**
- Concept 3: parameter-tuning-complexity-aware (new parameterization depth)
- Concept 5: self-improvement-via-decomposition (recursive grammar learning)
- Concept 6: plasticity-dynamic-strategy (query-intent inference)
- Concept 8: cost-aware-learning-budgeting (new cache TTL strategies)
- Concept 9: learning-to-decompose-granularity (octave-aware granularity learning)
- Concept 12: adversarial-learning-hard-examples (active learning via stress testing)
- Concept 13: in-context-learning-few-shot (system prompt optimization)

---

## Implementation Priorities (Highest First)

**Phase 1 (Immediate, 3-5 lines each):**
1. Concept 2: Failure mode tracking in scan_tension()
2. Concept 8: Cache TTL per layer in consolidate()
3. Concept 13: In-context example ranking in prompts

**Phase 2 (Medium, 10-20 lines each):**
1. Concept 7: Error recovery policy learning in deep_search()
2. Concept 3: Complexity estimation in decompose_query()
3. Concept 9: Granularity parameter learning

**Phase 3 (Architectural, refactoring required):**
1. Concept 5: Decomposition grammar as learnable program structure
2. Concept 6: Query-intent inference from embedding geometry
3. Concept 4: Cross-domain policy transfer

---

## Key Insight for Semiosis

RLM's core contribution is learning WHEN to decompose, not HOW to decompose. The algorithm (REPL loop) is fixed; the strategy (recursion depth, sub-call placement, output stitching) is learned.

**Semiosis parallel:** 
The cone algorithm is fixed (Lorentz geometry, tension/flow ops). The strategy (aperture tightness, octave selection, memory layer prioritization) should be learned from outcome feedback.

This transforms semiosis from a static geometric system to an adaptive learning system that improves with every query.
