---
name: RLM Paper - Agent Reasoning & Intuition Extraction
date: 2026-06-21
source: arxiv 2512.24601 Recursive Language Models (Zhang, Kraska, Khattab)
focus: Agent decomposition, intent recognition, uncertainty handling, confidence calibration, explainability
---

## Executive Summary

The Recursive Language Model (RLM) paper encodes critical agent reasoning patterns:
1. **Goal Decomposition** — recursive task decomposition via symbolic manipulation
2. **Intent Recognition** — models implicitly recognize task complexity and decompose accordingly
3. **Uncertainty Handling** — agents navigate non-deterministic REPL outputs and errors
4. **Confidence Calibration** — agents detect when outputs are untrustworthy (syntax errors, incomplete states)
5. **Explainability** — symbolic code execution provides traces agents can reason about

This document maps these patterns to semiosis agent-intuition rows and derives new agent-reasoning PRD rows.

---

## 1. GOAL DECOMPOSITION PATTERNS

### RLM Paper Evidence

**Section 2: RLM Design, Algorithm 1 (The RLM Loop)**

The core RLM loop embeds goal decomposition:
```
state <- InitREPL(prompt=P)
while True do
    code <- LLMM(hist)  // Agent RECOGNIZES task structure
    (state, stdout) <- REPL(state, code)
    hist <- hist || code || Metadata(stdout)
    if state[Final] is set then return state[Final]
```

The agent (LLMM) must:
1. Recognize the initial prompt structure
2. Symbolically manipulate `P` (a variable handle, not text in context)
3. Decompose P into slices
4. Recursively invoke itself on slices
5. Stitch results together

**Key Decomposition Strategies Observed (Section 5: RLM Trajectory Analyses)**

- **Probing then Decomposition**: Models probe input (peek at head/tail) then strategically slice
- **Task Complexity Scaling** (Section 3): Decomposition depth scales with task complexity:
  - S-NIAH (O(1)): Single needle-seek, no decomposition needed
  - BrowseComp-Plus (O(n)): Linear multi-document reasoning, depth=1 sufficient
  - OOLONG (O(n)): Semantic aggregation, depth=1
  - OOLONG-Pairs (O(n²)): Pair aggregation, depth=2+ required
  - CodeQA (O(1) fixed files): Reasoning over fixed set, minimal depth

**Finding: Task Complexity Determines Decomposition Depth**

Agents implicitly recognize O(n) vs O(n²) vs O(1) complexity and allocate recursion depth accordingly.

### Semiosis Agent-Intuition Mapping

**Relevant Existing Rows (from Session 2):**
- `hierarchical-query-decomposition` (row 614) — decompose queries using hierarchy structure
- `query-decomposition-via-latent-semantics` (row 824) — learn system's own ontology
- `meta-reasoning-cone-probe` (row 705) — explain failure modes

**New PRD Rows Derived:**

#### Row: agent-complexity-sensing
- **Title:** Agents implicitly sense task complexity and allocate reasoning depth
- **Description:** RLM agents adaptively decompose based on perceived input complexity (O(n), O(n²), O(1)). Semiosis should expose this: when query complexity is detected as O(n²) (e.g., pairwise comparisons), system automatically enables depth-2+ retrieval; for O(1) queries, return single-octave answer. This grounds agent intuition in measurable complexity properties.
- **Implementation:** RecursiveAnswerEngine.decompose() detects query structure (single-target, pairwise, aggregation); sets max_depth accordingly; tests show depth allocation matches human complexity assessment 90%+.
- **Witness Path:** core/recursive.py detect_query_complexity() + test_complexity_depth_allocation

#### Row: agent-probe-then-slice-pattern
- **Title:** Agents probe input structure before decomposing
- **Description:** RLM agents peek at input before slicing (look at length, head, tail). They don't blindly split; they gather metadata then decompose. Semiosis agents should similarly probe the query/corpus: is the corpus dense or sparse? Are there natural clusters (high-entropy regions)? This metadata guides decomposition strategy. Example: if encoder detects corpus has 5 clear clusters vs 100 diffuse ones, prefer depth=1 (coarse) vs depth=3 (fine).
- **Implementation:** KnowledgeBase.ingest() probes corpus structure (entropy profile, cluster count estimate); agent_api.search() adapts decomposition strategy; test shows strategy selection correlates with actual corpus structure.
- **Witness Path:** core/pipeline.py _probe_corpus_structure + test_corpus_aware_decomposition

#### Row: agent-stitch-results-verification
- **Title:** Agent reasoning verifies stitched results from recursive calls
- **Description:** When RLMs stitch results from multiple recursive calls, they include verification: do the stitched results form a coherent answer? Do token counts make sense? This is quality-control reasoning. Semiosis agents should similarly verify when combining results from multiple octaves or memory layers: do the pieces fit together coherently? Flag when stitching produces incoherent results.
- **Implementation:** agent_api.py consolidate() checks coherence of merged facts; returns coherence_score; tests show low coherence flags problematic merges.
- **Witness Path:** core/agent_api.py check_result_coherence + test_stitching_verification

---

## 2. INTENT RECOGNITION

### RLM Paper Evidence

**Section 5: Analyses of RLM Trajectories**

Observed intent-recognition behaviors:
1. **Implicit Model Capability Awareness**: Models recognize when a task requires their own recursive invocation vs pure code execution. Example: simple lookups (grep-like) use code only; complex aggregations invoke sub-LM.

2. **In-Context Priors Improve Decomposition**: "In-context examples improve decomposition even if unrelated" (Section 5). Models use examples to calibrate intent understanding. Example: showing one decomposition example dramatically improves Qwen's decomposition quality.

3. **Error-Driven Decomposition Adjustment**: When code throws syntax errors (Qwen exhibits more syntax errors), agents adjust strategy. Higher error rates trigger fallback to coarser decomposition (depth 0 instead of depth 2).

**Finding: Intent is Calibrated via Examples and Error Feedback**

Agents refine their understanding of intent by observing feedback (error rates, example structure).

### Semiosis Agent-Intuition Mapping

#### Row: agent-capability-self-awareness
- **Title:** Agents recognize their own capability limits and match task complexity to capability
- **Description:** RLM agents know when to invoke sub-LMs vs pure code: simple retrieval = code only; complex reasoning = sub-LM. They implicitly calibrate. Semiosis agents should similarly recognize what they can handle: is the query within-cone (retrievable via direct knn)? or out-of-cone (needs probing/explanation)? High-capability agent escalates to more complex decomposition; low-capability (or when tired) returns simpler answers with caveats.
- **Implementation:** agent_api.py KnowledgeBase.search() calls _estimate_capability_match(query, stored_answers); returns (match_score, recommended_depth); agent can check match_score and decide to escalate or give up.
- **Witness Path:** core/agent_api.py estimate_capability_match + test_capability_aware_search

#### Row: agent-intent-calibration-via-examples
- **Title:** Agent intent understanding improves with in-context examples
- **Description:** RLM paper shows in-context examples (even unrelated) improve decomposition. Semiosis should expose example-based calibration: when an agent asks "show me an example of a good query decomposition", the KB returns exemplars from high-utility prior queries. Agent then adjusts intent model and decomposes better.
- **Implementation:** agent_api.py KnowledgeBase.get_exemplars(query_kind, k=3) returns high-value prior queries of same kind; tests show agent decomposition quality improves 15%+ after seeing exemplars.
- **Witness Path:** core/agent_api.py get_exemplars + test_exemplar_driven_calibration

#### Row: agent-error-driven-fallback
- **Title:** Agents detect errors and gracefully degrade strategy
- **Description:** When code execution produces errors (syntax errors, timeouts, assertion failures), RLM agents detect and adjust. Semiosis agents should similarly: if knn at depth=3 times out or returns low-confidence results, automatically fall back to depth=2 + broader retrieval. Track error rates per strategy and prefer lower-error paths.
- **Implementation:** core/recursive.py RecursiveAnswerEngine.descend() wraps calls with try-except; counts errors per octave_depth; switches to lower depth if error_rate > threshold; tests show automatic fallback prevents 80%+ of cascading failures.
- **Witness Path:** core/recursive.py error-driven fallback + test_graceful_degradation_on_errors

---

## 3. UNCERTAINTY HANDLING

### RLM Paper Evidence

**Section 5: Error Analysis**

RLM agents face sources of uncertainty:
1. **Non-Deterministic Code Execution**: REPL output varies (randomness, state changes)
2. **Syntax Errors**: Qwen exhibits 16% templating-error rate (Appendix A); agents must handle malformed code
3. **Incomplete States**: Code may not fully terminate (timeout, resource limit); agents must decide: continue or return partial answer?
4. **Output Truncation**: Print outputs truncated to prevent context overflow; agents work with partial information

**Observed Coping Strategies:**
- Agents use metadata (stdout length, execution time) to infer state quality
- Agents retry with simplified code when errors occur
- Agents explicitly check state.Final before returning (line: "if state[Final] is set")

**Appendix A Finding: Negative Results Section**
- Syntax errors common with smaller models (Qwen)
- Output token limits force early termination
- Sequential (slow) calls accumulate uncertainty
- Final-answer detection is brittle

### Semiosis Agent-Intuition Mapping

#### Row: agent-uncertainty-detection-via-metadata
- **Title:** Agents infer uncertainty from execution metadata, not just content
- **Description:** RLM agents use stdout length, execution time, error counts to assess result quality. Semiosis agents should similarly: high retrieval uncertainty when (a) multiple octaves disagree, (b) entropy of results is high, (c) no consensus meaning_vector, (d) centroid is far from query. Expose uncertainty_score = f(disagree_count, result_entropy, centroid_distance) in every search result.
- **Implementation:** core/agent_api.py compute_uncertainty_metadata(results, query_embedding, octaves) returns uncertainty_score; search() includes it; agent can threshold and request clarification.
- **Witness Path:** core/agent_api.py compute_uncertainty_metadata + test_uncertainty_from_metadata

#### Row: agent-retry-with-degradation
- **Title:** Agents retry with simpler/narrower strategy when execution fails
- **Description:** When code fails (syntax error), RLM agents retry with simplified code. Semiosis agents should retry with degraded strategy: query fails at depth=3 -> retry at depth=2 + broader apertures -> retry at depth=1 + coarse clusters. Each retry is simpler and faster; agent stops when uncertainty drops below threshold.
- **Implementation:** core/recursive.py RecursiveAnswerEngine.descend() with retry loop; each retry reduces decomposition complexity; tests show 2-3 retries recover 85% of failed queries.
- **Witness Path:** core/recursive.py retry_with_degradation + test_retry_strategy

#### Row: agent-partial-completion-assessment
- **Title:** Agents assess whether partial results are acceptable or incomplete
- **Description:** RLM agents work with time/resource limits and must decide: is the partial answer acceptable? Semiosis agents face similar: queries may time out before exhaustive retrieval. Formalize: completeness_score = (results_found / expected_results_lower_bound) * coherence_score. If completeness < 0.3, mark result as incomplete; if 0.3-0.7, mark as partial; if >0.7, return as complete.
- **Implementation:** core/agent_api.py assess_completion(results, query_complexity) returns (completeness_score, status='complete'|'partial'|'incomplete'); search() includes it; test shows human agreement ~85%.
- **Witness Path:** core/agent_api.py assess_completion + test_completeness_assessment

#### Row: agent-malformed-input-recovery
- **Title:** Agents detect malformed/ambiguous queries and request disambiguation
- **Description:** Analogous to syntax errors: some queries are ambiguous or malformed (typos, nonsense, contradictory). Instead of silent failure, agents should detect and ask for clarification. Example: query="GPU optimization and usability and performance" has ambiguous conjunction; agent asks "do you want (performance OR usability)? or (performance AND usability)?"
- **Implementation:** core/agent_api.py query_validator checks for ambiguity patterns (multiple AND/OR, conflicting terms); disambiguate(query) returns (is_ambiguous, clarification_options); test on WebGL queries shows 90%+ accuracy.
- **Witness Path:** core/agent_api.py disambiguate_query + test_ambiguous_query_detection

---

## 4. CONFIDENCE CALIBRATION

### RLM Paper Evidence

**Section 4: Results & Discussion**

Observation 3: "Degradation Function of Complexity"
- GPT-5 degradation scales with task complexity
- RLM degradation is MORE GRADUAL than vanilla models
- But degradation STILL OCCURS at extreme lengths (beyond 2^14 tokens)

This reveals confidence bounds: agents should know confidence drops as task complexity increases.

**Observation 5: Extended Reasoning Beyond Context**
- LongCoT-mini: RLM(GPT-5.2, depth=1) achieves 50.6% vs 38.7% base
- With decomposition hints: 65.6%

Hints improve confidence because agent can better calibrate its strategy.

### Semiosis Agent-Intuition Mapping

#### Row: agent-confidence-scales-with-task-complexity
- **Title:** Agent confidence should degrade gracefully with task complexity
- **Description:** RLM results show confidence degrades with O(n), O(n²), O(n³) complexity. Semiosis should track confidence as function of query_complexity: return confidence_score = 1.0 for O(1), 0.8 for O(n), 0.5 for O(n²), 0.3 for O(n³). This honest uncertainty calibration lets downstream systems adjust (request human review for low-confidence answers).
- **Implementation:** core/recursive.py compute_confidence_from_complexity(query_complexity, actual_result_quality); test on WebGL queries with known complexity shows confidence predictions match observed error rates.
- **Witness Path:** core/recursive.py compute_confidence_from_complexity + test_confidence_scales_correctly

#### Row: agent-hints-improve-confidence
- **Title:** Providing context/hints improves agent confidence and accuracy
- **Description:** RLM paper: decomposition hints improve results (38.7% -> 65.6%). Semiosis agents similarly: when context is rich (exemplars, prior queries, cluster memberships), confidence improves. Formalize: confidence = f(context_richness, task_complexity). Give agents richer context (exemplars, similar queries) to boost confidence.
- **Implementation:** agent_api.py KnowledgeBase.search() with hints=True includes exemplars, similar_prior_queries, cluster_metadata; measure result_quality with/without hints; tests show 20%+ improvement with hints.
- **Witness Path:** core/agent_api.py enrich_context_with_hints + test_hints_improve_confidence

#### Row: agent-confidence-communication
- **Title:** Agents explicitly communicate confidence and calibration to users
- **Description:** Confidence should not be silent. Agents should report: "I found X with confidence 0.85 (high complexity question, limited context)". Users then adjust their trust. This is the "explainability" aspect: confidence bounds are part of explanation.
- **Implementation:** core/agent_api.py SearchHit includes confidence_score and confidence_reasoning (e.g., "high due to 3-octave agreement, low due to O(n²) complexity"). API /explain endpoint returns full reasoning.
- **Witness Path:** core/agent_api.py SearchHit.confidence + test_confidence_communication

---

## 5. EXPLAINABILITY

### RLM Paper Evidence

**Section 5: Key Appendices**

Appendix E: RLM Trajectory Examples
- Shows full code execution traces
- Code is human-readable: regex filtering, recursive calls are explicit
- Agents (and users) can inspect every step

Appendix C: System Prompts
- Emphasizes "code execution, recursive sub-calling, state management"
- Prompts guide agents toward transparent, traceable reasoning

**Finding: Explainability via Symbolic Traces**

Code execution is inherently more explainable than opaque neural reasoning. Agents reason symbolically (code) with explicit state management.

### Semiosis Agent-Intuition Mapping

#### Row: agent-reasoning-trace-export
- **Title:** Agents expose full reasoning traces (decomposition, retrieval, stitching)
- **Description:** Like RLM's code execution traces, Semiosis agents should export reasoning_trace: (1) query decomposition step, (2) octave retrieval steps, (3) result stitching, (4) confidence assessment. Users (and other agents) can audit the reasoning.
- **Implementation:** core/agent_api.py KnowledgeBase.search(return_trace=True) returns (results, trace); trace includes decomposition, octave_steps, stitching_logic, confidence_reasons; test on WebGL queries shows traces are human-readable.
- **Witness Path:** core/agent_api.py SearchTraceResult + test_reasoning_trace_export

#### Row: agent-state-management-explicit
- **Title:** Agent state is explicit and inspectable at each step
- **Description:** RLM agents manage state explicitly (state.Final, state.stdout). Semiosis agents should similarly: state = {query, decomposed_queries, octave_frontier, retrieved_nodes, stitched_result, confidence_score, final_answer}. At each step, state is well-defined and inspectable. This enables debugging: "why did agent choose octave 2 over 3?"
- **Implementation:** core/agent_api.py define AgentState dataclass; search() maintains state at each step; debug_search(query, step_limit) stops at step and shows state; test shows users can diagnose failures by inspecting state.
- **Witness Path:** core/agent_api.py AgentState dataclass + test_state_inspection

#### Row: agent-failure-explanation
- **Title:** When agents fail, they explain why in terms of reasoning logic
- **Description:** RLM's negative results (Appendix A) document failure modes (syntax errors, output limits, brittle final-answer detection). Semiosis should do the same: when search fails, explain why: "no clusters matched the query", "query on cluster boundary (ambiguous)", "corpus too small for depth-2 reasoning". This grounds explainability in measurable system properties.
- **Implementation:** core/agent_api.py explain_failure(query, attempt_trace) analyzes trace and returns (failure_mode, root_cause, suggestion); test on failing WebGL queries shows explanations are actionable.
- **Witness Path:** core/agent_api.py explain_failure + test_failure_explanation

---

## 6. MENTAL MODELS & DECISION-MAKING

### RLM Paper Evidence

**Section 1: Problem Statement**

RLM agents must build implicit models of:
1. **Task Structure**: Is this a simple lookup (O(1)) or complex multi-document reasoning (O(n²))?
2. **Input Characteristics**: How long is the prompt? How dense is the information?
3. **Their Own Capabilities**: Can I solve this in one pass, or do I need recursion?

**Section 2: Three Critical Design Choices**

The design choices reveal agent mental models:
1. **Symbolic Handle to Prompt** (not text in context) — agent models that context limits are a constraint to work around symbolically
2. **Unbounded Output** (not limited by context window) — agent models that output should scale with problem, not window size
3. **Symbolic Recursion** (not separate LLM invocations) — agent models that looping inside code is cheaper/faster than separate sub-calls

### Semiosis Agent-Intuition Mapping

#### Row: agent-mental-model-of-hierarchy
- **Title:** Agents build and use mental models of hierarchy (octaves, clusters, apertures)
- **Description:** Semiosis agents should implicitly learn: coarse octaves are fast but low-precision; fine octaves are slow but high-precision. They build a mental model of when each is useful. This model is reflected in their decomposition choices and confidence calibration. Expose via: agent.explain_mental_model(octave) -> "octave 64 is fast and good for broad topics; octave 512 is slow and good for fine distinctions".
- **Implementation:** core/agent_api.py build_mental_model(query_history, performance_data) uses outcomes to calibrate model; explain_mental_model(octave) returns structured explanation; test shows agent decisions correlate with inferred model.
- **Witness Path:** core/agent_api.py build_mental_model + test_mental_model_inference

#### Row: agent-decision-criteria
- **Title:** Agents make explicit decisions based on observable system properties
- **Description:** Rather than black-box decisions, agents should decide based on measurable criteria: "choose depth=2 if query_complexity > 1.5" or "fall back to coarse octave if fine-octave_latency > 1s". Document the decision thresholds.
- **Implementation:** core/agent_api.py DecisionCriteria dataclass; search() logs which criteria triggered which decisions; test shows criteria explain >80% of decisions.
- **Witness Path:** core/agent_api.py DecisionCriteria + test_decision_criteria_logging

#### Row: agent-cost-benefit-reasoning
- **Title:** Agents reason about cost-benefit: is deeper reasoning worth the latency?
- **Description:** RLM agents trade off precision (recursion depth) vs cost (token count, latency). Semiosis agents should similarly: "query is O(n²), depth=2 is needed (80% chance of answer), but costs 500ms. Depth=1 is 300ms but only 50% chance. User deadline is 2s, so depth=2 is OK." Formalize cost_benefit = (expected_precision - cost_penalty) and decide based on user constraints.
- **Implementation:** core/recursive.py compute_cost_benefit(depth, cost_budget, precision_target) returns score; descend() uses cost_benefit to choose depth; test on constrained queries shows decisions match human risk/benefit analysis.
- **Witness Path:** core/recursive.py cost_benefit_reasoning + test_cost_benefit_decisions

---

## 7. NEW PRD ROWS SUMMARY

All new agent-reasoning rows discovered from RLM paper:

| ID | Title | Category | Est. Effort | Priority |
|---|---|---|---|---|
| agent-complexity-sensing | Agents sense task complexity and allocate depth | Reasoning | Medium | High |
| agent-probe-then-slice | Agents probe input before decomposing | Reasoning | Low | High |
| agent-stitch-verification | Agents verify stitched results for coherence | Reasoning | Low | Medium |
| agent-capability-self-awareness | Agents recognize own capability limits | Reasoning | Medium | High |
| agent-intent-calibration | Intent improves with in-context exemplars | Reasoning | Low | Medium |
| agent-error-driven-fallback | Agents degrade strategy gracefully on errors | Reasoning | Medium | High |
| agent-uncertainty-detection | Agents infer uncertainty from metadata | Reasoning | Low | High |
| agent-retry-with-degradation | Agents retry with simpler strategies | Reasoning | Medium | Medium |
| agent-partial-completion | Agents assess whether results are complete | Reasoning | Low | Medium |
| agent-malformed-input-recovery | Agents detect ambiguous queries and clarify | Reasoning | Medium | Low |
| agent-confidence-complexity | Confidence degrades with task complexity | Calibration | Low | High |
| agent-hints-improve-confidence | Hints (context/exemplars) boost confidence | Calibration | Low | Medium |
| agent-confidence-communication | Agents communicate confidence explicitly | Calibration | Low | Medium |
| agent-reasoning-trace | Agents export full reasoning traces | Explainability | Low | High |
| agent-state-explicit | Agent state is explicit and inspectable | Explainability | Medium | High |
| agent-failure-explanation | Agents explain failures via reasoning logic | Explainability | Medium | Medium |
| agent-mental-model | Agents build and use implicit models | Decision-Making | Medium | Medium |
| agent-decision-criteria | Agent decisions based on observable criteria | Decision-Making | Low | High |
| agent-cost-benefit | Agents reason about cost-benefit tradeoffs | Decision-Making | Medium | Medium |

---

## 8. MAPPING TO EXISTING PRD ROWS

**Enhanced by RLM insights:**
- `hierarchical-query-decomposition` (row 614) — now grounded in RLM complexity sensing
- `query-decomposition-via-latent-semantics` (row 824) — now informed by intent calibration
- `meta-reasoning-cone-probe` (row 705) — now includes error-driven fallback
- `uncertainty-quantification-in-retrieval` (row 831) — now uses metadata-based inference
- `explain-retrieval-trace` (row 501) — now includes explicit state management + decision logging
- `failure-mode-taxonomy` (row 845) — now includes malformed-input recovery

**New rows form cohesive agent-reasoning layer** that bridges semiosis cone math with agent decision-making.

---

## 9. IMPLEMENTATION ROADMAP

### Phase 1: Uncertainty & Confidence (weeks 1-2)
- agent-uncertainty-detection
- agent-confidence-complexity
- agent-confidence-communication
- Tests validate confidence calibration on WebGL corpus

### Phase 2: Decomposition & Decision (weeks 2-4)
- agent-complexity-sensing
- agent-probe-then-slice
- agent-decision-criteria
- agent-cost-benefit
- Integrate into RecursiveAnswerEngine

### Phase 3: Error Handling & Recovery (weeks 4-5)
- agent-error-driven-fallback
- agent-retry-with-degradation
- agent-partial-completion
- agent-malformed-input-recovery
- Add resilience tests

### Phase 4: Explainability & Transparency (weeks 5-6)
- agent-reasoning-trace
- agent-state-explicit
- agent-failure-explanation
- Documentation + examples

### Phase 5: Mental Models & Calibration (weeks 6-8)
- agent-mental-model
- agent-capability-self-awareness
- agent-intent-calibration
- agent-hints-improve-confidence
- agent-stitch-verification

---

## 10. EVIDENCE SUMMARY

All claims grounded in RLM paper sections:
- **Decomposition patterns**: Section 5, Appendix E
- **Complexity scaling**: Section 3
- **Error handling**: Section 5, Appendix A
- **Confidence degradation**: Section 4, Observations 3-5
- **Explainability**: Appendix C (system prompts), Appendix E (examples)
- **Decision-making**: Section 1-2 (design choices)

---

## 11. REFERENCES

Paper: Recursive Language Models (Zhang, Kraska, Khattab)
ArXiv: https://arxiv.org/abs/2512.24601
License: CC BY 4.0
Code: https://github.com/alexzhang13/rlm

Key Sections:
- Section 1: Problem formulation + RLM idea
- Section 2: RLM design + Algorithm 1
- Section 3: Task complexity scaling
- Section 4: Results + performance analysis
- Section 5: Trajectory analysis + error analysis
- Appendix A: Negative results + failure modes
- Appendix C: System prompts
- Appendix E: Full trajectory examples
