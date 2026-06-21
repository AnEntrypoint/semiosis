# Session 3: Agent Reasoning & Intuition Extraction from RLM Paper

**Date:** 2026-06-21  
**Focus:** Recursive Language Models (arxiv 2512.24601) — agent decomposition, intent recognition, uncertainty handling, confidence calibration, explainability  
**Output:** 19 new PRD rows + comprehensive extraction document

---

## Overview

This session extracts agent-reasoning patterns from the Recursive Language Models (RLM) paper and maps them to Semiosis agent-intuition features. The RLM paper provides empirical evidence of how language models:

1. Decompose complex tasks recursively
2. Recognize task intent and complexity
3. Handle execution uncertainty
4. Calibrate confidence in results
5. Provide transparent, auditable reasoning via code execution traces

---

## 5 Core Agent Reasoning Patterns

### 1. Goal Decomposition (Rows: agent-complexity-sensing, agent-probe-then-slice, agent-stitch-verification)

**RLM Evidence:** Section 2, 5; Algorithm 1

RLM agents decompose long inputs by:
- Recognizing input structure (O(1) vs O(n) vs O(n^2) complexity)
- Probing the prompt before decomposing (examining length, head, tail)
- Symbolically manipulating the input via REPL handles (not copying text into context)
- Stitching results from sub-calls while verifying coherence

**Key Finding:** Task complexity scales decomposition depth. Simple O(1) tasks (needle-in-haystack) require zero recursion. Complex O(n^2) tasks (pairwise aggregation) require depth-2+ recursion.

**Semiosis Integration:**
- Detect query complexity and allocate octave retrieval depth automatically
- Probe corpus structure before deciding retrieval strategy
- Verify stitched results maintain coherence (no contradictions across octaves)

---

### 2. Intent Recognition (Rows: agent-capability-self-awareness, agent-intent-calibration, agent-error-driven-fallback)

**RLM Evidence:** Section 5; Appendix A (error analysis)

RLM agents recognize:
- When their own capabilities are insufficient (error rates spike)
- How to improve decomposition with in-context examples (38.7% -> 65.6% improvement)
- When to adjust strategy based on errors (syntax errors trigger fallback to simpler code)

**Key Finding:** In-context examples improve intent understanding even if unrelated. Error feedback drives strategy adjustment.

**Semiosis Integration:**
- Estimate capability match between query and stored knowledge
- Use exemplars from prior queries to calibrate decomposition intent
- Automatically degrade strategy when errors accumulate

---

### 3. Uncertainty Handling (Rows: agent-uncertainty-detection, agent-retry-with-degradation, agent-partial-completion, agent-malformed-input-recovery)

**RLM Evidence:** Section 5 (trajectory analysis), Appendix A (negative results)

RLM agents face:
- Non-deterministic REPL outputs
- Syntax errors (16% rate in Qwen; higher in smaller models)
- Incomplete states (timeouts, resource limits)
- Malformed/ambiguous output from sub-calls

**Key Finding:** Agents infer state quality from metadata (stdout length, execution time, error counts), not just content. They retry with simplified strategies when errors occur.

**Semiosis Integration:**
- Compute uncertainty from metadata (result entropy, octave disagreement, centroid distance)
- Retry with degraded strategies (depth=3 -> depth=2 -> depth=1)
- Assess result completeness (complete/partial/incomplete) and mark appropriately
- Detect ambiguous queries and request clarification

---

### 4. Confidence Calibration (Rows: agent-confidence-complexity, agent-hints-improve-confidence, agent-confidence-communication)

**RLM Evidence:** Section 4 (Observations 3, 5)

RLM results show:
- Confidence degrades gracefully with task complexity (constant < linear < quadratic)
- Decomposition hints significantly improve confidence (38.7% base -> 65.6% with hints)
- Confidence remains honestly uncertain even with hints (not overconfident)

**Key Finding:** Confidence is a function of task complexity and context richness. Honest uncertainty bounds are more useful than false confidence.

**Semiosis Integration:**
- Return confidence scores calibrated to task complexity
- Boost confidence when richer context (exemplars, prior queries) is available
- Communicate confidence explicitly in results and explanations

---

### 5. Explainability (Rows: agent-reasoning-trace, agent-state-explicit, agent-failure-explanation, agent-mental-model, agent-decision-criteria, agent-cost-benefit)

**RLM Evidence:** Appendix C (system prompts emphasizing transparency), Appendix E (trajectory examples with full code traces)

RLM explainability stems from:
- Symbolic code execution (every step is human-readable)
- Explicit state management (state.Final, state.stdout are transparent)
- Failure mode documentation (Appendix A negative results)
- System prompts guiding agents toward traceable reasoning

**Key Finding:** Symbolic execution is inherently more explainable than opaque neural reasoning. Traces enable debugging and auditing.

**Semiosis Integration:**
- Export full reasoning traces (decomposition, retrieval, stitching, confidence assessment)
- Expose explicit agent state at each step (inspectable, debuggable)
- Explain failures in terms of system properties (no clusters matched, query ambiguous, corpus too small)
- Document mental models agents build about hierarchy (coarse=fast, fine=precise)
- Log all decision criteria used (complexity thresholds, latency budgets, precision targets)
- Show cost-benefit reasoning (why depth=2 was chosen over depth=1)

---

## 19 New PRD Rows Summary

| Group | Row ID | Title | Priority |
|-------|--------|-------|----------|
| **Decomposition** | agent-complexity-sensing | Sense task complexity and allocate depth | High |
| | agent-probe-then-slice | Probe structure before decomposing | High |
| | agent-stitch-verification | Verify stitched result coherence | Medium |
| **Intent** | agent-capability-self-awareness | Recognize capability limits | High |
| | agent-intent-calibration | Intent improves with exemplars | Medium |
| | agent-error-driven-fallback | Degrade gracefully on errors | High |
| **Uncertainty** | agent-uncertainty-detection | Infer uncertainty from metadata | High |
| | agent-retry-with-degradation | Retry with simpler strategies | Medium |
| | agent-partial-completion | Assess result completeness | Medium |
| | agent-malformed-input-recovery | Detect ambiguous queries | Low |
| **Confidence** | agent-confidence-complexity | Confidence scales with complexity | High |
| | agent-hints-improve-confidence | Context/hints boost confidence | Medium |
| | agent-confidence-communication | Communicate confidence explicitly | Medium |
| **Explainability** | agent-reasoning-trace | Export full reasoning traces | High |
| | agent-state-explicit | State explicit and inspectable | High |
| | agent-failure-explanation | Explain failures via reasoning logic | Medium |
| **Decision-Making** | agent-mental-model | Build and use mental models | Medium |
| | agent-decision-criteria | Decisions based on observable criteria | High |
| | agent-cost-benefit | Reason about cost-benefit tradeoffs | Medium |

---

## Implementation Phases

### Phase 1: Uncertainty & Confidence (weeks 1-2)
- agent-uncertainty-detection
- agent-confidence-complexity
- agent-confidence-communication
- Tests validate calibration on WebGL corpus

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

## Mapping to Existing PRD Rows

These new rows enhance and ground existing semiosis features:

**Enhanced by agent-reasoning insights:**
- `hierarchical-query-decomposition` (row 614) — now informed by complexity-sensing
- `query-decomposition-via-latent-semantics` (row 824) — now uses intent-calibration
- `meta-reasoning-cone-probe` (row 705) — now includes error-driven fallback
- `uncertainty-quantification-in-retrieval` (row 831) — now uses metadata-based inference
- `explain-retrieval-trace` (row 501) — now includes explicit state + decision logging
- `failure-mode-taxonomy` (row 845) — now includes malformed-input recovery

**Forming a cohesive agent-intuition layer** that bridges cone math with agent decision-making.

---

## Evidence from RLM Paper

All patterns grounded in paper sections:

**Decomposition & Complexity:**
- Section 2: RLM Design, Algorithm 1 (core loop with LLMM invocation)
- Section 3: Task complexity scaling (O(1) to O(n^2) complexity levels)
- Section 4, Observation 3: "Degradation function of complexity"
- Section 5: Trajectory analysis showing probe-then-decompose pattern
- Appendix E: Full trajectory examples with probing and stitching

**Intent Recognition:**
- Section 1: Problem statement (agents must model task structure)
- Section 2: Three critical design choices (revealing agent mental models)
- Section 5: Error analysis showing error-driven strategy adjustment
- Appendix A: Negative results (16% syntax error rate, affects decomposition quality)

**Uncertainty & Confidence:**
- Section 4, Observation 5: "Extended reasoning beyond context" with hints improving results (38.7% -> 65.6%)
- Section 4, Observation 3: Degradation scales with task complexity
- Section 5: Trajectory examples showing agents work with partial/incomplete information
- Appendix A: Negative results documenting failure modes (syntax errors, output limits, incomplete states)

**Explainability:**
- Appendix C: System prompts emphasizing "code execution, recursive sub-calling, state management"
- Appendix E: Full trajectory examples with human-readable code execution traces
- Section 5: Explicit discussion of decomposition patterns and reasoning

---

## Key Insights for Agent Intuition

1. **Agents can sense complexity:** Observable task properties (decomposability, information density, pairwise vs linear) trigger different strategies. Expose these properties and let agents decide.

2. **Agents learn from feedback:** Error rates, example quality, outcome distributions teach agents how to improve decomposition and strategy. Build feedback loops.

3. **Agents calibrate confidence honestly:** Rather than fake certainty, expose uncertainty bounds and let downstream systems adjust (humans review low-confidence, auto-accept high-confidence).

4. **Agents reason via transparency:** Code execution traces, state inspection, decision logging enable agents and humans to debug and audit reasoning. Invest in explainability.

5. **Agents adapt to constraints:** Cost-benefit reasoning, latency budgets, precision targets are all observable. Let agents trade-off according to user constraints.

---

## Next Steps

1. Implement Phase 1 (uncertainty & confidence) as proof-of-concept
2. Add tests validating confidence calibration on WebGL corpus
3. Integrate into existing RecursiveAnswerEngine and agent_api.py
4. Document decision thresholds and reasoning in AGENTS.md
5. Create agent-intuition-guide.md for external agent developers

---

## References

**Paper:**  
- Title: Recursive Language Models  
- Authors: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)  
- ArXiv: https://arxiv.org/abs/2512.24601  
- License: CC BY 4.0  

**Key Sections:**
- Section 1: Problem formulation
- Section 2: RLM design + Algorithm 1
- Section 3: Task complexity scaling
- Section 4: Results + performance analysis
- Section 5: Trajectory analysis + error analysis
- Appendix A: Negative results + failure modes
- Appendix C: System prompts
- Appendix E: Trajectory examples

**Related Semiosis Docs:**
- docs/rlm-agent-reasoning-extraction.md (full extraction with evidence)
- docs/paper-insights-summary.md (integration of prior papers)
- docs/agent-guide.md (agent surface documentation)
