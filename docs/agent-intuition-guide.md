# Agent Intuition Guide

How to read semiosis signals as implicit intuition for navigation, confidence, and self-regulation.

## 1. Cone Fundamentals

The KB organizes knowledge as hyperbolic entailment cones (Ganea 2018). Each node has an apex in
Lorentz space and a half-aperture `psi`. A cone contains another when its apex lies within the
parent's angular opening -- containment is semantic entailment.

Signal reading:
- `aperture` near `0.1` (floor): node encodes a sharp, highly specific concept.
- `aperture` near `pi/2`: node is a broad category, many children entailed.
- `score` on a SearchHit: dot-product relevance to query in embedding space (not cone containment).

The cone math drives hierarchy and energy; the embedding centroid drives retrieval rank.

Example 1: query "photosynthesis"
- Hit A: aperture=0.15, score=0.92, evidence_path_count=3 -- specific node, multi-octave consensus, high confidence.
- Hit B: aperture=0.85, score=0.74, evidence_path_count=1 -- broad biology node, single-octave, less specific.

Example 2: query "bank" (ambiguous)
- Hit A: aperture=0.3, local_entropy=1.2 -- high entropy signals ambiguous concept membership.
- Action: call `scan_tension(query)` to find which sub-cone dominates; use `deep_search` for disambiguation.

## 2. Implicit Signals From SearchHit

Every SearchHit carries four confidence dimensions:

| Field | Intuition | Action when high | Action when low |
|-------|-----------|-----------------|-----------------|
| `aperture` | concept breadth | use for exploration | use for precise retrieval |
| `local_entropy` | member diversity | expect broad coverage | expect tight cluster |
| `evidence_path_count` | multi-octave consensus | high confidence in relevance | single-scale hit, verify |
| `uncertainty_score` | 1 - normalized_score | re-query with rephrasing | proceed confidently |

A hit with `evidence_path_count >= 3` and `uncertainty_score < 0.2` is high-confidence retrieval.
A hit with `evidence_path_count == 1` and `uncertainty_score > 0.6` should be treated as weak evidence.

Example 1: synthesizing an answer
- Filter hits to `evidence_path_count >= 2` before grounding claims.
- Use `local_entropy` to detect when a concept spans multiple sub-domains (entropy > 1.0 = synthesize carefully).

Example 2: deciding when to stop searching
- If top 3 hits all have `uncertainty_score < 0.15`, retrieval has converged.
- If variance in `aperture` across top hits is high, the KB has conflicting hierarchy levels -- call `diagnose()`.

## 3. Energy and Accuracy Tradeoff

The `QueryPriority` enum controls the depth-vs-speed tradeoff:

- `HIGH`: multi-octave RRF fusion, tight MMR diversity (lam=0.3 favors diversity), slower.
- `MEDIUM`: default balanced (lam from settings, 2 octave levels).
- `LOW`: single coarse octave, fast, higher recall gap.

Use `HIGH` when evidence quality matters (final answer synthesis, claim grounding).
Use `LOW` when scanning many queries for routing (intent detection, topic identification).

Energy cost scales with octave depth. `total_energy` in `DiagnoseReport` reflects the KB's current
information density -- high energy with few nodes means over-compressed state (too many merges).

Example 1: agent planning loop
- Use `LOW` to score 20 sub-questions for routing.
- Use `HIGH` only on the 3 highest-scored to ground the answer.

Example 2: adaptive retrieval
- Start with `MEDIUM`; if top hit `uncertainty_score > 0.5`, retry with `HIGH`.
- If retry still returns `uncertainty_score > 0.5`, ingest more texts (KB gap detected).

## 4. Decision Trees for Retrieval

When to call which method:

```
receive query
 |
 +-- single concept, fast answer needed? -> search(k=3, priority=LOW)
 |
 +-- need causal chain or multi-hop? -> deep_search(k=5)
 |
 +-- exploring a domain for the first time? -> search(k=10, priority=HIGH)
 |       then: scan_tension(query) to find boundary concepts
 |
 +-- need diverse coverage? -> search(k=10, priority=HIGH)
         filter: keep hits where evidence_path_count >= 2
         then: build_context_pack() for token-budgeted context
```

When to call `consolidate()`:
- `DiagnoseReport.redundant_pairs > nodes // 4`
- `failure_mode == BOUNDARY_AMBIGUOUS`
- After bulk ingest (100+ texts) before a critical query

When to call `diagnose()`:
- Before starting a reasoning chain (baseline health check)
- When search returns < k hits on a non-empty KB
- When `uncertainty_score` is consistently high across multiple queries

## 5. Failure Modes and Recovery

`DiagnoseReport.failure_mode` encodes the KB health state:

**NONE**: KB is healthy; proceed normally.

**OUTSIDE_CONE**: Mean aperture > 1.2. Query concepts lie outside current cone coverage.
- Recovery: ingest focused texts on the target domain; re-run `diagnose()` after ingest.

**BOUNDARY_AMBIGUOUS**: High tension with many redundant pairs.
- Recovery: call `consolidate()`; check `ConsolidateReport.merges` to confirm resolution.
- If merges == 0 after consolidate, the tension is real semantic conflict -- surface to user.

**OVER_COMPRESSED**: Too few nodes per octave (aggressive merging collapsed hierarchy).
- Recovery: ingest diverse texts; use `deep_search()` instead of `search()` until nodes_per_octave > 3.

**OCTAVE_MISMATCH**: High aperture variance across octaves (entropy_divergence > 0.4).
- Recovery: use `deep_search()` for cross-octave queries; check encoder Matryoshka dims for gaps.

Example 1: OUTSIDE_CONE recovery
```python
report = kb.diagnose()
if report.failure_mode == FailureMode.OUTSIDE_CONE:
    kb.ingest(fetch_domain_texts(query))
    report = kb.diagnose()
    assert report.mean_aperture < 1.2
```

Example 2: BOUNDARY_AMBIGUOUS recovery
```python
report = kb.diagnose()
if report.failure_mode == FailureMode.BOUNDARY_AMBIGUOUS:
    cr = kb.consolidate()
    # cr.merges tells you how many redundant pairs were resolved
    if cr.merges == 0:
        # real semantic tension -- flag to reasoner
        pass
```

## 6. Multi-Scale Patterns

Matryoshka octaves give semiosis a coarse-to-fine retrieval ladder. Key patterns:

**Pattern: coarse routing -> fine grounding**
1. `search(query, k=1, priority=LOW)` -- identify the octave domain.
2. `deep_search(query, k=5)` -- trace from coarse cone into fine-grained evidence.

**Pattern: octave disagreement as ambiguity signal**
- `evidence_path_count == 1` means only one octave contained this node.
- Compare octave values across top hits: if all differ, the query spans multiple scales.
- Action: decompose query into sub-questions, one per dominant octave.

**Pattern: energy-efficient context packing**
- Use `build_context_pack(query, max_tokens=2000)` to get token-budgeted, overlap-deduped context.
- Prefer `context_pack` over raw `search()` when the downstream context window is constrained.
- `compress_context()` applies cone-energy folding to reduce a long context by ~30-50%.

Example 1: hierarchical question answering
- Coarse query "machine learning" -> hit with prefix=64 (root octave), aperture=0.8.
- Fine query "gradient descent convergence" -> hit with prefix=512, aperture=0.2.
- Use the root hit for framing, the fine hit for the actual answer.

Example 2: multi-hop reasoning
- Hop 1: `deep_search("cause of X")` -> evidence node A.
- Hop 2: `navigate(node_A_id, direction="flow_out")` -> downstream implications.
- Combine: trace from A to effect nodes using the flow gradient.

## 7. Mental Models

Three mental models for working with semiosis:

**The cone as confidence radius**: aperture is the uncertainty radius around a concept. A tight
cone (low aperture) is a high-confidence specific claim. A wide cone is a prior. Retrieval
returns the best-fitting cones to your query -- read aperture as how much to trust the specificity.

**Evidence paths as votes**: `evidence_path_count` is how many Matryoshka scales agreed this
node is relevant. Three octaves agreeing is stronger evidence than one. Treat it like a
multi-witness majority vote on relevance.

**Energy as information density**: `total_energy` from `context_energy()` measures how
redundant the top-k nodes are. High energy with few nodes = compressed, lossy representation.
Low energy = sparse, sparse KBs need more ingest. The sweet spot is medium energy with diverse
node coverage -- each node carries distinct information.

The three together: use `uncertainty_score` to decide confidence, `evidence_path_count` to
weight claims, and `total_energy` to decide whether more retrieval adds value or just noise.

## 8. Semantic Navigation: direction, trajectory, distance

Embedding space has measurable geometry. Use these primitives to navigate meaning rather than only retrieve it.

**Distance**: how far apart two concepts are at a given dimensionality.

```python
# All octaves at once -- pick the one with sharpest signal
dists = kb.semantic_distance("quantum", "classical", )       # dict[octave -> float]
p = kb.best_octave("quantum", "classical")                   # finds sharpest prefix
d = kb.semantic_distance("quantum", "classical", octave=p)   # scalar distance
```

Use cosine distance (default) for ranking; use_hyperbolic=True for manifold-faithful geodesic.

**Direction**: the vector pointing from one concept toward another.

```python
nodes = [n for n in kb._pipeline.store.all_nodes() if n.members]
sd = kb.compute_direction(str(nodes[0].id), str(nodes[1].id))
# sd.direction_vec: unit vector in octave subspace
# sd.magnitude: how far apart the centroids are
# sd.cosine_alignment: how co-directional the centroids are
```

**Direction search**: find what lies in a given direction from an anchor.

```python
steps = kb.direction_search("quantum", sd.direction_vec, k=5)
# steps: list[DirectionSearchResult], one entry per alpha step (e.g. alpha=[0.1, 0.5, 1.0, 2.0])
# each entry: .hits (ranked by alignment with direction_vec), .alpha, .alignment
for step in steps:
    print(step.alpha, step.alignment, [h.text for h in step.hits])
best = max(steps, key=lambda s: s.alignment)
```

Example: "what is more abstract than quantum?" -> compute_direction(quantum_node, abstract_node) -> direction_search("quantum", direction), then iterate `steps` to see how hits shift as alpha grows.

**Hierarchy folding**: map the downward intuition from a parent node.

```python
dirs = kb.fold_directions(str(parent_node.id))
# dirs: list of {child_id, direction_vec, magnitude, semantic_label}
# semantic_label from StubSummarizer or LLM summarizer if wired
```

Low magnitude = nearby sub-concept; high magnitude = distant branch. Use semantic_label to name directions without reading all member texts.

**Agentic inference loop**: when confidence is low, reflect and retry.

```python
result = kb.search_with_reflection(
    query,
    reflect_fn=lambda q: llm(f"Rephrase to highlight the key concept: {q}")
)
# result["original"]: initial hits
# result["reflected"]: hits after LLM rephrasing (empty if uncertainty_score <= 0.5)
# result["reflected_query"]: the rephrased string the LLM produced
```

Without reflect_fn, search_with_reflection still returns the original hits with the uncertainty flag -- agents can decide themselves whether to re-query.

**Worked example A -- conceptual comparison**:
1. kb.best_octave("entropy", "information") -> 128
2. kb.semantic_distance("entropy", "information", octave=128) -> 0.18 (close)
3. kb.semantic_distance("entropy", "gravity", octave=128) -> 0.71 (far)
4. Use this to rank which concepts are in the same region.

**Worked example B -- reasoning path**:
1. kb.search("explain quantum entanglement", k=3, priority=QueryPriority.LOW) -> route
2. top hit uncertainty_score=0.65 -> call search_with_reflection with reflect_fn
3. LLM rephrases -> "quantum nonlocal correlation" -> retry
4. New top hit uncertainty_score=0.22 -> use this answer

## 9. Advanced Primitives: Compression, Recursion, and Analogy

### 9.1 Info-bottleneck compression

compress_hierarchy(query, max_nodes) retains only the highest-relevance nodes.
Use before build_context_pack() when context window is limited.

  result = kb.compress_hierarchy("neural architectures", max_nodes=5)
  # result.info_retained_ratio < 0.5 means sparse KB; consider ingesting more

### 9.2 Manifold complexity sensing

sense_complexity(query) estimates intrinsic dimensionality of query neighborhood.
Maps to RLM: task complexity = manifold dimension.

  mc = kb.sense_complexity(query)
  # constant -> 0-d, single concept, use octave 64
  # linear -> 1-d, use 128
  # quadratic -> 2-d, use 256
  # exponential -> high-d, use 512+
  hits = kb.search(query, k=10)  # use mc.suggested_octave when available

### 9.3 Analogy reasoning

find_analogy(text_a, text_b, text_c) solves A:B::C:X via direction arithmetic.
embed(c) + (embed(b) - embed(a)) is the target; nearest nodes to target are returned.

  result = kb.find_analogy("cat", "animal", "dog")
  # result.hits are nodes near "what dog is to cat as animal is to cat"

### 9.4 Concept boundary analysis

concept_boundary(node_a, node_b) finds the decision surface between two clusters.
Low margin (< 0.1) -> BOUNDARY_AMBIGUOUS FailureMode expected.

  cb = kb.concept_boundary(nid_a, nid_b)
  if cb.margin < 0.1:
      # expect ambiguous retrieval between these two concepts

### 9.5 Entropy management

entropy_dispel(entropy_ceiling) auto-prunes nodes where entropy proxy > ceiling.
High entropy = noise or over-general cluster. Call after bulk ingest.

  report = kb.entropy_dispel(entropy_ceiling=1.5)
  # report.dispelled_ids are pruned node ids

### 9.6 Energy-aware fold budget

fold_budget(query, max_tokens, candidates) greedily selects texts by query relevance.
Implements RLM fold_budget(K, candidates): prefer high-relevance texts under token limit.

  result = kb.fold_budget("neural networks", 500, long_candidate_list)
  # result.included are the selected texts

### 9.7 Attention weight proxy

attention_score(node_id, query) approximates Transformer attention weight.
Scaled dot-product: softmax(dot(q,k)/sqrt(d)) over all nodes.

  a = kb.attention_score(node_id, "machine learning")
  # a.weight in [0,1]; high = node likely attended by model given this query

## Section 10: Agentic Inference and Energy Descent

Four methods for iterative refinement, categorical reasoning, activation-based embedding, and energy-guided traversal.

### 10.1 agentic_reflect

Runs LLM-in-the-loop recursive KB query refinement: each round searches, observes uncertainty, and lets the LLM refine the query before the next round. Use when the initial query is vague or returns high `uncertainty_score`; three rounds usually converge.

```python
steps = kb.agentic_reflect(
    query="distributed systems fault tolerance",
    llm_fn=lambda q: llm(f"Refine this KB query to be more precise: {q}"),
    max_rounds=3
)
# steps: list[ReflectStep]; each has .query, .hits, .uncertainty_score
# stop early when steps[-1].uncertainty_score < 0.2
for step in steps:
    print(step.query, step.uncertainty_score)
```

### 10.2 categorical_parent_score

Ranks parent nodes by summarizing their member texts and comparing `embed(summary)` to `embed(query)`. Use to find the most semantically apt container or category for a concept -- prefer this over raw `search()` when you need a parent, not a leaf.

```python
hits = kb.categorical_parent_score("attention mechanism", k=5)
# hits: list[CategoricalParentHit]; each has .node_id, .summary, .score
# highest score = best-fitting parent category for the query
best = hits[0]
print(best.node_id, best.score, best.summary)
```

### 10.3 activation_embed

Returns an embedding as a plain float list, routing through an `ActivationPredictor` if one is fitted, otherwise falling back to the standard encoder. Use on the NLA-style activation-prediction path where you need raw floats rather than a store-internal vector.

```python
vec = kb.activation_embed("transformer self-attention")
# vec: list[float] at the root encoder dimension
# plug into external tools, distance metrics, or visualizers directly
import numpy as np
arr = np.array(vec)
```

### 10.4 energy_gradient_search

Follows the minimum-energy path through the cone hierarchy from the query embedding, stepping toward lower-energy nodes at each hop. Returns a dict with `steps`, `terminal` (the lowest-energy node reached), and `energy_drop` (total energy reduction). Use for efficient concept localization without brute-force search across all nodes.

```python
result = kb.energy_gradient_search("causal inference", max_steps=10)
# result["steps"]: list of node ids traversed
# result["terminal"]: id of the lowest-energy node reached
# result["energy_drop"]: float; large drop = strong localization
if result["energy_drop"] < 0.05:
    # shallow gradient: concept may be spread across multiple cones
    # fall back to deep_search()
    hits = kb.deep_search("causal inference", k=5)
```
