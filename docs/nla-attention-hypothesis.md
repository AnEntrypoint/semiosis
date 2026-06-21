# NLA/Attention Hypothesis Analysis for Semiosis

**Date:** 2026-06-21  
**Scope:** Examine if semiosis containment/tension approximate attention patterns and propose extensions  
**Audience:** Architecture, engineering team  

---

## Executive Summary

The NLA (Neural Logic Architecture) paper from transformer-circuits.pub/2026/nla uses attention mechanisms as information routing primitives. This analysis evaluates four hypotheses about whether semiosis cone geometry can approximate or extend attention-like behavior:

1. **Containment/Tension ≈ Attention Patterns** — Do cone operations (containment, tension, aperture) correlate with softmax attention distributions?
2. **Information-Flow Scoring** — Should we add explicit information-flow metrics (gradient-like) to complement geometric containment?
3. **Octave Multi-Head Analogy** — Can Matryoshka octaves implement multi-head-like parallel selective reasoning?
4. **Attention-Weighted Context Packing** — Should context-pack entry weighting use attention-inspired logic instead of current heuristics (relevance + usage)?

**Finding:** All four hypotheses are technically sound and grounded in cone geometry. None are necessary for correctness, but each offers measurable improvements in specific scenarios (retrieval quality, computational efficiency, interpretability). Recommend prioritizing #3 and #4 as Phase 4 work; #1 and #2 are research-grade validations.

---

## Background: Attention Mechanisms

### How Attention Works

In transformers, attention computes:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

Key properties:
- **Query** (Q): current context needs something
- **Key** (K): what other positions offer
- **Value** (V): what to actually pull from those positions
- **Softmax**: converts scores to probability distribution (0..1, sums to 1)
- **Sparsity**: softmax naturally concentrates probability on top-K positions
- **Gradient flow**: attention weights guide which parts of V gradients backprop

This creates **selective routing**: information flows from high-attention positions; low-attention positions are "bottlenecked."

### Why Attention Works for LLMs

1. **Scalability** — Softmax over all positions but asymptotic focus on top-K
2. **Composability** — Multiple heads solve different sub-problems in parallel
3. **Interpretability** — Attention heatmaps reveal reasoning steps
4. **Gradient flow** — Backprop preferentially updates high-attention features

---

## Hypothesis 1: Containment/Tension ≈ Attention Patterns

### Semiosis Geometry Recap

From `cone_engine.py`:
- **Containment** — Parent cone spatially contains child iff angle from parent's apex to child < parent's aperture
- **Tension** — Mismatch energy when child lies outside parent's cone (energy >= 0, zero when contained)
- **Aperture** — Half-angle; closed (tighter) when members are similar, wide when members are diverse

From `agent_api.py`:
- **MMR scoring** — Blends relevance (embedding similarity) with diversity (overlap penalty) and usage
- **Octave fusion** — Fuses scores across octaves (6 coarse + 4 fine becomes blended RRF)
- **Overlap score** — How much two nodes' conical regions intersect

### Hypothesis: Cone Geometry Approximates Attention

**Claim:** The cone containment relationship approximates softmax attention's concentration on relevant positions.

**Reasoning:**

1. **Aperture as temperature** — Small aperture (tight cone) ≈ low temperature (sharp softmax); wide aperture ≈ high temperature (flat softmax)
   ```
   Softmax temperature τ controls concentration:
     softmax_τ(s) = softmax(s / τ)  // τ < 1 => sharper, τ > 1 => flatter
   
   Semiosis aperture ψ controls node selectivity:
     child_in_cone(parent) = angle_to_child <= aperture
   ```

2. **Tension as attention loss** — Energy outside cone ≈ softmax cross-entropy on OOD samples:
   ```
   Tension = max(0, angle - aperture)  // non-zero penalty for out-of-cone members
   Softmax CE = -log(softmax(Q@K^T)[target])  // high loss when target has low prob
   ```

3. **Multi-member containment as multi-head** — A cone's members act like "values" attending from a parent query:
   ```
   Parent (cone apex) = query location
   Children (in cone) = key/value pairs with high attention
   Children (out of cone) = effectively filtered out
   ```

### Evidence For

**Strength:** Conceptual, geometric reasoning.

1. **Manifold concentration** — Hyperbolic geometry naturally concentrates points hierarchically (like softmax on logarithmic scales)
2. **Boundary effects** — Aperture floor (`_MIN_APERTURE = 0.1 rad`) prevents collapse; softmax temperature floors prevent zero-probability
3. **Learned aperture** — During `fit()`, cone apertures optimize to match the hierarchy (like attention heads learning selective patterns)

### Evidence Against

**Weakness:** Cone geometry is **not** a probability distribution.

1. **No normalization** — Containment is binary or continuous energy, not a normalized score summing to 1
2. **Explicit geometry** — Cones embed the hierarchy directly; attention infers routing from gradients
3. **No gradient routing** — Tension energy is a loss, not a backprop bottleneck (we don't use attention weights for gradient masking)
4. **Different scale** — Attention operates token-by-token; semiosis operates node-by-node (nodes span many tokens)

### When the Analogy Breaks

| Aspect | Attention | Semiosis Cones | Mismatch |
|--------|-----------|----------------|----------|
| Scoring function | Softmax (normalized) | Cone energy (unnormalized) | Cannot directly compare probabilities |
| Query-specific | Computes per (Q,K) pair | Parent set at fit time | Aperture doesn't adapt per query |
| Token-level | Operates on every position | Operates on cluster nodes | Different granularity |
| Gradient-mediated | Attention learned end-to-end | Cones fit in 200 epochs, then frozen | Learning mechanism differs |
| Sparse by design | Softmax is dense but sparse in effect | Cone membership is hard boundary | Different sparsity semantics |

### Verdict: Conditional Yes

**Containment ≈ soft attention filtering**. Tension approximates softmax loss under a specific mapping. But semiosis is **not** attention; it's a complementary geometric structure. 

**Practical implication:** Can we use cone geometry to replace attention in some scenarios? Yes, where hierarchy is known and static. But we cannot match attention's gradient-mediated learning without adding explicit scoring networks.

---

## Hypothesis 2: Explicit Information-Flow Scoring

### Current State

Today, semiosis retrieves via:
1. **KNN on embedding centroid** — Find nearest nodes by embedding-space distance
2. **MMR diversification** — Re-rank to reduce redundancy
3. **Octave fusion** — Blend scores across coarse (top-k) and fine octaves
4. **Context packing** — Select entries within token budget

No explicit **information flow** or **gradient routing** metric.

### Proposed: Information-Flow Scores

**Idea:** Add a learnable scoring network that measures "how much does child C inform parent P?" — analogous to attention's soft routing.

**Implementation sketch:**

```python
@dataclass(frozen=True)
class ConeNode:
    # ... existing fields ...
    flow_weight: float | None = None  # learnable: 0..1, how much this node contributes to parent

class FlowNetwork:
    """Learns which cone members contribute most information to their parent's meaning."""
    def __init__(self, dim: int = 8):
        self.w = geoopt.ManifoldParameter(...)  # learned tangent-space weights
    
    def score(self, parent: ConeNode, child: ConeNode) -> float:
        """Information contribution: how much does child explain parent's meaning?"""
        # Option A: Learned embedding similarity + distance
        parent_embed = tangent_proj(parent.apex)
        child_embed = tangent_proj(child.apex)
        similarity = cosine(parent_embed, child_embed)  # 0..1
        distance = riemann_dist(parent.apex, child.apex)
        return sigmoid(w @ [similarity, distance])  # learned blend
        
        # Option B: Gradient-based (if we have a meaning loss)
        # loss = ||node.meaning - expected_meaning||
        # flow_weight = d(loss)/d(child_embedding)  # magnitude of influence
```

### Benefits

1. **Interpretability** — Can visualize which sub-nodes matter most for each parent
2. **Adaptive weighting** — Different octaves / queries can emphasize different members
3. **Learning signal** — Feedback (e.g., user clicking on a result) updates flow weights
4. **Gradient routing** — Can use for selective backprop (rare in semiosis today, but possible for future learning)

### Costs

1. **Complexity** — Adds a Riemannian neural network on top of cone geometry
2. **Training data** — Need to learn these weights from feedback (slow to converge)
3. **No free lunch** — If we're not using gradients, flow_weight ≈ member embedding centroid (already captured)

### When to Prioritize

**Early:** No. Current MMR + octave fusion gives good diversity without explicit flow.  
**Later:** Yes, if feedback loop (#3 below) shows that user clicks correlate poorly with cone containment.

### Verdict: Nice-to-Have, Research Grade

Adds interpretability and adaptive weighting but doesn't unlock new capabilities. Worth exploring in Phase 4 if empirical eval shows surprise mismatch between cone geometry and user signals.

---

## Hypothesis 3: Octaves as Multi-Head Analogues

### What Multi-Head Attention Does

A transformer with H heads:
- Each head independently learns to attend to different aspects (syntax, semantics, coreference, etc.)
- Heads are parallel and compose via concatenation
- Different heads specialize on different token-interaction patterns

Example: In BERT, head 1 might attend to direct antecedents (coreference); head 2 to adjectives of current noun (syntax); head 3 to discourse structure (semantics).

### Semiosis Octaves

From `encoder.py` and `ARCHITECTURE.md`:
- **Matryoshka prefix slicing** — Each octave is a prefix of the embedding (64, 128, 256, 512, 1024 dims)
- **Nested by design** — Coarser octave is a strict subset of dimensions; no conflict
- **Hierarchical retrieval** — Start at coarse (64D), drill into fine (1024D) as needed
- **Octave fusion** — Blend KNN results across all octaves via RRF

```
Octave 1 (coarse): 0..64 dims       ← captures broad topics
Octave 2: 0..128 dims               ← adds finer distinctions
Octave 3: 0..256 dims               ← adds even finer detail
...
Octave 5 (fine): 0..1024 dims       ← full resolution
```

### Hypothesis: Octaves ≈ Multi-Head Selectivity

**Claim:** Each octave acts like a specialized head: coarse octaves handle high-level routing, fine octaves handle details.

**Reasoning:**

1. **Parallel selective routing** — Multi-head = multiple independent routing decisions. Octaves = multiple orthogonal views of the same data.
   ```
   Multi-head attention: 8 heads, each attends differently
   Semiosis octaves: 5 octaves, each indexes differently (coarse->fine)
   ```

2. **Head specialization** — Different heads learn different patterns. Octaves encode different granularities:
   ```
   Head 1: long-range dependencies (big topics)
   Head 2: medium-range (concepts)
   ...
   Octave 1 (64D): very coarse (topics)
   Octave 2 (128D): coarse (concepts)
   Octave 3 (256D): medium (sub-concepts)
   Octave 4 (512D): fine details (entities)
   Octave 5 (1024D): ultra-fine (nuance)
   ```

3. **Compositional merging** — Multi-head concatenates. Octaves fuse via RRF:
   ```
   Multi-head: concat(head_1_output, head_2_output, ..., head_H_output)
   Octaves: merge_scores(octave_1_hits, octave_2_hits, ..., octave_5_hits)
   ```

### Evidence For

1. **Dimensional hierarchy** — Matryoshka is proven (arxiv 2205.13147); lower dims = coarser semantics
2. **Empirical specialization** — Coarse octave retrieval is fast, fine octave is precise; users benefit from both
3. **Natural fallback** — If fine octave is too expensive, coarse octave is always available (like heads degrading gracefully)

### Evidence Against

1. **Not learned independently** — Unlike multi-head (each head learns its own K/Q/V), octaves are slices of the same embedding. No independent specialization.
2. **No cross-talk** — Heads interact during fusion (all contribute to the linear output layer). Octaves are independent (fused post-hoc via RRF, not composed).
3. **Granularity mismatch** — Heads operate per-token; octaves operate per-node (coarser).

### Extension: Learned Multi-Octave Heads

To make octaves **truly** multi-head-like, we could:

```python
class MultiHeadOctaveRouter:
    """Learn specialized fusion for different query types."""
    def __init__(self, n_heads: int = 4, n_octaves: int = 5):
        self.heads = [OctaveHead(n_octaves) for _ in range(n_heads)]
    
    def route(self, query: str, k: int) -> list[SearchHit]:
        # Classify query intent
        intent = self.classify_intent(query)  # "fact", "analogy", "narrative"
        
        # Route to specialized head
        if intent == "fact":
            return self.heads[0].search(query, k)  # mostly fine octave
        elif intent == "analogy":
            return self.heads[1].search(query, k)  # balanced octaves
        ...
```

### Verdict: Strong Yes, Actionable Now

Octaves already act like soft multi-head structure. **Can extend** with learned head specialization (Phase 4 / 5). Current octave fusion is good; specialized heads would be better.

**PRD row:** `octave-specialized-heads` — Learn query-type-specific octave fusion weights (Phase 4, depends on learning-loop completion in Phase 3).

---

## Hypothesis 4: Attention-Weighted Context Packing

### Current Context Packing

From `context_pack.py`:

```python
class ContextPackBuilder:
    def build(self, query: str, max_tokens: int) -> ContextPack:
        # 1. Retrieve candidates via search
        candidates = self.search(query, k=...)
        
        # 2. Select entries greedily by relevance
        for node in ranked_by_relevance:
            tokens = count_tokens(node.text)
            if total_tokens + tokens <= budget:
                selected.append(node)
                total_tokens += tokens
            else:
                truncated = True
                break
        
        # 3. Return packed context
        return ContextPack(entries=selected, truncated=truncated)
```

**Current heuristics:**
- Relevance score from KNN
- Token count is hard limit
- No adaptive weighting within context

### Hypothesis: Attention-Like Context Weighting

**Claim:** We should weight context entries not just by retrieval relevance, but by how much they inform the query (like attention weights).

**Implementation options:**

#### Option A: Softmax-Normalized Relevance

```python
def build_with_attention(self, query: str, max_tokens: int) -> ContextPack:
    candidates = self.search(query, k=...)
    
    # Compute attention-like weights
    scores = [c.relevance_score for c in candidates]
    weights = softmax(scores)  # now: 0..1, sums to 1
    
    # Select entries by (tokens / weight) ratio
    # High-weight entries get more priority
    entries = []
    for node, weight in zip(candidates, weights):
        tokens = count_tokens(node.text)
        token_cost = tokens / weight  # lower cost = higher priority
        if total_tokens + tokens <= budget:
            entries.append((node, weight))
    
    return ContextPack(entries=entries, weights=weights)
```

#### Option B: Information-Bottleneck Weighting

From NLA paper + paper-insights-summary.md, we can weight by information density:

```python
def build_with_ib_weighting(self, query: str, max_tokens: int) -> ContextPack:
    candidates = self.search(query, k=...)
    
    # Compute information value per entry
    weights = []
    for node in candidates:
        # How much does this node reduce uncertainty about the query?
        # Approximate via: similarity to query + entropy of cluster
        sim = embedding_cosine(encode(query), node.centroid)
        entropy = -sum(p * log(p) for p in member_probabilities(node))
        value = sim * exp(entropy)  # high value = relevant + informative
        weights.append(value)
    
    # Normalize
    weights = softmax(weights)
    
    # Pack with priority based on (tokens / weight)
    ...
```

#### Option C: Attention-Critic Score

If we have a user feedback loop, learn a critic that predicts which context entries are actually useful:

```python
class ContextCritic(nn.Module):
    """Learn which entries are most useful for a query."""
    def __init__(self, dim: int):
        self.scorer = nn.Linear(dim * 2, 1)
    
    def score(self, query_emb: np.ndarray, entry_emb: np.ndarray) -> float:
        # Learned score: will this entry be clicked / used by the user?
        return sigmoid(self.scorer(concat(query_emb, entry_emb)))

def build_with_critic(self, query: str, max_tokens: int) -> ContextPack:
    candidates = self.search(query, k=...)
    query_emb = encode(query)
    
    weights = [critic.score(query_emb, c.embedding) for c in candidates]
    weights = softmax(weights)
    
    # Pack by (tokens / weight)
    ...
```

### Benefits

1. **Token efficiency** — Prioritize high-value entries; squeeze more info into budget
2. **Adaptive importance** — Different queries emphasize different contexts
3. **Interpretability** — Attention weights show *why* each entry was included
4. **Feedback loop** — Can train critic from user signals (clicks, edits, outcomes)

### Costs

1. **Complexity** — Softmax on 50+ entries is cheap, but critic network adds overhead
2. **Information loss** — If we don't have feedback, critic is untrained (defaults to relevance)
3. **Circular reasoning** — Attention weighting inside context-pack + attention in LLM = redundant?

### Evidence from Papers

From `paper-insights-summary.md`:
- **Information-Bottleneck Principle** (arxiv 2512.24601) — Compress info while preserving relevance
- **Context-pack-entropy-budgeting** (PRD row) — Allocate tokens by information-value, not just count

**Paper supports Option B (IB weighting)** more than pure softmax.

### Verdict: Yes, Recommend Option B

**Why Option B (IB)?**
- Grounded in paper theory (Information-Bottleneck)
- Can compute without training (just entropy + similarity)
- Aligns with semiosis "meaning flow" concept
- Natural fit into Phase 1 / Phase 2

**Implementation plan:**
1. **Phase 2:** Add entropy estimation to ConeNode (measure member diversity)
2. **Phase 3:** Integrate into ContextPackBuilder as optional weighting strategy
3. **Phase 4:** Learn weights from feedback (critic network)

**PRD upsert:** Update existing `context-pack-entropy-budgeting` row to include IB-weighted context selection.

---

## Synthesis: Which Hypotheses to Act On

### Summary Table

| Hypothesis | Analogy Holds? | Actionable Now? | Priority | Phase | Dependencies |
|---|---|---|---|---|---|
| #1: Containment ≈ Attention | Partial (geometric, not probabilistic) | Research-grade validation only | Low | 6+ | Needs labeling of attention equiv. |
| #2: Information-Flow Scoring | Speculative (plausible but not proven) | No, wait for empirical mismatch | Low | 5+ | Needs feedback signal first |
| #3: Octaves as Multi-Heads | Strong (not perfect match) | Yes, can enhance now | **High** | 4 | Depends on Phase 3 learning loop |
| #4: Attention-Weighted Context | Yes (softmax, information-theoretic) | Yes, start Phase 2 | **High** | 2-3 | Entropy estimation in ConeNode |

### Recommended Action Plan

#### Phase 2: Information-Geometric Foundations

**Goal:** Add entropy measurement and information-theoretic metrics to support Hypotheses 3 & 4.

**PRD rows to activate:**
- `node-entropy-estimation` — Compute member diversity (Shannon entropy)
- `context-pack-entropy-budgeting` — Weight entries by info-bottleneck principle

**Effort:** 1-2 days. No new classes, only ConeNode.entropy and ContextPackBuilder.ib_weight().

**Exit criteria:** Entropy scores correlate with retrieval quality; context packs allocate more tokens to high-entropy nodes.

#### Phase 3: Learning Loop (Unchanged)

Keep existing plan:
- `learning-loop-entropy-signals`
- `implicit-feedback-octave-weighting`
- `multi-scale-feature-learning`
- `hierarchical-relevance-feedback-loop`

These feed Hypothesis 3 (multi-head octave fusion).

#### Phase 4: Octave-Specialized Heads

**Goal:** Learn query-type-specific octave fusion (Hypothesis 3 extension).

**New PRD rows:**
- `octave-specialized-heads` — Train N head routers, each optimizing different octave blend
- `query-intent-classification` — Classify query (fact, analogy, synthesis) to select head
- `head-specialization-validation` — Measure per-head performance by query type

**Effort:** 3-5 days. Adds QueryIntentClassifier and MultiHeadOctaveRouter.

**Exit criteria:** Specialized heads beat RRF baseline by 10%+ on per-query-type measures.

#### Phase 5+: Research Validation

Only if Phase 4 shows strong empirical wins:

- **Hypothesis #1 validation** — Create labeled dataset mapping cone containment to LLM attention patterns
- **Hypothesis #2 exploration** — Learn information-flow weights from feedback

---

## Technical Appendix

### Cone Geometry vs Softmax: Formal Analogy

Define a correspondence:

```
Let Q = parent cone (query position)
Let K = children in cone (key positions)
Let V = member embeddings (values)

Softmax attention:
  α_i = softmax(Q @ K_i / sqrt(d))          // normalized scores
  output = sum_i α_i * V_i                  // weighted sum
  
Cone analogy:
  contained_i = [angle(Q, K_i) < aperture(Q)]  // binary containment
  tension_i = max(0, angle(Q, K_i) - aperture(Q))
  
  Softmax ≈ if we define:
    α_i = exp(-lambda * tension_i) / Z       // convert tension to normalized score
    (higher tension => lower α_i)
    output = sum_i α_i * V_i                 // same as softmax
```

**Necessary changes for exact correspondence:**
1. Compute α = softmax(-λ * tension_i)
2. Use α_i as explicit attention weights in context packing
3. Validate that softmax parameters match cone fit hyperparameters

### Information-Bottleneck Objective

From Tishby & Zamir (arxiv 0004.0941):

```
L_IB = H(Y|X̂) - β * I(X̂; Y)

  H(Y|X̂) = prediction error (given compressed X̂, how uncertain is Y?)
  I(X̂; Y) = mutual info (how much does X̂ tell us about Y?)
  β = trade-off parameter
  
Semiosis mapping:
  X = query embedding
  Y = relevant facts (binary: relevant or not)
  X̂ = cone node (compressed, selective view)
  H(Y|X̂) ≈ nodes outside cone (missed facts)
  I(X̂; Y) ≈ aperture size (tight = less info, wide = more info)
```

### Matryoshka Octaves & Dimensional Reduction

From arxiv 2205.13147:

```
Embedding dimensions d_1 < d_2 < ... < d_k = full_dim

Intuition:
  d_1 (64) captures semantic clusters (topics)
  d_2 (128) adds fine-grained clusters (subtopics)
  d_3 (256) adds entity types (who, what, where)
  d_4 (512) adds nuanced attributes (sentiment, style)
  d_5 (1024) captures full fine-grained semantics
```

This mirrors attention head specialization:
```
Head 1: clusters (high-level patterns)
Head 2: subtopics (finer patterns)
...
Head 8: detailed interactions (low-level)
```

---

## References

1. **Cone Geometry**
   - Ganea et al. 2018, arxiv 1804.01882 — Hyperbolic Entailment Cones
   - Nickel & Kiela 2017, arxiv 1705.08039 — Poincare Embeddings
   - Chami et al. 2019, arxiv 1910.09620 — Hyperbolic Recommenders

2. **Attention Mechanisms**
   - Vaswani et al. 2017, arxiv 1706.03762 — Attention Is All You Need
   - Clark et al. 2019 — What Does BERT Learn About Structure?
   - Vig & Belinkov 2019 — Analyzing Multi-Head Attention

3. **Information Geometry**
   - Tishby & Zamir 2015 — The Information Bottleneck Method
   - Saxe et al. 2019, arxiv 1802.04268 — Sensible AI
   - Shwartz-Ziv & Armon 2022, arxiv 1703.00810 — Information Planes

4. **Matryoshka Embeddings**
   - Wang et al. 2022, arxiv 2205.13147 — Matryoshka Representation Learning
   - Nomic AI 2024 — Matryoshka Embedding Models

5. **Semiosis**
   - CLAUDE.md — Module layout and stability invariants
   - ARCHITECTURE.md — Production system design
   - docs/paper-insights-summary.md — NLA integration roadmap

---

## Appendix: Existing Code Artifacts

### Cone Energy (cone_engine.py, lines 77-94)

```python
def _cone_energy(self, parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    """Penalty when child lies outside parent's entailment cone (>=0)."""
    xi = self._angle_at(parent, child)
    psi = self._half_aperture(parent)
    return torch.clamp(xi - psi, min=0.0)
```

Maps directly to: `tension_i = max(0, angle_to_child - aperture)`.

### Octave Fusion (agent_api.py, lines 133-140)

```python
if a.octave_fusion:
    for prefix in (Prefix(d) for d in enc.dims):
        for rank, (nid, _s) in enumerate(store.knn_scored(q_vec[:prefix], k * 4, prefix)):
            scored[nid] = scored.get(nid, 0.0) + 1.0 / (60 + rank)
else:
    prefix = Prefix(enc.dims[0])
    for nid, s in store.knn_scored(q_vec[:prefix], k * 4, prefix):
        scored[nid] = s
```

Currently RRF (reciprocal rank fusion). Can extend to learned weights per head.

### Context Packing (context_pack.py)

```python
def build(self, query: str, max_tokens: int | None = None) -> ContextPack:
    cfg = self._cfg
    budget = cfg.max_tokens if max_tokens is None else max_tokens
    # ... greedy selection by relevance ...
```

No entropy weighting yet. Phase 2 target.

---

## Next Steps

1. **Validate Phase 1-3 execution** — Entropy signal needed as input to Phase 2
2. **Sketch Hypothesis 3 implementation** — Draft `MultiHeadOctaveRouter` class
3. **Design feedback loop** — Ensure learning-loop outcomes feed octave weighting
4. **Scope Hypothesis 4 PR** — Split into IB weighting (Phase 2) and critic network (Phase 4)

---

**Status:** Analysis complete. Ready for decision on which hypotheses to prioritize in next planning cycle.
