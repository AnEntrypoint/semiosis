# SEMIOSIS: PAPER-TO-IMPLEMENTATION HIERARCHICAL STRUCTURES MAPPING

## (1) EXPLICIT TREE/CONE STRUCTURES IN PAPERS & CODE

### A. Entailment Cones (Ganea 2018, 1804.01882 + HypCBM 2026)
SOURCE: ARCHITECTURE.md identifies "One structure | entailment cones = hierarchy + overlap + relations"

In code (core/cone_engine.py):
- HyperbolicConeEngine class implements Lorentz-manifold cones
- Each node = ConeNode(apex, aperture, members)
  * apex: LorentzVec point on Lorentz manifold (hyperbolic geometry)
  * aperture: half-opening angle theta (radians); child inside parent iff angle <= aperture
  * members: tuple of PhraseId objects contained in this cone
- Hierarchy encoded via containment: contains(parent, child) -> margin > 0
- Core invariant: parent cone contains child cone if angle to child <= parent aperture
- Three derived relations from cone math:
  1. Entailment: asymmetric containment (one fully inside the other)
  2. Overlap: symmetric containment (both directions positive, ambiguous)
  3. Independence: no containment (disjoint cones)

### B. Matryoshka Octaves (arxiv 2205.13147)
Prefix-nested embedding dimensions create a hierarchy of scales.

In code:
- EncoderSettings.octaves = (64, 128, 256, 512, 1024) [ascending, nested]
- Each octave = prefix slice of the full embedding
- Prefix constraint: octaves[i] < octaves[i+1] (nesting)
- Retrieval per-octave: query sliced to Prefix k, then knn searched
- ConeNode stores .prefix field to track which octave it belongs to
- ClusterTree.prefix marks the embedding prefix for that octave's clustering

Hierarchy flow: coarse (64D) -> fine (512D, 1024D)
- RecursiveAnswerEngine.descend() walks octave_idx coarse-to-fine
- At each octave, beam search finds frontier nodes
- Stops at finest octave or when aperture < min_aperture_stop (0.1 rad)

### C. Hierarchical Clustering (BERTopic/TaxoGen style)
In code (interfaces.py, cone_engine.py):
- ClusterTree: edges (parent-child pairs) + assignments (phrase -> node)
- HierarchicalClusterer protocol: fit(vecs, prefix) -> ClusterTree
- ConeEmbedder: takes ClusterTree, fits apexes on Lorentz manifold
- Result: tree structure lifted into hyperbolic space
  * parent cone naturally contains child cone by aperture math
  * containment is learned via margin loss (triplet: pos edges < margin, neg edges > margin)

---

## (2) INFORMATION FOLDING / COMPRESSION

### A. Cone Energy & Semiotic Distancing
Core primitive: cone_energy(parent, child) = max(0, angle - parent_aperture)
- "Penalty when child lies outside parent's entailment cone"
- Forms basis for semiotic distancing (tension, flow, energy in cone_engine.py lines 189-300)

### B. Context Rot Mitigation (context_pack.py)
"Token-budgeted context-pack builder; mitigates context rot via dedup and semiotic distancing"

Folding mechanism:
1. Deduplication: overlap_score(node_a, node_b) > threshold => drop one
   - Uses containment asymmetry to rank which survives
2. Distance-based summarization (lines 134-143):
   - Distant cones (overlap_score < distance_summary_threshold) use _summary_text()
   - Summary = node.digest OR node.label OR "[cluster of N: ...]"
   - Replaces full member texts with compact digest
   - Saves tokens by compressing distant clusters to cluster marker

Flow:
- Over-fetch k=max(max_dedup_candidates, 1) nodes
- Keep unique by overlap (dedup)
- For distant nodes: emit summary instead of full text
- Budget-aware _pack(): prioritize by relevance, truncate if needed

### C. Layered Memory (semiotic_memory.py)
4-layer compression from most specific to most general:
1. Layer 1 (facts): pinned explicit facts, never summary
2. Layer 2 (cone summaries): distant clusters as digests, near clusters as full members
3. Layer 3 (session): metadata (query count, recent foci, active octave)
4. Token-bounded assembly: fit all 4 layers in budget_tokens (default 2048)

Decay weight: exp(-recency_lambda * age) => older facts downweighted
Clock-based LRU: facts evicted when > max_pinned (default 64)

### D. Recursive Octave-Descent (recursive.py)
Compression via hierarchical search:
- decompose(query) splits "A and B" into independent subqueries
- descend() recurses octave_idx: coarse -> fine
  * At each level, knn (over-fetch) then narrow via containment gates
  * gated = [child | contains(node, child) > 0.0]
  * Stops early when aperture < min_aperture_stop (fine-grained enough)
- Merges sub-answers: evidence_node_ids = union of all recursive results
- Result: walk the hierarchy top-down, prune low-relevance branches, stop early

---

## (3) DIMENSIONS & EMBEDDING SPACES

### A. Cone Space Dimension
ConeSettings.dim = 8 (small)
- Random init on Lorentz manifold: random_normal((n_nodes, 8+1), std=0.1)
- "+1" is the time-like coordinate (Lorentz metric signature)
- Full Lorentz vector: apex[0] = time-like, apex[1:] = spatial (8D)
- Lorentzian inner product: <a,b>_L = -a[0]*b[0] + sum(a[i]*b[i])
- Manifold curvature: k=1.0 (geoopt.Lorentz parameter)

### B. Embedding Space Dimension per Octave
- Base model: sentence-transformers/all-MiniLM-L6-v2 (384D full embedding)
- Octaves slice this into prefixes:
  * Octave 0: embedding[:64] (64D)
  * Octave 1: embedding[:128] (128D)
  * Octave 2: embedding[:256] (256D)
  * Octave 3: embedding[:512] (512D)
  * Octave 4: embedding[:1024] (1024D) -- clamped to min(1024, 384)
- Each octave coexists in store (prefix-namespaced node ids: root@64, root@128, ...)

### C. Token & Distance Metrics
- Token counter: heuristic ~4 chars per token (HeuristicTokenCounter)
- Hyperbolic distance: manifold.dist(apex_a, apex_b) on Lorentz manifold
- Geodesic distance formula: arcosh(<apex_a, apex_b>_L) when metric allows
- Flow weight: containment_asymmetry / (geodesic_distance + eps)
  * Units: (dimensionless margin) / (manifold distance)
  * Sign indicates direction: >0 = generalization, <0 = specialization

---

## (4) CONNECTION TO INFORMATION-THEORETIC ENTROPY

### A. No explicit entropy formulation in current code
The codebase does NOT currently compute Shannon entropy, KL divergence, or mutual information.

### B. Implicit information-theoretic principles

1. **Compression via hierarchy**: octave prefixes implement progressive disclosure
   - Start with coarse (64D), refine to fine (1024D)
   - Early layers retain most task-relevant structure
   - Matches information-bottleneck principle (compress to relevance)

2. **Tension as information redundancy**
   - tension(a, b) = overlap_score - |containment_asymmetry|
   - High overlap + low asymmetry = ambiguous, redundant information
   - Low tension = clear hierarchy (low redundancy)

3. **Context energy as semiotic spread**
   - context_energy = sum of pairwise geodesic distances
   - Minimized by farthest-point k-center (select_representatives)
   - Analogous to entropy maximization (spread information across latent space)

4. **Decay weight**: exp(-lambda * age)
   - Exponential decay on older facts mirrors time-decay entropy loss
   - Recency prioritization = information freshness bias

5. **Flow weight as directional information gradient**
   - flow_weight = asymmetry / distance
   - Quantifies how fast containment changes along the hierarchy
   - Peaks where hierarchy is steepest (most information gain)

### C. What would add entropy formalism
To make the connection explicit, could measure:
- Aperture distribution entropy H(apertures) across nodes (cone sharpness diversity)
- Centroid clustering quality: divergence between member embeddings and cluster centroid
- Octave dimensionality reduction: KL(D_coarse || D_fine) information loss
- Relevance distribution over retrieved nodes (concentration vs. diffusion)

---

## PAPER SOURCES (from ARCHITECTURE.md)

| Topic | Paper | Citation |
|-------|-------|----------|
| Lorentz manifold + geoopt | geoopt docs | arxiv 2005.02819 |
| Entailment cones | Ganea 2018 | arxiv 1804.01882 |
| Entailment cones v2 | HypCBM 2026 | (inferred from code) |
| Matryoshka embeddings | ArXiv pre-print | arxiv 2205.13147 |
| Main system | Semantic-Cones v2/v3 | arxiv 2506.06946 |
| Versioning (lakeFS/CDC) | ArXiv pre-print | arxiv 2601.05270 |
| Observability (drift metric) | ArXiv pre-print | arxiv 2108.13557 |

---

## SUMMARY

**Hierarchies present:**
1. Hyperbolic cone structure (Ganea): explicit parent-child via aperture containment
2. Matryoshka octaves: explicit prefix-nested embedding hierarchy
3. Clustering tree: implicit via edges in ClusterTree, fitted to cones

**Compression mechanisms:**
1. Context deduplication by overlap score
2. Distance-based summarization (digests for far cones)
3. Token budgeting with priority (relevance > older facts)
4. Recursive octave-descent with early stopping

**Dimensions:**
1. Cone space: 8D (Lorentz), +1 time-like = 9-dim vector
2. Embedding per octave: 64D, 128D, 256D, 512D, 1024D (nested prefixes of 384D base)
3. Token space: 2048-token budgets per layer/context-pack

**Entropy connection:**
- No explicit Shannon entropy calculation
- Implicit: tension as redundancy, aperture distribution, octave compression ratio
- Could extend with information-bottleneck metrics on aperture/centroid distributions
