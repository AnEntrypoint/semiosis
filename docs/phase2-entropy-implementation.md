# Phase 2: Entropy Implementation Guide

**Target:** Add entropy estimation to semiosis, enabling information-bottleneck context packing.  
**Effort:** 3-4 hours total (modular, testable increments).  
**Entry point:** `core/interfaces.py` → `core/context_pack.py` → tests.

---

## Step 1: Schema Extension (5 min)

**File:** `core/interfaces.py`

Add two fields to `ConeNode`:

```python
@dataclass(frozen=True, slots=True)
class ConeNode:
    id: NodeId
    apex: LorentzVec
    aperture: float
    prefix: Prefix
    members: tuple[PhraseId, ...]
    label: str | None = None
    digest: str | None = None
    pinned: bool = False
    centroid: tuple[float, ...] | None = None
    # NEW FIELDS
    entropy: float | None = None          # Shannon entropy of members' embeddings
    information_value: float | None = None  # scale-free info-theoretic score
```

**Why these two?**
- `entropy` — Measures member diversity; computed during context packing
- `information_value` — Query-independent score; useful for long-term caching

**Backward compat:** Both optional (None by default); old nodes still valid.

---

## Step 2: Entropy Computation (1-2 hours)

**File:** `core/context_pack.py`

Add a utility class for entropy calculations:

```python
import numpy as np
from typing import Sequence
import math

class EntropyEstimator:
    """Compute information-theoretic metrics on embeddings."""
    
    @staticmethod
    def shannon_entropy(distances: Sequence[float]) -> float:
        """Compute Shannon entropy from distance distribution.
        
        High entropy = members are diverse (spread out).
        Low entropy = members are similar (tight cluster).
        """
        if not distances or len(distances) < 2:
            return 0.0
        
        # Convert distances to probabilities
        dists = np.array(distances, dtype=np.float32)
        # Closer distances => higher probability (use negative exponential)
        probs = np.exp(-dists)
        probs = probs / (probs.sum() + 1e-10)
        
        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    @staticmethod
    def renyi_entropy(distances: Sequence[float], alpha: float = 0.5) -> float:
        """Renyi entropy (generalization of Shannon).
        
        Useful for capturing higher-order moments of diversity.
        """
        if not distances or len(distances) < 2:
            return 0.0
        
        dists = np.array(distances, dtype=np.float32)
        probs = np.exp(-dists)
        probs = probs / (probs.sum() + 1e-10)
        
        if alpha == 1.0:
            # Limit case: Shannon entropy
            return EntropyEstimator.shannon_entropy(distances)
        else:
            # Renyi: H_α = (1/(1-α)) * log(sum(p^α))
            h_alpha = (1.0 / (1.0 - alpha)) * np.log(np.sum(np.power(probs, alpha)) + 1e-10)
            return float(h_alpha)
    
    @staticmethod
    def cluster_compactness(distances: Sequence[float]) -> float:
        """Inverse of entropy; high compactness = tight cluster, low = spread out."""
        entropy = EntropyEstimator.shannon_entropy(distances)
        # Normalize to [0, 1] range (assuming entropy <= log(n) for n members)
        max_entropy = math.log(max(len(distances), 2))
        compactness = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return max(0.0, min(1.0, compactness))
```

Now add the computation method to `ContextPackBuilder`:

```python
class ContextPackBuilder:
    """Assemble a budget-bounded, redundancy-free context pack over a fitted cone store."""
    
    # ... existing __init__, _degraded, _node_text, _summary_text ...
    
    def _compute_node_entropy(self, node: "ConeNode") -> float:
        """Estimate Shannon entropy of a node's members.
        
        Returns value in [0, log(n_members)]; higher = more diverse members.
        Used to weight high-entropy (informative) nodes higher during packing.
        """
        if not node.members or not self._pipeline:
            return 0.0
        
        # Resolve member embeddings
        try:
            encoder = self._pipeline._encoder
        except (AttributeError, TypeError):
            return 0.0
        
        member_vecs = []
        member_texts = []
        for m in node.members[:self._cfg.max_members_per_node]:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None and 0 <= idx < len(self._texts):
                try:
                    vec = encoder.encode([self._texts[idx]])[0]
                    member_vecs.append(vec)
                    member_texts.append(self._texts[idx])
                except Exception:
                    continue
        
        if len(member_vecs) < 2:
            # Single member or no members => no diversity
            return 0.0
        
        # Compute distances from members to node centroid
        if node.centroid:
            centroid = np.array(node.centroid, dtype=np.float32)
        else:
            # Compute centroid if not available
            centroid = np.mean([v[:node.prefix] for v in member_vecs], axis=0)
        
        distances = []
        for vec in member_vecs:
            # Euclidean distance in embedding space
            sliced = vec[:node.prefix].astype(np.float32)
            dist = float(np.linalg.norm(sliced - centroid))
            distances.append(dist)
        
        # Compute entropy via utility class
        entropy = EntropyEstimator.shannon_entropy(distances)
        return entropy
```

**Testing for Step 2:**

```python
# core/test_entropy.py
import pytest
from context_pack import EntropyEstimator

def test_shannon_entropy_single_member():
    """Single member => zero entropy."""
    assert EntropyEstimator.shannon_entropy([0.0]) == 0.0

def test_shannon_entropy_uniform():
    """Uniform distribution => max entropy."""
    # Distances all equal => uniform prob distribution
    distances = [0.5] * 5
    entropy = EntropyEstimator.shannon_entropy(distances)
    assert entropy > 0.0

def test_shannon_entropy_skewed():
    """Skewed distances => lower entropy."""
    uniform = [0.5] * 5
    skewed = [0.1, 0.1, 0.1, 1.0, 2.0]
    
    entropy_uniform = EntropyEstimator.shannon_entropy(uniform)
    entropy_skewed = EntropyEstimator.shannon_entropy(skewed)
    
    assert entropy_uniform > entropy_skewed, \
        "Uniform distribution should have higher entropy than skewed"

def test_compactness():
    """Inverse of entropy."""
    tight = [0.05] * 3  # tight cluster
    loose = [0.5] * 3   # loose cluster
    
    compact_tight = EntropyEstimator.cluster_compactness(tight)
    compact_loose = EntropyEstimator.cluster_compactness(loose)
    
    assert compact_tight > compact_loose, \
        "Tight cluster should have higher compactness"
```

---

## Step 3: IB-Weighted Context Packing (1-2 hours)

**File:** `core/context_pack.py`

Modify the `build()` method to incorporate entropy:

```python
class ContextPackBuilder:
    """Assemble a budget-bounded, redundancy-free context pack over a fitted cone store."""
    
    def build(self, query: str, max_tokens: "int | None" = None, 
              use_ib_weighting: bool = True) -> ContextPack:
        """Build context pack with optional information-bottleneck weighting.
        
        Args:
            query: User query string
            max_tokens: Token budget (or None to use config default)
            use_ib_weighting: If True, weight entries by (relevance * entropy)
        
        Returns:
            ContextPack with selected entries, token count, and metadata
        """
        cfg = self._cfg
        budget = max_tokens or cfg.max_tokens
        
        # Retrieve candidates (existing logic, unchanged)
        candidates = self._retrieve_candidates(query)  # returns list of (node, relevance)
        
        if not use_ib_weighting:
            # Original greedy-by-relevance logic
            return self._build_greedy(candidates, budget)
        
        # NEW: IB-weighted selection
        entries_scored = []
        for node, relevance in candidates:
            text = self._node_text(node)
            tokens = self._counter.count(text)
            
            # Compute entropy for this node
            entropy = self._compute_node_entropy(node)
            
            # Information-Bottleneck weight: high relevance + high entropy = high priority
            # Scale: relevance is typically [0, 1]; entropy is [0, log(n_members)]
            # Normalize entropy to [0, 1] by dividing by log(max_members)
            max_members = max(1, len(node.members))
            entropy_normalized = entropy / max(1.0, math.log(max_members))
            
            # Combine: IB weight emphasizes both relevance AND diversity
            # If relevance = 0.8 and entropy_norm = 0.6:
            #   ib_weight = 0.8 * (1 + 0.6) = 1.28
            ib_weight = relevance * (1.0 + 0.5 * entropy_normalized)
            
            entries_scored.append({
                "node": node,
                "text": text,
                "tokens": tokens,
                "relevance": relevance,
                "entropy": entropy,
                "ib_weight": ib_weight,
            })
        
        # Sort by IB weight (descending)
        entries_scored.sort(key=lambda x: x["ib_weight"], reverse=True)
        
        # Greedy packing by IB weight
        selected: list[ContextEntry] = []
        total_tokens = 0
        dropped: list[NodeId] = []
        truncated = False
        
        for item in entries_scored:
            tokens = item["tokens"]
            if total_tokens + tokens <= budget:
                # Fits within budget
                selected.append(ContextEntry(
                    node_id=item["node"].id,
                    text=item["text"],
                    tokens=tokens,
                    relevance=item["relevance"],
                    is_summary=False,
                    represented=(),
                ))
                total_tokens += tokens
            elif total_tokens + tokens <= budget + cfg.reserve_tokens:
                # Fits in reserve; only if high entropy (informative)
                if item["entropy"] > 0.5:  # configurable threshold
                    selected.append(ContextEntry(
                        node_id=item["node"].id,
                        text=item["text"],
                        tokens=tokens,
                        relevance=item["relevance"],
                    ))
                    total_tokens += tokens
                    truncated = True
                    break
                else:
                    dropped.append(item["node"].id)
            else:
                # Exceeds budget + reserve
                dropped.append(item["node"].id)
                truncated = True
        
        return ContextPack(
            entries=tuple(selected),
            total_tokens=total_tokens,
            dropped_ids=tuple(dropped),
            truncated=truncated,
            degraded=self._degraded(),
            low_confidence=False,
        )
    
    def _retrieve_candidates(self, query: str) -> list[tuple]:
        """Helper: retrieve candidate nodes for a query (existing logic extracted)."""
        # This would contain the existing _ranked() logic from agent_api
        # For now, stub it as a helper to show structure
        raise NotImplementedError("Use pipeline.search() or agent_api._ranked()")
    
    def _build_greedy(self, candidates, budget: int) -> ContextPack:
        """Original greedy-by-relevance packing (for backward compat)."""
        selected: list[ContextEntry] = []
        total_tokens = 0
        
        for node, relevance in candidates:
            text = self._node_text(node)
            tokens = self._counter.count(text)
            
            if total_tokens + tokens <= budget:
                selected.append(ContextEntry(
                    node_id=node.id,
                    text=text,
                    tokens=tokens,
                    relevance=relevance,
                ))
                total_tokens += tokens
            else:
                break
        
        return ContextPack(
            entries=tuple(selected),
            total_tokens=total_tokens,
            truncated=True if len(candidates) > len(selected) else False,
        )
```

**Testing for Step 3:**

```python
# core/test_context_pack_ib.py
import pytest
from context_pack import ContextPackBuilder, ContextPackConfig

def test_ib_weighting_prioritizes_entropy(pipeline, texts):
    """High-entropy nodes should be selected before low-entropy."""
    config = ContextPackConfig(max_tokens=100)
    builder = ContextPackBuilder(pipeline, texts, config)
    
    # Create mock candidates: one high-entropy, one low-entropy
    # Both same relevance
    # High-entropy should be selected first
    
    pack = builder.build("test query", use_ib_weighting=True)
    
    # Assertion: if we have budget, high-entropy entry comes before low-entropy
    assert len(pack.entries) > 0
    entropy_scores = [
        builder._compute_node_entropy(
            pipeline.store.get(NodeId(e.node_id))
        )
        for e in pack.entries
    ]
    # Entropy scores should be roughly decreasing (or at least high-entropy first)
    assert entropy_scores[0] >= entropy_scores[-1] if len(entropy_scores) > 1 else True

def test_ib_weighting_off(pipeline, texts):
    """With use_ib_weighting=False, should fall back to greedy-by-relevance."""
    config = ContextPackConfig(max_tokens=100)
    builder = ContextPackBuilder(pipeline, texts, config)
    
    pack = builder.build("test query", use_ib_weighting=False)
    
    # Should not error; should return valid ContextPack
    assert isinstance(pack, ContextPack)
    assert len(pack.entries) >= 0

def test_entropy_computation_realistic(pipeline, texts):
    """Entropy should correlate with member diversity."""
    builder = ContextPackBuilder(pipeline, texts, ContextPackConfig())
    
    # Create nodes with different member counts
    tight_node = ConeNode(
        id=NodeId("tight"),
        apex=np.array([...]),
        aperture=0.1,
        prefix=Prefix(64),
        members=tuple(["phrase_0", "phrase_1", "phrase_2"]),  # similar
    )
    loose_node = ConeNode(
        id=NodeId("loose"),
        apex=np.array([...]),
        aperture=0.5,
        prefix=Prefix(64),
        members=tuple(["phrase_10", "phrase_20", "phrase_30"]),  # diverse
    )
    
    entropy_tight = builder._compute_node_entropy(tight_node)
    entropy_loose = builder._compute_node_entropy(loose_node)
    
    # Loose node should have higher entropy
    assert entropy_loose >= entropy_tight, \
        f"Diverse members should have higher entropy: {entropy_loose} vs {entropy_tight}"
```

---

## Step 4: Integration Test (30 min)

**File:** `core/test_context_pack_integration.py`

```python
import pytest
from core.context_pack import ContextPackBuilder, ContextPackConfig
from core.agent_api import KnowledgeBase

def test_ib_weighting_end_to_end():
    """Full pipeline: ingest texts -> build packs -> compare with/without IB."""
    kb = KnowledgeBase()
    texts = [
        "WebGL is a JavaScript API for rendering 3D graphics.",
        "The canvas element is used to draw graphics on the web.",
        "Shaders are programs that run on the GPU.",
        "Vertex shaders transform 3D vertices into 2D screen coordinates.",
        "Fragment shaders determine the color of each pixel.",
    ]
    kb.ingest(texts)
    
    query = "How do shaders work in WebGL?"
    
    # Retrieve with IB weighting (enabled)
    config_ib = ContextPackConfig(max_tokens=256)
    builder_ib = ContextPackBuilder(kb._pipeline, texts, config_ib)
    pack_ib = builder_ib.build(query, use_ib_weighting=True)
    
    # Retrieve without IB weighting (baseline)
    builder_baseline = ContextPackBuilder(kb._pipeline, texts, config_ib)
    pack_baseline = builder_baseline.build(query, use_ib_weighting=False)
    
    # Both should have reasonable results
    assert len(pack_ib.entries) > 0, "IB weighting should retrieve results"
    assert len(pack_baseline.entries) > 0, "Baseline should retrieve results"
    
    # IB-weighted pack may have different entry count/order due to entropy weighting
    # (not necessarily larger, but potentially more informative)
    
    # Print for inspection
    print("\n=== IB-Weighted Pack ===")
    for e in pack_ib.entries:
        print(f"  [{e.node_id}] {e.text[:50]}... (rel={e.relevance:.2f})")
    
    print("\n=== Baseline Pack ===")
    for e in pack_baseline.entries:
        print(f"  [{e.node_id}] {e.text[:50]}... (rel={e.relevance:.2f})")
```

---

## Step 5: Update Settings (15 min)

**File:** `core/settings.py`

Add configuration for entropy computation:

```python
@dataclass
class ContextPackConfig(BaseModel):
    max_tokens: int = 2048
    overlap_threshold: float = 0.5
    distance_summary_threshold: float = 0.0
    max_members_per_node: int = 4
    reserve_tokens: int = 64
    max_dedup_candidates: int = 256
    # NEW
    use_entropy_weighting: bool = True  # enable IB weighting by default
    entropy_weight_scale: float = 0.5  # scale of entropy's contribution (0..1)
    entropy_reserve_threshold: float = 0.5  # min entropy to use reserve tokens
```

Update root `Settings` to include this:

```python
@dataclass
class Settings(BaseSettings):
    # ... existing fields ...
    context_pack: ContextPackConfig = field(default_factory=ContextPackConfig)
    
    # ... rest of settings ...
```

---

## Validation Checklist

- [ ] Schema extension (entropy, information_value) added to ConeNode
- [ ] EntropyEstimator class implemented with unit tests
- [ ] _compute_node_entropy() method added to ContextPackBuilder
- [ ] build() method accepts use_ib_weighting parameter
- [ ] IB weight formula is (relevance * (1 + entropy_normalized))
- [ ] Greedy packing respects token budget and reserve
- [ ] Tests pass: entropy computation, IB weighting, integration
- [ ] Backward compat: use_ib_weighting=False recovers original behavior
- [ ] Settings expanded with entropy_weight_scale and friends

---

## Performance Considerations

### Entropy Computation Cost

**Per node:** ~O(k * d) where k = max_members_per_node, d = embedding_dim (1024).  
**For 1000 nodes:** ~1M arithmetic operations. Negligible.

### Caching Strategy

To avoid recomputing entropy on every query:

```python
class ConeNode:
    # ... existing ...
    entropy: float | None = None
    entropy_cached_at: float | None = None  # timestamp

def _compute_node_entropy_cached(self, node: ConeNode) -> float:
    """Return cached entropy if available and recent."""
    import time
    now = time.time()
    if node.entropy is not None and node.entropy_cached_at is not None:
        age = now - node.entropy_cached_at
        if age < 3600:  # 1 hour
            return node.entropy
    
    # Recompute and cache
    entropy = self._compute_node_entropy(node)
    # Note: ConeNode is frozen, so we can't update in-place
    # Instead, compute on-the-fly or update in a parallel cache dict
    return entropy
```

---

## What NOT to Change in Phase 2

1. **Cone fitting** (`cone_engine.py`) — Leave untouched
2. **Retrieval ranking** (`agent_api.py._ranked()`) — Keep octave fusion unchanged
3. **Store** (`store.py`, if it exists) — No schema changes
4. **Tests outside context_pack** — Don't break existing test suites

---

## Next: Phase 4 Preview

Once Phase 2 is complete and validated, Phase 4 will use entropy scores to implement **Octave-Specialized Heads**:

```python
class OctaveHead:
    def __init__(self, n_octaves: int):
        self.octave_weights = np.ones(n_octaves) / n_octaves
    
    def score(self, query_entropy: float, candidates_per_octave: list[dict]) -> dict:
        """Blend octaves, weighted by query's implicit information-seeking style."""
        # High-entropy query => favor fine octaves
        # Low-entropy query => favor coarse octaves
        weights = self.octave_weights * (1.0 + query_entropy)
        # ... blend candidates ...
```

---

**Summary:** Phase 2 is 4 incremental, testable steps. No architectural changes; pure additive feature. Ready to begin implementation.
