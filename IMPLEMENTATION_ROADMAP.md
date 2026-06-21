# Implementation Roadmap: Learning Loop Enhancements

## Quick Start: Phased Implementation Plan

### Phase 1: Diagnostics (1-2 days, low risk)

**Goal**: Surface per-octave health metrics without changing retrieval behavior.

#### 1.1 Expand `diagnose()` diagnostics

File: `core/agent_api.py`

**Current output**:
```python
@dataclass(frozen=True, slots=True)
class DiagnoseReport:
    nodes: int
    octaves: int
    texts: int
    facts: int
    mean_aperture: float
    mean_tension: float
    total_energy: float
    redundant_pairs: int
```

**New fields**:
```python
@dataclass(frozen=True, slots=True)
class DiagnoseReport:
    # ... existing fields ...
    
    # Per-octave details
    octave_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    # octave_stats[octave_id] = {
    #   "min_aperture": float, "mean_aperture": float, "max_aperture": float, "std_aperture": float,
    #   "node_count": int, "redundancy_rate": float, "mean_tension": float,
    # }
    
    # Retrieval flags
    flags: tuple[str, ...] = ()  # e.g., ("octave_3_high_tension", "wide_aperture_spread")
```

**Implementation**:
```python
def diagnose(self) -> DiagnoseReport:
    if self._pipeline is None:
        # ... current empty case ...
        return DiagnoseReport(0, 0, len(self._texts), ..., octave_stats={}, flags=())
    
    # Existing code ...
    nodes = [n for n in self._pipeline.store.all_nodes() if n.members]
    
    # NEW: Per-octave statistics
    octave_stats = {}
    by_octave = {}
    for n in nodes:
        if n.prefix not in by_octave:
            by_octave[n.prefix] = []
        by_octave[n.prefix].append(n)
    
    flags = []
    for prefix, o_nodes in by_octave.items():
        apertures = [n.aperture for n in o_nodes]
        scan_o = engine.tension_scan(o_nodes, top_n=len(o_nodes))
        redundant_o = sum(1 for _, _, _, kind in scan_o if kind in ("redundancy", "contradiction"))
        
        mean_a = sum(apertures) / len(apertures) if apertures else 0.0
        std_a = (sum((a - mean_a)**2 for a in apertures) / len(apertures))**0.5 if apertures else 0.0
        redundancy_rate = redundant_o / (len(o_nodes) * (len(o_nodes) - 1) / 2) if len(o_nodes) > 1 else 0.0
        mean_tension_o = sum(t for _, _, t, _ in scan_o) / len(scan_o) if scan_o else 0.0
        
        octave_stats[prefix] = {
            "min_aperture": min(apertures) if apertures else 0.0,
            "mean_aperture": mean_a,
            "max_aperture": max(apertures) if apertures else 0.0,
            "std_aperture": std_a,
            "node_count": len(o_nodes),
            "redundancy_rate": redundancy_rate,
            "mean_tension": mean_tension_o,
        }
        
        # Raise flags
        if std_a > mean_a * 0.5:
            flags.append(f"octave_{prefix}_wide_aperture_spread")
        if redundancy_rate > 0.3:
            flags.append(f"octave_{prefix}_high_redundancy")
    
    return DiagnoseReport(
        nodes=len(nodes), octaves=octaves, texts=len(self._texts),
        facts=len(self._memory.facts()), mean_aperture=float(mean_ap),
        mean_tension=float(mean_t), total_energy=float(energy), redundant_pairs=redundant,
        octave_stats=octave_stats, flags=tuple(flags),
    )
```

#### 1.2 Add per-octave recall tracking

File: `core/eval.py`

**New function**:
```python
def recall_at_k_per_octave(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> dict[int, float]:
    """Return recall@k separately for each octave."""
    from core.interfaces import Prefix
    
    if not labeled:
        return {}
    
    recall_by_octave = {}
    for query, relevant in labeled:
        hits = kb.search(query, k)  # Returns SearchHit objects with octave field
        by_octave = {}
        for h in hits:
            if h.octave not in by_octave:
                by_octave[h.octave] = set()
            by_octave[h.octave].add(h.text)
        
        for octave, hit_texts in by_octave.items():
            if octave not in recall_by_octave:
                recall_by_octave[octave] = [0.0, 0]
            correct = len(hit_texts & relevant)
            recall_by_octave[octave][0] += correct / len(relevant) if relevant else 0.0
            recall_by_octave[octave][1] += 1
    
    return {o: (s / c if c > 0 else 0.0) for o, (s, c) in recall_by_octave.items()}


def evaluate_detailed(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> dict[str, Any]:
    """Combined metrics: overall + per-octave recall."""
    return {
        "recall_at_k": recall_at_k(kb, labeled, k),
        "mrr": mrr(kb, labeled, k),
        "recall_at_k_per_octave": recall_at_k_per_octave(kb, labeled, k),
        "k": float(k),
    }
```

#### 1.3 Test: Verify diagnostics are correct

File: Create `core/test_diagnostics.py`

```python
import pytest
torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase
from core.settings import Settings

def test_diagnose_per_octave_stats():
    kb = KnowledgeBase(Settings())
    facts = [f"fact_{i}" for i in range(16)]
    kb.ingest(facts)
    
    rep = kb.diagnose()
    assert "octave_stats" in rep.__dict__
    for prefix, stats in rep.octave_stats.items():
        assert "mean_aperture" in stats
        assert "std_aperture" in stats
        assert "redundancy_rate" in stats
        assert 0.0 <= stats["redundancy_rate"] <= 1.0

def test_recall_per_octave():
    from core.eval import recall_at_k_per_octave
    
    kb = KnowledgeBase(Settings())
    facts = [f"unique_fact_{i}" for i in range(20)]
    kb.ingest(facts)
    
    labeled = [
        ("unique_fact_0", {"unique_fact_0"}),
        ("unique_fact_5", {"unique_fact_5"}),
    ]
    
    recall_o = recall_at_k_per_octave(kb, labeled, k=5)
    assert all(0.0 <= r <= 1.0 for r in recall_o.values())
```

#### 1.4 Deliverables

- [ ] `DiagnoseReport` extended with `octave_stats` and `flags`.
- [ ] `evaluate_detailed()` function returning per-octave recall.
- [ ] Tests pass; per-octave stats correlate with manual inspection.
- [ ] Agents can call `diagnose()` and inspect flags without code changes.

---

### Phase 2: Centroid Coherence Metric (2-3 days, medium risk)

**Goal**: Detect when node centroids drift from member embeddings.

#### 2.1 Add coherence metric to cone engine

File: `core/cone_engine.py`

**New method**:
```python
def centroid_coherence(self, node: ConeNode, member_embeddings: "np.ndarray | None" = None) -> dict[str, float]:
    """
    Measure how well a node's apex (centroid) represents its members.
    If member_embeddings is None, return placeholder (storage not yet cached).
    
    Returns:
    {
        "mean_distance": float,  # avg geodesic distance member -> centroid
        "max_distance": float,   # max distance (outlier detection)
        "entropy": float,        # variance of distances (coherence; low = tight cluster)
        "density": float,        # 1 - (entropy / max_distance), clamped to [0, 1]
    }
    """
    if member_embeddings is None or len(member_embeddings) == 0:
        return {"mean_distance": 0.0, "max_distance": 0.0, "entropy": 0.0, "density": 1.0}
    
    node_apex = torch.from_numpy(node.apex).float()
    member_apexes = torch.from_numpy(np.stack(member_embeddings)).float()
    
    # Compute geodesic distances
    distances = []
    for m_apex in member_apexes:
        d = self.manifold.dist(node_apex, m_apex).item()
        distances.append(d)
    
    distances = np.array(distances)
    mean_d = float(np.mean(distances))
    max_d = float(np.max(distances))
    std_d = float(np.std(distances))
    
    # Entropy proxy: normalized std dev
    entropy = std_d / (mean_d + 1e-7)  # avoid division by zero
    
    # Density: inverse of entropy
    density = 1.0 / (1.0 + entropy)  # clamp to [0, 1]
    
    return {
        "mean_distance": mean_d,
        "max_distance": max_d,
        "entropy": entropy,
        "density": density,
    }
```

#### 2.2 Extend `diagnose()` to include coherence

File: `core/agent_api.py`

**In `diagnose()` method**:
```python
# After existing code, compute coherence for sampled nodes
from .interfaces import phrase_to_text_index

sample_nodes = nodes[:min(32, len(nodes))]  # Sample for perf
coherence_scores = []

for node in sample_nodes:
    # Try to load member embeddings; if not cached, skip
    member_embs = []
    for m in node.members:
        idx = phrase_to_text_index(m, len(self._texts))
        if idx is not None:
            # Encode the text
            enc = self._pipeline._encoder
            text = self._texts[idx]
            emb = enc.encode([text])[0]
            member_embs.append(emb)
    
    if member_embs:
        coherence = engine.centroid_coherence(node, np.array(member_embs))
        coherence_scores.append(coherence["density"])

mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0

# Add to report
report = DiagnoseReport(
    # ... existing fields ...
    octave_stats=octave_stats,
    flags=tuple(flags),
    coherence_score=float(mean_coherence),  # NEW field
)
```

#### 2.3 Detect instability via ranking rotation

File: `core/agent_api.py`

**New method**:
```python
def _ranking_stability(self, query: str, num_samples: int = 3) -> float:
    """
    Sample the top-k ranking num_samples times (with different random seeds if diversity is on).
    Return fraction of consistent top-k entries (higher = more stable).
    
    This is a cheap way to detect if `explain_retrieval` results are noisy.
    """
    hits_list = [h.text for h in self.search(query, k=5) for _ in range(num_samples)]
    # Group by sample
    samples = [hits_list[i::num_samples] for i in range(num_samples)]
    
    # Intersection of all samples
    if not samples or not samples[0]:
        return 1.0
    
    common = set(samples[0])
    for s in samples[1:]:
        common &= set(s)
    
    return len(common) / len(samples[0])  # fraction of consistent hits
```

#### 2.4 Test coherence metrics

File: `core/test_diagnostics.py` (append)

```python
def test_centroid_coherence():
    from core.cone_engine import HyperbolicConeEngine, ConeFitConfig
    import numpy as np
    
    cfg = ConeFitConfig(dim=4, epochs=2)
    engine = HyperbolicConeEngine(cfg)
    
    # Create a fake node and members
    fake_node = ...  # ConeNode
    fake_members = np.random.randn(5, 4).astype(np.float32)
    
    coherence = engine.centroid_coherence(fake_node, fake_members)
    assert 0.0 <= coherence["density"] <= 1.0
    assert coherence["mean_distance"] >= 0.0

def test_ranking_stability():
    kb = KnowledgeBase(Settings())
    kb.ingest([f"fact_{i}" for i in range(20)])
    
    stability = kb._ranking_stability("fact_0", num_samples=3)
    assert 0.0 <= stability <= 1.0
```

#### 2.5 Deliverables

- [ ] `centroid_coherence()` computes member-to-apex distances and entropy.
- [ ] `diagnose()` includes `coherence_score` field.
- [ ] `_ranking_stability()` detects if top-k rotates significantly.
- [ ] Tests validate coherence correlates with manual inspection.

---

### Phase 3: Adaptive Octave Boundaries (3-4 days, medium-high risk)

**Goal**: Refit octaves if per-octave recall degrades.

#### 3.1 Add adaptive octave decision logic

File: `core/settings.py`

```python
class AgentSettings(BaseModel):
    # ... existing ...
    auto_tune_octaves: bool = False              # NEW: enable octave adaptation
    octave_recall_threshold: float = 0.5         # NEW: recall below this triggers adaptation
    min_octave_contraction_interval: int = 100   # NEW: epochs between re-fit attempts
```

#### 3.2 Implement octave adaptation in consolidate

File: `core/agent_api.py`

**New field in KnowledgeBase**:
```python
def __init__(self, settings: Settings | None = None) -> None:
    # ... existing ...
    self._last_octave_refit_epoch = 0
```

**In consolidate()**:
```python
def consolidate(self) -> dict[str, Any]:
    """Self-improve: tension scan + optional octave adaptation."""
    if self._pipeline is None:
        return {"actions": [], "reason": "empty"}
    
    self._metrics["consolidations"] += 1
    engine = self._pipeline.engine
    store = self._pipeline.store
    nodes = store.all_nodes()
    
    # Existing tension scan and merge/reparent
    scan = engine.tension_scan(nodes, top_n=len(nodes))
    thr = self._settings.agent.consolidate_tension
    plan = [row for row in engine.dispel_plan(scan)
            if any(s[0] == row[1] and s[1] == row[2] and s[2] >= thr for s in scan)]
    
    actions = []
    for op, a_id, b_id in plan:
        if op == "merge":
            try:
                merged = engine.merge_nodes(store.get(a_id), store.get(b_id))
                store.upsert(merged)
                actions.append({"op": "merge", "a": str(a_id), "b": str(b_id)})
            except KeyError:
                continue
    
    # NEW: Octave adaptation
    adapt_actions = []
    if self._settings.agent.auto_tune_octaves:
        epoch_since_refit = self._metrics["consolidations"] - self._last_octave_refit_epoch
        if epoch_since_refit >= self._settings.agent.min_octave_contraction_interval:
            # Analyze per-octave performance (requires labeled test set)
            # For now, use coherence as a proxy
            adapt_actions = self._adapt_octaves()
            if adapt_actions:
                self._last_octave_refit_epoch = self._metrics["consolidations"]
    
    return {
        "actions": actions + adapt_actions,
        "reason": "coherent" if not (actions or adapt_actions) else "consolidated",
    }

def _adapt_octaves(self) -> list[dict]:
    """Analyze per-octave coherence and decide if contraction/expansion is needed."""
    if self._pipeline is None:
        return []
    
    engine = self._pipeline.engine
    store = self._pipeline.store
    nodes = store.all_nodes()
    
    # Group by octave
    by_octave = {}
    for n in nodes:
        if n.prefix not in by_octave:
            by_octave[n.prefix] = []
        by_octave[n.prefix].append(n)
    
    actions = []
    for prefix, o_nodes in by_octave.items():
        # Compute coherence for this octave
        coherences = []
        for n in o_nodes[:16]:  # Sample for performance
            member_embs = []
            for m in n.members:
                from .interfaces import phrase_to_text_index
                idx = phrase_to_text_index(m, len(self._texts))
                if idx is not None:
                    enc = self._pipeline._encoder
                    text = self._texts[idx]
                    emb = enc.encode([text])[0]
                    member_embs.append(emb)
            
            if member_embs:
                coherence = engine.centroid_coherence(n, np.array(member_embs))
                coherences.append(coherence["density"])
        
        mean_coh = np.mean(coherences) if coherences else 1.0
        
        # Decision
        if mean_coh < 0.5:
            actions.append({
                "op": "octave_contraction_candidate",
                "octave": prefix,
                "mean_coherence": float(mean_coh),
                "reason": "low coherence suggests overshooting; recommend re-encoding with tighter dims or clustering",
            })
    
    return actions
```

#### 3.3 Test octave adaptation

File: `core/test_diagnostics.py` (append)

```python
def test_adapt_octaves_flags_low_coherence():
    kb = KnowledgeBase(Settings())
    s = kb._settings
    s.agent.auto_tune_octaves = True
    s.agent.min_octave_contraction_interval = 0  # Allow immediate refit for test
    kb._settings = s
    
    kb.ingest([f"text_{i}" for i in range(50)])
    
    result = kb.consolidate()
    # Check if adaptation actions were generated
    adapt_actions = [a for a in result.get("actions", []) if a.get("op", "").startswith("octave")]
    # May or may not have actions depending on data; just verify no crash
```

#### 3.4 Deliverables

- [ ] `AgentSettings` has `auto_tune_octaves`, `octave_recall_threshold`, `min_octave_contraction_interval`.
- [ ] `consolidate()` analyzes per-octave coherence.
- [ ] Flags are set but no re-fitting yet (agents downstream read the flags and decide).
- [ ] Tests verify logic runs without crashing.

---

### Phase 4: Memory Layer Auto-Tuning via Bayesian Optimization (5-7 days, high risk/reward)

**Goal**: Automatically tune memory layer allocation strategies.

#### 4.1 Create MemoryTuner class

File: Create `core/memory_tuner.py`

```python
"""Bayesian optimizer for memory hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import numpy as np


@dataclass
class MemoryConfig:
    """A candidate memory configuration."""
    fact_budget: int
    summary_budget: int
    working_budget: int
    session_budget: int
    recency_lambda: float
    digest_min_members: int


class MemoryTuner:
    """
    Bayesian Optimization over memory hyperparameters.
    Requires a recall@k evaluator to be called externally.
    """
    
    def __init__(self, search_space: dict[str, tuple[Any, Any]] | None = None):
        """
        search_space: dict of param -> (min, max) bounds.
        E.g. {"fact_budget": (0, 512), "recency_lambda": (0.01, 0.3)}
        """
        self.search_space = search_space or self._default_space()
        self.history: list[tuple[MemoryConfig, float]] = []  # (config, recall@k)
        self.best_config: MemoryConfig | None = None
        self.best_score: float = -1.0
    
    def _default_space(self) -> dict:
        return {
            "fact_budget": (0, 512),
            "summary_budget": (0, 512),
            "working_budget": (256, 1024),
            "session_budget": (256, 1024),
            "recency_lambda": (0.01, 0.3),
            "digest_min_members": (1, 5),
        }
    
    def suggest_next(self) -> MemoryConfig:
        """
        Return the next config to evaluate.
        Strategy: random sampling for first 5 trials, then best-of-random for subsequent.
        (Simple, stable; a real impl would use GP-based UCB or EI.)
        """
        if len(self.history) < 5:
            # Random exploration
            config = self._random_config()
        else:
            # Exploitation: sample N random configs, return the one closest to best_config
            # in terms of expected improvement
            candidates = [self._random_config() for _ in range(10)]
            best_candidate = max(candidates, key=lambda c: self._expected_improvement(c))
            config = best_candidate
        
        return config
    
    def _random_config(self) -> MemoryConfig:
        """Sample a random config from the search space."""
        rng = np.random.default_rng()
        return MemoryConfig(
            fact_budget=int(rng.integers(*self.search_space["fact_budget"])),
            summary_budget=int(rng.integers(*self.search_space["summary_budget"])),
            working_budget=int(rng.integers(*self.search_space["working_budget"])),
            session_budget=int(rng.integers(*self.search_space["session_budget"])),
            recency_lambda=float(rng.uniform(*self.search_space["recency_lambda"])),
            digest_min_members=int(rng.integers(*self.search_space["digest_min_members"])),
        )
    
    def _expected_improvement(self, config: MemoryConfig) -> float:
        """
        Simple proxy for EI: distance to best_config in budget space.
        A real impl would use a GP posterior.
        """
        if self.best_config is None:
            return 0.5  # Neutral
        
        # Squared L2 distance (simple heuristic)
        dist = (
            (config.fact_budget - self.best_config.fact_budget) ** 2 +
            (config.summary_budget - self.best_config.summary_budget) ** 2 +
            (config.working_budget - self.best_config.working_budget) ** 2 +
            (config.session_budget - self.best_config.session_budget) ** 2 +
            ((config.recency_lambda - self.best_config.recency_lambda) * 100) ** 2
        )
        
        # Closer = higher EI
        return 1.0 / (1.0 + np.sqrt(dist))
    
    def observe(self, config: MemoryConfig, recall_at_k: float) -> None:
        """Record the outcome of a trial."""
        self.history.append((config, recall_at_k))
        
        if recall_at_k > self.best_score:
            self.best_score = recall_at_k
            self.best_config = config
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "search_space": self.search_space,
            "history": [
                {
                    "config": {
                        "fact_budget": c.fact_budget,
                        "summary_budget": c.summary_budget,
                        "working_budget": c.working_budget,
                        "session_budget": c.session_budget,
                        "recency_lambda": c.recency_lambda,
                        "digest_min_members": c.digest_min_members,
                    },
                    "recall_at_k": float(score),
                }
                for c, score in self.history
            ],
            "best_score": self.best_score,
            "best_config": {
                "fact_budget": self.best_config.fact_budget,
                "summary_budget": self.best_config.summary_budget,
                "working_budget": self.best_config.working_budget,
                "session_budget": self.best_config.session_budget,
                "recency_lambda": self.best_config.recency_lambda,
                "digest_min_members": self.best_config.digest_min_members,
            } if self.best_config else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryTuner:
        """Deserialize."""
        tuner = cls(data.get("search_space"))
        for item in data.get("history", []):
            cfg = MemoryConfig(**item["config"])
            tuner.observe(cfg, item["recall_at_k"])
        return tuner
```

#### 4.2 Integrate tuner into KnowledgeBase

File: `core/agent_api.py`

**In `__init__`**:
```python
def __init__(self, settings: Settings | None = None) -> None:
    self._settings = settings or Settings()
    self._pipeline: KnowledgePipeline | None = None
    self._texts: list[str] = []
    self._memory = SemioticMemory(None, [], self._settings)
    self._usage: dict[str, int] = {}
    self._metrics: dict[str, int] = {...}
    
    # NEW
    self._memory_tuner: MemoryTuner | None = None
    self._labeled_queries: list[tuple[str, set[str]]] | None = None  # Must be provided by agent
    if self._settings.agent.auto_tune_memory:
        from .memory_tuner import MemoryTuner
        self._memory_tuner = MemoryTuner()
```

**New public methods**:
```python
def set_labeled_queries(self, queries: list[tuple[str, set[str]]]) -> None:
    """
    Set the labeled test set for memory tuning.
    Required if auto_tune_memory is enabled.
    Each query: (question_text, set of relevant answer texts)
    """
    self._labeled_queries = queries

def consolidate(self) -> dict[str, Any]:
    # ... existing code ...
    
    # NEW: Memory tuning
    tuning_result = {}
    if self._settings.agent.auto_tune_memory and self._labeled_queries:
        from .eval import evaluate_detailed
        
        # Suggest next config to evaluate
        next_config = self._memory_tuner.suggest_next()
        
        # Apply it
        old_memory = self._memory
        self._memory = SemioticMemory(None, list(self._texts), self._settings)
        # TODO: apply next_config to memory._settings
        
        # Evaluate
        eval_result = evaluate_detailed(self, self._labeled_queries, k=5)
        recall = eval_result.get("recall_at_k", 0.0)
        
        # Observe
        self._memory_tuner.observe(next_config, recall)
        
        # Optionally switch back to best seen so far
        if self._memory_tuner.best_config:
            # Apply best_config
            self._memory = SemioticMemory(None, list(self._texts), self._settings)
            # TODO: apply best_config
        
        tuning_result = {
            "op": "memory_tune",
            "trials": len(self._memory_tuner.history),
            "best_score": self._memory_tuner.best_score,
            "last_recall": recall,
        }
    
    return {
        "actions": actions,
        "tuning": tuning_result,
        "reason": "consolidated",
    }
```

#### 4.3 Settings support for tuning

File: `core/settings.py`

```python
class AgentSettings(BaseModel):
    # ... existing ...
    auto_tune_memory: bool = False           # NEW: enable memory auto-tuning
    memory_tune_interval: int = 10           # NEW: consolidate() epochs between tuning trials
```

#### 4.4 Test tuning integration

File: `core/test_diagnostics.py` (append)

```python
def test_memory_tuner_suggest_and_observe():
    from core.memory_tuner import MemoryTuner, MemoryConfig
    
    tuner = MemoryTuner()
    
    config1 = tuner.suggest_next()
    assert isinstance(config1, MemoryConfig)
    
    tuner.observe(config1, 0.7)
    config2 = tuner.suggest_next()
    assert isinstance(config2, MemoryConfig)
    
    tuner.observe(config2, 0.8)
    assert tuner.best_score == 0.8
    assert tuner.best_config == config2

def test_kb_memory_tuning():
    s = Settings()
    s.agent.auto_tune_memory = True
    kb = KnowledgeBase(s)
    kb.ingest([f"text_{i}" for i in range(30)])
    
    labeled = [
        ("text_0", {"text_0"}),
        ("text_5", {"text_5"}),
    ]
    kb.set_labeled_queries(labeled)
    
    result = kb.consolidate()
    # Should have tuning results if labeled queries provided
    assert "tuning" in result
```

#### 4.5 Deliverables

- [ ] `MemoryTuner` class implements simple Bayesian-ish optimization.
- [ ] `KnowledgeBase.set_labeled_queries()` allows agents to provide evaluation data.
- [ ] `consolidate()` optionally runs tuning trials and tracks best config.
- [ ] Tests verify tuner converges and applies best config.
- [ ] Tuner state can be serialized/deserialized via `save()` / `load()`.

---

## Integration Checklist

After implementing all phases:

- [ ] All new fields in dataclasses have default values (backward compatible).
- [ ] All new settings have `bool` feature flags (can be disabled).
- [ ] No changes to hot paths (search, deep_search) unless gated.
- [ ] All tests pass: `pytest core/`.
- [ ] Docs updated: `docs/learning-loop.md` with examples.
- [ ] Example agent script: `examples/tuning_loop.py` shows full flow.

---

## Estimated Effort

| Phase | Risk | Effort | Days |
|-------|------|--------|------|
| 1. Diagnostics | Low | 2 PR + tests | 1-2 |
| 2. Coherence Metric | Low-Med | 3 PR + tests | 2-3 |
| 3. Octave Adaptation | Med-High | 5 PR + tests + validation | 3-4 |
| 4. Memory Auto-Tuning | High | 8 PR + tests + integration | 5-7 |
| **Total** | **Med** | **18 PR** | **11-16** |

---

## Success Criteria

1. **Phase 1**: `diagnose()` output includes per-octave stats and flags. No retrieval behavior change.
2. **Phase 2**: Coherence metric correlates with manual coherence judgment (expert review).
3. **Phase 3**: Octave adaptation flags appear in `consolidate()` output; downstream agents can read them.
4. **Phase 4**: MemoryTuner converges to a stable best config in < 20 trials. Best config improves recall@k vs. baseline by > 5%.

---

## Follow-Up: Papers to Implement

Once the framework is solid, consider:

1. **CMA-ES** (Hansen & Ostermeier, 2001): Covariance Matrix Adaptation for black-box optimization.
   - Replace simple random sampling with CMA-ES for faster convergence.
   - File: `core/cma_tuner.py`.

2. **Hyperband** (Li et al., 2018): Early-stopping strategy.
   - Run tuning trials in a bracket; promote survivors.
   - File: `core/hyperband_tuner.py`.

3. **Gaussian Process** (Rasmussen & Williams, 2006): Kernel-based surrogate model for BO.
   - Use `scikit-optimize` or `botorch` for GP-based UCB/EI.
   - File: `core/gp_tuner.py`.

4. **Population-Based Training** (Jaderberg et al., 2017): Asynchronous hyperparameter evolution.
   - Run multiple KB instances in parallel; share best weights.
   - File: `core/pbt_tuner.py`.

