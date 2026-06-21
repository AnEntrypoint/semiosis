# Implementation Examples: Concrete Code Snippets

This document provides copy-paste-ready code examples for each hypothesis implementation.

---

## Hypothesis 1: Per-Octave Diagnostics

### Example 1.1: Expand DiagnoseReport

```python
# core/agent_api.py

from dataclasses import dataclass, field

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
    
    # NEW
    octave_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    coherence_score: float = 0.0
```

### Example 1.2: Compute Per-Octave Stats in diagnose()

```python
# core/agent_api.py, in KnowledgeBase.diagnose()

def diagnose(self) -> DiagnoseReport:
    if self._pipeline is None:
        return DiagnoseReport(
            0, 0, len(self._texts), len(self._memory.facts()),
            0.0, 0.0, 0.0, 0,
            octave_stats={}, flags=(), coherence_score=0.0
        )
    
    engine = self._pipeline.engine
    nodes = [n for n in self._pipeline.store.all_nodes() if n.members]
    octaves = len({n.prefix for n in nodes})
    mean_ap = sum(n.aperture for n in nodes) / len(nodes) if nodes else 0.0
    scan = engine.tension_scan(nodes, top_n=len(nodes))
    mean_t = sum(t for _, _, t, _ in scan) / len(scan) if scan else 0.0
    redundant = sum(1 for _, _, _, kind in scan if kind in ("redundancy", "contradiction"))
    energy = engine.context_energy(nodes[:32])
    
    # NEW: Per-octave stats
    octave_stats = {}
    by_octave = {}
    for n in nodes:
        if n.prefix not in by_octave:
            by_octave[n.prefix] = []
        by_octave[n.prefix].append(n)
    
    flags = []
    for prefix, o_nodes in by_octave.items():
        apertures = [n.aperture for n in o_nodes]
        scan_o = engine.tension_scan(o_nodes, top_n=min(len(o_nodes), 20))
        redundant_o = sum(1 for _, _, _, kind in scan_o if kind in ("redundancy", "contradiction"))
        
        mean_a = sum(apertures) / len(apertures) if apertures else 0.0
        std_a = (sum((a - mean_a)**2 for a in apertures) / len(apertures))**0.5 if apertures else 0.0
        redundancy_rate = redundant_o / max(1, len(o_nodes) * (len(o_nodes) - 1) / 2)
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
        
        if std_a > mean_a * 0.5 and mean_a > 0.1:
            flags.append(f"octave_{prefix}_wide_aperture_spread")
        if redundancy_rate > 0.3:
            flags.append(f"octave_{prefix}_high_redundancy")
    
    # Compute coherence (sample)
    coherence_scores = []
    sample = nodes[:min(16, len(nodes))]
    for n in sample:
        member_embs = []
        for m in n.members:
            from .interfaces import phrase_to_text_index
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None:
                try:
                    enc = self._pipeline._encoder
                    text = self._texts[idx]
                    emb = enc.encode([text])[0]
                    member_embs.append(emb)
                except Exception:
                    pass
        
        if len(member_embs) > 1:
            coherence = engine.centroid_coherence(n, np.array(member_embs, dtype=np.float32))
            coherence_scores.append(coherence["density"])
    
    mean_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
    
    return DiagnoseReport(
        nodes=len(nodes), octaves=octaves, texts=len(self._texts),
        facts=len(self._memory.facts()), mean_aperture=float(mean_ap),
        mean_tension=float(mean_t), total_energy=float(energy), redundant_pairs=redundant,
        octave_stats=octave_stats, flags=tuple(flags), coherence_score=float(mean_coherence),
    )
```

### Example 1.3: Per-Octave Recall in eval.py

```python
# core/eval.py

def recall_at_k_per_octave(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> dict[int, float]:
    """Recall@k per octave."""
    if not labeled:
        return {}
    
    recall_by_octave: dict[int, list[float]] = {}
    
    for query, relevant in labeled:
        if not relevant:
            continue
        hits = kb.search(query, k)  # Returns SearchHit objects with octave field
        
        by_octave = {}
        for h in hits:
            if h.octave not in by_octave:
                by_octave[h.octave] = set()
            by_octave[h.octave].add(h.text)
        
        for octave, hit_texts in by_octave.items():
            if octave not in recall_by_octave:
                recall_by_octave[octave] = []
            correct = len(hit_texts & relevant)
            recall_by_octave[octave].append(correct / len(relevant))
    
    return {o: (sum(rs) / len(rs)) for o, rs in recall_by_octave.items()}


def evaluate_detailed(kb, labeled: Sequence[tuple[str, set[str]]], k: int = 5) -> dict[str, Any]:
    """Overall + per-octave metrics."""
    return {
        "recall_at_k": recall_at_k(kb, labeled, k),
        "mrr": mrr(kb, labeled, k),
        "recall_at_k_per_octave": recall_at_k_per_octave(kb, labeled, k),
        "k": float(k),
    }
```

---

## Hypothesis 2: Centroid Coherence

### Example 2.1: Add centroid_coherence() to HyperbolicConeEngine

```python
# core/cone_engine.py

def centroid_coherence(self, node: ConeNode, member_embeddings: "np.ndarray | None" = None) -> dict[str, float]:
    """
    Measure node coherence: how well the apex (centroid) represents members.
    
    Args:
        node: The ConeNode.
        member_embeddings: [N, D] array of member embedding vectors (or None).
    
    Returns:
        {
            "mean_distance": float,   # avg geodesic distance
            "max_distance": float,    # max distance (outlier)
            "entropy": float,         # normalized std dev
            "density": float,         # coherence score in [0, 1]
        }
    """
    if member_embeddings is None or len(member_embeddings) == 0:
        return {"mean_distance": 0.0, "max_distance": 0.0, "entropy": 0.0, "density": 1.0}
    
    node_apex = torch.from_numpy(node.apex).float()
    member_apexes = torch.from_numpy(np.atleast_2d(member_embeddings)).float()
    
    distances = []
    for m_apex in member_apexes:
        d = float(self.manifold.dist(node_apex, m_apex).item())
        distances.append(d)
    
    distances_arr = np.array(distances, dtype=np.float32)
    mean_d = float(np.mean(distances_arr))
    max_d = float(np.max(distances_arr))
    std_d = float(np.std(distances_arr))
    
    # Entropy proxy
    entropy = std_d / (mean_d + _EPS)
    
    # Density: inverse of normalized entropy
    density = 1.0 / (1.0 + entropy)
    
    return {
        "mean_distance": mean_d,
        "max_distance": max_d,
        "entropy": entropy,
        "density": density,
    }
```

### Example 2.2: Ranking Stability Check

```python
# core/agent_api.py, in KnowledgeBase class

def ranking_stability(self, query: str, num_samples: int = 3) -> float:
    """
    Test ranking stability: fraction of consistent top-k across samples.
    """
    if self._pipeline is None or not query:
        return 1.0
    
    samples = []
    for _ in range(num_samples):
        hits = self.search(query, k=5)
        texts = tuple(h.text for h in hits)
        samples.append(set(texts))
    
    if not samples or not samples[0]:
        return 1.0
    
    # Intersection of all samples
    common = samples[0]
    for s in samples[1:]:
        common &= s
    
    return len(common) / max(1, len(samples[0]))
```

---

## Hypothesis 3: Centroid Recomputation

### Example 3.1: Trigger Recomputation in consolidate()

```python
# core/agent_api.py, in consolidate()

def consolidate(self) -> dict[str, Any]:
    if self._pipeline is None:
        return {"actions": [], "reason": "empty"}
    
    self._metrics["consolidations"] += 1
    engine = self._pipeline.engine
    store = self._pipeline.store
    nodes = store.all_nodes()
    
    # Existing tension scan
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
    
    # NEW: Centroid recomputation on information density shift
    recompute_actions = []
    if self._settings.agent.auto_tune_centroids:
        recompute_actions = self._recompute_stale_centroids()
        actions.extend(recompute_actions)
    
    return {
        "actions": actions,
        "reason": "coherent" if not actions else "consolidated",
    }

def _recompute_stale_centroids(self) -> list[dict]:
    """Recompute centroids for nodes with high member-count drift."""
    if self._pipeline is None:
        return []
    
    engine = self._pipeline.engine
    store = self._pipeline.store
    enc = self._pipeline._encoder
    
    actions = []
    density_threshold = getattr(self._settings.agent, "info_density_threshold", 0.2)
    
    for node in store.all_nodes():
        if not node.members:
            continue
        
        # Check if recomputation is needed
        last_count = getattr(node, "last_fit_member_count", len(node.members))
        member_growth = (len(node.members) - last_count) / max(1, last_count)
        
        if member_growth < density_threshold:
            continue
        
        # Gather member embeddings
        from .interfaces import phrase_to_text_index
        member_embs = []
        for m in node.members:
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None:
                try:
                    text = self._texts[idx]
                    emb = enc.encode([text])[0]
                    member_embs.append(emb)
                except Exception:
                    pass
        
        if len(member_embs) < 2:
            continue
        
        # Recompute centroid
        old_apex = node.apex
        new_apex = engine._lorentz_mean([e for e in member_embs])
        
        # Recompute aperture
        import torch
        p = torch.from_numpy(new_apex).float().unsqueeze(0)
        required = _MIN_APERTURE
        for emb in member_embs:
            c = torch.from_numpy(emb).float().unsqueeze(0)
            angle = float(engine._angle_at(p, c).item())
            required = max(required, angle)
        
        # Update node
        import dataclasses
        updated = dataclasses.replace(
            node,
            apex=new_apex,
            aperture=required + _EPS,
            last_fit_member_count=len(node.members),
        )
        store.upsert(updated)
        
        actions.append({
            "op": "recompute_centroid",
            "node_id": str(node.id),
            "member_growth": member_growth,
            "aperture_delta": required - node.aperture,
        })
    
    # Re-close transitivity
    if actions:
        all_nodes = store.all_nodes()
        edges = [
            (p.id, c.id) for p in all_nodes for c in all_nodes
            if engine.contains(p, c) > 0.0 and p.id != c.id
        ]
        closed = engine.close_transitivity(all_nodes, edges)
        for n in closed:
            store.upsert(n)
    
    return actions
```

### Example 3.2: Settings for Centroid Recomputation

```python
# core/settings.py, in AgentSettings

class AgentSettings(BaseModel):
    usage_weight: float = Field(0.0, ge=0.0)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    octave_fusion: bool = False
    incremental_ingest: bool = True
    consolidate_tension: float = 0.3
    max_query_chars: int = Field(2048, ge=1)
    
    # NEW
    auto_tune_centroids: bool = False
    info_density_threshold: float = 0.2  # Recompute if member count grows > 20%
```

---

## Hypothesis 4: Memory Auto-Tuning

### Example 4.1: MemoryTuner Class

```python
# core/memory_tuner.py (NEW FILE)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import numpy as np


@dataclass
class MemoryConfig:
    """Candidate memory configuration."""
    fact_budget: int
    summary_budget: int
    working_budget: int
    session_budget: int
    recency_lambda: float
    digest_min_members: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_budget": self.fact_budget,
            "summary_budget": self.summary_budget,
            "working_budget": self.working_budget,
            "session_budget": self.session_budget,
            "recency_lambda": self.recency_lambda,
            "digest_min_members": self.digest_min_members,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryConfig:
        return cls(**d)


class MemoryTuner:
    """
    Simple Bayesian-ish optimizer over memory hyperparameters.
    Uses random sampling + best-of-random exploitation.
    """
    
    def __init__(self, search_space: dict[str, tuple[Any, Any]] | None = None):
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
        """Return next config to evaluate."""
        if len(self.history) < 5:
            return self._random_config()
        else:
            # Exploitation: best-of-random
            candidates = [self._random_config() for _ in range(10)]
            best = max(candidates, key=lambda c: self._expected_improvement(c))
            return best
    
    def _random_config(self) -> MemoryConfig:
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
        """Simple EI proxy: distance to best_config (closer = higher)."""
        if self.best_config is None:
            return 0.5
        
        dist = (
            (config.fact_budget - self.best_config.fact_budget) ** 2 +
            (config.summary_budget - self.best_config.summary_budget) ** 2 +
            (config.working_budget - self.best_config.working_budget) ** 2 +
            (config.session_budget - self.best_config.session_budget) ** 2 +
            ((config.recency_lambda - self.best_config.recency_lambda) * 100) ** 2
        )
        
        return 1.0 / (1.0 + np.sqrt(dist))
    
    def observe(self, config: MemoryConfig, recall_at_k: float) -> None:
        """Record outcome."""
        self.history.append((config, recall_at_k))
        if recall_at_k > self.best_score:
            self.best_score = recall_at_k
            self.best_config = config
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "search_space": self.search_space,
            "history": [
                {"config": c.to_dict(), "recall_at_k": float(score)}
                for c, score in self.history
            ],
            "best_score": self.best_score,
            "best_config": self.best_config.to_dict() if self.best_config else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryTuner:
        tuner = cls(data.get("search_space"))
        for item in data.get("history", []):
            cfg = MemoryConfig.from_dict(item["config"])
            tuner.observe(cfg, item["recall_at_k"])
        return tuner
```

### Example 4.2: Integration in KnowledgeBase

```python
# core/agent_api.py

from .memory_tuner import MemoryTuner, MemoryConfig

class KnowledgeBase:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._pipeline: KnowledgePipeline | None = None
        self._texts: list[str] = []
        self._memory = SemioticMemory(None, [], self._settings)
        self._usage: dict[str, int] = {}
        self._metrics: dict[str, int] = {
            "queries": 0, "ingests": 0, "record_outcomes": 0, "consolidations": 0,
        }
        
        # NEW
        self._memory_tuner: MemoryTuner | None = None
        self._labeled_queries: list[tuple[str, set[str]]] | None = None
        if self._settings.agent.auto_tune_memory:
            self._memory_tuner = MemoryTuner()
    
    def set_labeled_queries(self, queries: list[tuple[str, set[str]]]) -> None:
        """Set evaluation queries for memory tuning."""
        self._labeled_queries = queries
    
    def consolidate(self) -> dict[str, Any]:
        """Self-improve: tension + optional memory tuning."""
        if self._pipeline is None:
            return {"actions": [], "reason": "empty"}
        
        self._metrics["consolidations"] += 1
        engine = self._pipeline.engine
        store = self._pipeline.store
        nodes = store.all_nodes()
        
        # Existing tension scan and merges
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
        
        # NEW: Memory tuning
        tuning_result = {}
        if self._memory_tuner and self._labeled_queries:
            tuning_result = self._tune_memory()
        
        return {
            "actions": actions,
            "tuning": tuning_result,
            "reason": "consolidated",
        }
    
    def _tune_memory(self) -> dict[str, Any]:
        """Run one memory tuning trial."""
        from .eval import evaluate_detailed
        
        # Suggest next config
        next_config = self._memory_tuner.suggest_next()
        
        # Apply config (create a temporary KB with that config)
        old_memory = self._memory
        temp_settings = self._settings.model_copy()
        # TODO: Apply next_config to temp_settings.memory
        self._memory = SemioticMemory(None, list(self._texts), temp_settings)
        
        # Evaluate
        eval_result = evaluate_detailed(self, self._labeled_queries, k=5)
        recall = eval_result.get("recall_at_k", 0.0)
        
        # Observe
        self._memory_tuner.observe(next_config, recall)
        
        # Optionally revert to best config or keep new one
        if self._memory_tuner.best_config:
            best_settings = self._settings.model_copy()
            # TODO: Apply best_config to best_settings.memory
            self._memory = SemioticMemory(None, list(self._texts), best_settings)
        
        return {
            "op": "memory_tune",
            "trials": len(self._memory_tuner.history),
            "best_score": self._memory_tuner.best_score,
            "last_recall": recall,
        }
```

### Example 4.3: Settings for Memory Tuning

```python
# core/settings.py, in AgentSettings

class AgentSettings(BaseModel):
    # ... existing ...
    
    # NEW
    auto_tune_memory: bool = False
    memory_tune_interval: int = 10  # Consolidations between tuning trials
```

---

## Tests: Copy-Paste Skeleton

### test_diagnostics.py

```python
# core/test_diagnostics.py (NEW FILE)

import pytest
torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase, DiagnoseReport
from core.eval import recall_at_k_per_octave, evaluate_detailed
from core.settings import Settings
from core.cone_engine import HyperbolicConeEngine, ConeFitConfig
from core.memory_tuner import MemoryTuner, MemoryConfig


class TestOctaveDiagnostics:
    def test_diagnose_includes_octave_stats(self):
        kb = KnowledgeBase(Settings())
        kb.ingest([f"fact_{i}" for i in range(32)])
        
        rep = kb.diagnose()
        assert isinstance(rep, DiagnoseReport)
        assert "octave_stats" in rep.__dict__
        assert len(rep.octave_stats) > 0
        
        for prefix, stats in rep.octave_stats.items():
            assert "mean_aperture" in stats
            assert "std_aperture" in stats
            assert "redundancy_rate" in stats
            assert 0.0 <= stats["redundancy_rate"] <= 1.0
    
    def test_diagnose_flags_wide_aperture(self):
        kb = KnowledgeBase(Settings())
        kb.ingest([f"fact_{i}" for i in range(50)])
        
        rep = kb.diagnose()
        # May or may not have flags; just verify structure
        assert isinstance(rep.flags, tuple)
        assert all(isinstance(f, str) for f in rep.flags)
    
    def test_recall_per_octave(self):
        kb = KnowledgeBase(Settings())
        kb.ingest([f"unique_fact_{i}" for i in range(20)])
        
        labeled = [
            ("unique_fact_0", {"unique_fact_0"}),
            ("unique_fact_5", {"unique_fact_5"}),
        ]
        
        recall_o = recall_at_k_per_octave(kb, labeled, k=5)
        assert all(0.0 <= r <= 1.0 for r in recall_o.values())


class TestCentroidCoherence:
    def test_centroid_coherence_shape(self):
        cfg = ConeFitConfig(dim=4, epochs=2)
        engine = HyperbolicConeEngine(cfg)
        
        from core.interfaces import ConeNode, NodeId
        import numpy as np
        
        fake_node = ConeNode(
            id=NodeId("test"), apex=np.random.randn(5).astype(np.float64),
            aperture=0.5, prefix=64, members=("m1", "m2"),
        )
        fake_members = np.random.randn(5, 5).astype(np.float32)
        
        coherence = engine.centroid_coherence(fake_node, fake_members)
        assert "density" in coherence
        assert 0.0 <= coherence["density"] <= 1.0


class TestMemoryTuner:
    def test_suggest_and_observe(self):
        tuner = MemoryTuner()
        
        config1 = tuner.suggest_next()
        assert isinstance(config1, MemoryConfig)
        
        tuner.observe(config1, 0.7)
        assert tuner.best_score == 0.7
        assert tuner.best_config == config1
        
        config2 = tuner.suggest_next()
        tuner.observe(config2, 0.8)
        assert tuner.best_score == 0.8
        assert tuner.best_config == config2
    
    def test_serialization(self):
        tuner = MemoryTuner()
        config = tuner.suggest_next()
        tuner.observe(config, 0.75)
        
        data = tuner.to_dict()
        tuner2 = MemoryTuner.from_dict(data)
        
        assert tuner2.best_score == 0.75
        assert len(tuner2.history) == 1
```

---

## Quick Validation Checklist

After implementing each hypothesis:

### Hypothesis 1 Acceptance Criteria
- [ ] `diagnose()` returns `octave_stats` dict with ≥ 3 entries per octave.
- [ ] `flags` contains at least 1 flag for test data with wide apertures.
- [ ] `evaluate_detailed()` returns per-octave recall; all values in [0, 1].

### Hypothesis 2 Acceptance Criteria
- [ ] `centroid_coherence()` computes for nodes with ≥ 2 members.
- [ ] Coherence values in [0, 1]; empirically correlate with manual inspection.
- [ ] `ranking_stability()` returns stable value for deterministic queries.

### Hypothesis 3 Acceptance Criteria
- [ ] Recomputation triggers when member growth > threshold.
- [ ] Updated apertures are tighter after recompute (node better fits members).
- [ ] Transitive closure re-runs without errors.

### Hypothesis 4 Acceptance Criteria
- [ ] `MemoryTuner` converges to best config in < 20 trials.
- [ ] Best config improves recall@k vs. baseline by ≥ 5%.
- [ ] Tuner state serializes/deserializes correctly.

