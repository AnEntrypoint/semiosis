# Learning Loop Hypotheses: Octave Boundaries, Retrieval Diagnostics, Centroid Recomputation, Auto-Tuning

## Executive Summary

The semiosis learning loop (`record_outcome`, `consolidate`, `diagnose`) currently feeds back usage signals and resolves redundancy via tension scanning. This analysis identifies four actionable hypotheses to detect failure modes, adapt the cone structure, and auto-tune strategies via papers methods.

---

## Hypothesis 1: Octave Boundaries Should Adapt to Retrieval Performance

### Current State

- Octaves are fixed at initialization (`EncoderSettings.octaves = (64, 128, 256, 512, 1024)`).
- `RecursiveAnswerEngine.descend` treats octave sequence as immutable.
- `beam_k` and `max_depth` govern breadth/depth tradeoffs but do not adapt per-octave.
- `recursive.py:78` stops at `min_aperture_stop=0.1` regardless of prior retrieval success.

### Hypothesis

**Octave boundaries should contract/expand based on empirical recall@k performance per octave.**

When `evaluate(kb, labeled, k=5)` shows:
- `recall_at_k < 0.3` at octave N: evidence suggests clustering is too coarse; recommend narrowing (increase embedding dim or decrease cluster size at that octave).
- `recall_at_k > 0.8` at octave N: evidence suggests over-clustering; recommend coarsening (decrease dim or accept larger clusters).
- High variance across queries: aperture spreads too wide; trigger `summarize_cluster` more aggressively.

### Witness Points

1. **`diagnose()` expansion**: Add per-octave metrics:
   - `octave_apertures`: dict[octave_id, (min, mean, max, std_dev)]
   - `octave_recall`: dict[octave_id, recall@k] (requires labeled test set at consolidation time)
   - `octave_redundancy_rate`: fraction of tension-scan pairs per octave

2. **Signal routing**:
   - Pair `evaluate(kb, labeled)` with `consolidate()` in the learning loop.
   - If `recall_at_k[octave_n] < threshold`, flag `"octave_contraction_candidate"` in `DiagnoseReport`.
   - Agent downstream reads the flag and either ingest more detail-level texts or re-fit with adjusted encoder dims.

3. **Action**: Post-consolidation, if per-octave recall stalls, refit octaves via encoder prefix recomputation (recompute Matryoshka centroids at tighter boundaries).

### Strength

- **Testable**: Empirical recall deltas vs. aperture stats directly correlate.
- **Decoupled**: Does not require new loss terms; uses existing `eval.py` harness.
- **Monotonic**: Contraction/expansion are local decisions per octave; no global re-architecture.

### Risk

- Frequent re-fitting can fragment the store; require cooldown epoch before re-tuning.
- Labeled set must be stable and representative; bad labels corrupt the signal.

---

## Hypothesis 2: Entropy Divergence Detects Systematic Retrieval Failures

### Current State

- `tension_scan()` surfaces high-overlap pairs with ambiguous containment (redundancy, contradiction).
- `consolidate()` merges or re-parents; no early-warning signal for degradation.
- `record_outcome()` tracks usage but not failure rate (false negatives).
- No per-query success metrics; only top-k rank aggregates in `evaluate()`.

### Hypothesis

**Per-octave embedding centroid divergence indicates upcoming retrieval collapse.**

When a node's centroid drifts far from its member embeddings (via incremental ingest), the octave's retrieval coherence decays. Detect this via:

1. **Centroid-member distance entropy**:
   - For each node, compute mean distance from centroid to all member embeddings.
   - High entropy → members cluster loosely around the label → future queries will miss marginal members.
   
2. **Query embedding drift**:
   - Compare query embedding position relative to the coarse octave centroid vs. fine octave centroid.
   - Large gaps suggest the coarse level overshoots or undershoots the true query intent.

3. **Ranking instability**:
   - Track `explain_retrieval()` per query over time; if the same query's top-k rotates > 30% week-over-week, the cone structure is unstable.

### Witness Points

1. **New metric in `engine.py`**:
```python
def centroid_coherence(self, node: ConeNode, member_embeddings: list[np.ndarray]) -> dict:
    """Return (mean_distance, entropy, outlier_count) of members vs. centroid."""
```

2. **Integration in `diagnose()`**:
```python
cohesion_scores: dict[str, float]  # per-octave coherence, aggregated from nodes
retrieval_instability: float        # week-over-week top-k rotation rate
```

3. **Action threshold**: If `cohesion_scores[octave_n] < 0.5`, trigger `consolidate()` + re-digest top-k queries from that octave.

### Strength

- **Early warning**: Entropy divergence precedes hard failures (empty top-k, rank collapse).
- **Actionable**: Points directly to which octave is unstable.
- **Cheap**: Compute on-demand for diagnostics, not in every search path.

### Risk

- Requires historical tracking of embeddings; adds storage burden.
- Entropy is noisy early; needs smoothing (e.g., Kalman filter or rolling average).
- Depends on availability of member embeddings at recall time; may not be cached.

---

## Hypothesis 3: Centroid Recomputation Should Trigger on Information Density Shift

### Current State

- Cone apices are fit once via `cone_engine.fit()` on a static `ClusterTree`.
- Incremental ingest appends texts and re-runs `KnowledgePipeline`, which re-fits the entire cone forest.
- Apertures are frozen after fit; `close_transitivity()` only widens them.
- Node members grow unbounded; no re-clustering when a node's member set exceeds a saturation threshold.

### Hypothesis

**When a node's information density (bits per member) drops below a threshold, recompute its centroid and aperture to reflect the new member distribution.**

Mechanism:
1. **Information density**: `bits_per_member = log2(node_aperture * node_members_count)` or Wasserstein distance of member embeddings from the fitted apex.
2. **Trigger**: When a node ingests > 20% new members, recompute centroid via `_lorentz_mean()` on the new member embeddings.
3. **Aperture adjustment**: If new centroid is closer to member cloud, shrink aperture; if farther, widen.
4. **Transitive closure**: Re-run `close_transitivity()` to maintain containment guarantees.

### Witness Points

1. **New field in `ConeNode`**:
```python
last_fit_member_count: int      # cardinality at fit time
last_recompute_ts: float        # epoch ts of last centroid recompute
```

2. **Integration in `consolidate()`**:
```python
# If (current_members - last_fit_members) / last_fit_members > 0.2:
#   new_apex = engine._lorentz_mean(member_embeddings)
#   aperture_delta = max(distances_to_new_apex)
#   update node in store
```

3. **Test**: Ingest a batch where top-k flips; verify that post-recompute, old top-k and new top-k converge.

### Strength

- **Adaptive**: Cones self-tune as data composition evolves.
- **Incremental**: Recompute only stale nodes, not the entire forest.
- **Testable**: Empirical member-to-centroid distances validate coherence.

### Risk

- Recomputation is a write operation; requires store upsert and transitive-closure reclosure (expensive).
- May oscillate if ingest is bursty; need stabilization (batch rewrites, hysteresis on density threshold).
- Member embeddings must be available; presently not cached in `ConeNode`.

---

## Hypothesis 4: Papers Methods Enable Auto-Tuning of Memory Layer Strategies

### Current State

- `SemioticMemory` maintains four layers: facts, summaries, working, session.
- Layer policies are static: `MemorySettings.budget_tokens`, `digest_min_members`, `recency_lambda`.
- No feedback loop from query success/failure to memory allocation.
- `consolidate()` acts on cone structure, not memory strategy.

### Hypothesis

**Automatically tune memory layer allocation (budget split, recency decay, digest triggers) via a multi-armed bandit or gradient-free optimizer (e.g., CMA-ES, Nelder-Mead) over empirical recall@k.**

### Papers Methods

1. **Population-Based Training (PBT)** (Jaderberg et al., 2017):
   - Initialize N memory configurations (fact_budget, summary_budget, working_budget, session_budget, recency_lambda).
   - Each configuration runs a learning loop iteration: ingest → query → consolidate → evaluate.
   - Keep top-k by recall@k; mutate survivors; restart low performers.
   - Let this run in the background; agents pick the best configuration at consolidation time.

2. **Hyperband / Successive Halving** (Li et al., 2018):
   - Allocate trial budget (epochs, data size) exponentially.
   - Early-stop low-performers; double budget for survivors.
   - Quickly prune bad memory configurations without full training.

3. **Gradient-Free Bayesian Optimization** (Shahriari et al., 2015):
   - Model `f: (budget_allocation, recency_lambda, digest_threshold) -> recall@k` via Gaussian Process.
   - Use Upper Confidence Bound (UCB) or Expected Improvement (EI) to select next config to trial.
   - Converges in O(log N) trials vs. O(N) for grid search.

### Witness Points

1. **New `MemoryTuner` class in `semiotic_memory.py`**:
```python
class MemoryTuner:
    """Bayesian optimizer over memory hyperparameters."""
    def __init__(self, base_config: MemorySettings, search_space: dict):
        self.gp = ...  # GP regressor
        self.history = []  # (config, recall@k) tuples
    
    def suggest_next(self) -> MemorySettings:
        """Return next config to trial."""
    
    def observe(self, config: MemorySettings, recall_at_k: float) -> None:
        """Update GP with observed reward."""
    
    def best_config(self) -> MemorySettings:
        """Return the best config seen so far."""
```

2. **Integration in `agent_api.consolidate()`**:
```python
def consolidate(self) -> dict[str, Any]:
    # ... existing tension scan ...
    
    # Tune memory if enabled
    if self._settings.agent.auto_tune_memory:
        next_config = self._memory_tuner.suggest_next()
        self._memory = SemioticMemory(None, [], self._settings)
        self._memory._config = next_config
        # ... continue with new config
        eval_result = evaluate(self, labeled_queries, k=5)
        self._memory_tuner.observe(next_config, eval_result["recall_at_k"])
```

3. **Configuration space**:
   - `fact_budget: [0, 512]` tokens
   - `summary_budget: [0, 512]` tokens
   - `working_budget: [256, 1024]` tokens
   - `recency_lambda: [0.01, 0.3]` (decay rate)
   - `digest_min_members: [1, 5]` (when to summarize)

### Strength

- **Principled**: Grounded in published optimization methods.
- **Holistic**: Tunes the entire memory stack, not one lever.
- **Async**: Can run in background; convergence not a bottleneck.
- **Self-healing**: Bad configurations are automatically deprioritized.

### Risk

- Requires labeled evaluation set to be stable (for reproducible recall@k); may be expensive.
- Tuner convergence depends on hyperparameter search space; poor bounds can waste trials.
- Memory resets between trials risk forgetting useful facts; need snapshotting or warm-start.

---

## Integration Strategy: Phased Rollout

### Phase 1: Diagnostics (Low Risk)
1. Expand `diagnose()` to include per-octave aperture stats and centroid coherence (Hypothesis 2).
2. Add per-octave recall tracking to `evaluate()` harness.
3. Surface early warnings in `DiagnoseReport` (flag fields: `octave_contraction_candidate`, `retrieval_instability`).
4. **No changes to active retrieval path; read-only expansion.**

### Phase 2: Centroid Recomputation (Medium Risk)
1. Add information-density threshold to `AgentSettings`.
2. Implement `engine.centroid_coherence()` and recompute logic in `consolidate()`.
3. Gate behind feature flag `auto_tune_centroids: bool = False`.
4. A/B test: measure recall@k and centroid-member distance vs. baseline.

### Phase 3: Octave Boundary Adaptation (Medium-High Risk)
1. Pipe per-octave recall deltas into `consolidate()` decision logic.
2. Implement octave re-fit via encoder prefix adjustment (Matryoshka layer freezing).
3. Gate behind `adaptive_octaves: bool = False`.
4. Validate that contraction/expansion stabilize rather than oscillate.

### Phase 4: Memory Layer Auto-Tuning (High Risk, High Reward)
1. Implement `MemoryTuner` with Bayesian Optimization backend.
2. Integrate with `consolidate()` and require labeled evaluation set.
3. Gate behind `auto_tune_memory: bool = False` and validate labeled data availability.
4. Monitor for anomalies (e.g., all trials converge to extreme allocations).

---

## Validation Checklist

For each hypothesis, verify:

1. **Correlation**: Does the proposed metric (aperture stats, centroid distance, density) correlate with downstream recall@k?
2. **Causation**: Does acting on the metric (recompute, adapt) improve recall@k vs. no-op?
3. **Stability**: Do changes converge or oscillate? What is the settling time?
4. **Cost**: What is the compute/storage overhead? Does it scale to 1M+ nodes?
5. **Interpretability**: Can an agent operator read the metric and understand why a decision was made?

---

## Files Touched (if all hypotheses implemented)

- `core/cone_engine.py`: Add `centroid_coherence()`, density-aware aperture adjustment.
- `core/agent_api.py`: Expand `diagnose()`, integrate recomputation and tuner.
- `core/eval.py`: Add per-octave recall tracking.
- `core/semiotic_memory.py`: Add `MemoryTuner` class.
- `core/settings.py`: Add `AgentSettings.auto_tune_centroids`, `auto_tune_memory`, `info_density_threshold`.
- `core/interfaces.py`: Extend `ConeNode` with `last_fit_member_count`, `last_recompute_ts`.
- New: `core/memory_tuner.py` (Bayesian optimizer).
- New: `core/test_learning_loop_hypotheses.py` (integration tests).

---

## Summary Table

| Hypothesis | Signal | Action | Risk | Phase |
|-----------|--------|--------|------|-------|
| 1. Octave boundaries adapt | Per-octave recall@k vs. aperture | Refit octaves or adjust encoder dims | Medium | 3 |
| 2. Entropy divergence detects failures | Centroid-member distance, ranking instability | Trigger consolidate or re-digest | Low | 1 |
| 3. Centroid recomputation on density shift | Node member count, information density | Recompute apex/aperture, re-close transitivity | Medium | 2 |
| 4. Memory auto-tuning via papers methods | Recall@k under different memory configs | Bayesian Optimization (PBT, Hyperband, BO) | High | 4 |

---

## References

- Ganea et al. (2018): "Hyperbolic Entailment Cones for Learning Hierarchies" (ICML). Core cone math.
- Jaderberg et al. (2017): "Population Based Training of Neural Networks" (ICML). PBT for hyperparameter tuning.
- Li et al. (2018): "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (JMLR). Efficient successive halving.
- Shahriari et al. (2015): "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (IEEE). BO survey.

