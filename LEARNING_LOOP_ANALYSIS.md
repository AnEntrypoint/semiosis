# Learning Loop Analysis: Key Findings

## Status Quo

The semiosis learning loop consists of three methods:

1. **`record_outcome()`**: Accumulates usage counts for retrieved texts that proved useful.
2. **`consolidate()`**: Scans tension (redundancy/contradiction pairs) and applies dispel ops (merge, reparent, summarize).
3. **`diagnose()`**: Returns health snapshot: node count, octave count, mean aperture, mean tension, redundant pairs.

**Observation**: The loop is **local and reactive** — it fixes obvious problems (high tension) but does not adapt the cone structure based on retrieval performance or predict failure modes.

---

## Four Actionable Hypotheses

### 1. **Octave Boundaries Should Adapt to Retrieval Performance**

**Current gap**: Octaves are fixed at boot (`EncoderSettings.octaves = (64, 128, 256, 512, 1024)`). No mechanism adjusts clustering granularity if top-k recall degrades.

**Signal**: Per-octave `recall@k` from evaluation harness.

**Action**: 
- If `recall_at_k[octave_n] < 0.3`, apertures are too wide; recommend re-encoding with tighter embedding dimensions or smaller cluster targets.
- If `recall_at_k[octave_n] > 0.9`, over-clustering; consider coarsening.

**Validation**: Empirical recall@k vs. aperture distribution should show monotonic relationship.

**Cost**: Requires labeled evaluation set; re-fitting is expensive but can be batched/cooldown-gated.

---

### 2. **Entropy Divergence Detects Systematic Retrieval Failures**

**Current gap**: No early-warning signal for retrieval collapse. `consolidate()` only acts on tension; high tension may appear too late.

**Signals**:
- **Centroid-member distance entropy**: When node members cluster loosely around their fitted centroid, future queries will miss marginal members.
- **Ranking instability**: Same query's top-k rotates significantly week-over-week → cone structure is drifting.
- **Coarse-vs-fine gap**: Query embedding position differs greatly between coarse and fine octaves → retrieval is incoherent across scales.

**Action**:
- Compute `centroid_coherence(node)` = 1 / (1 + std_dev(member_to_centroid_distances)).
- If coherence < 0.5 for an octave, trigger `consolidate()` + re-digest top-k queries from that octave.

**Validation**: Coherence should anticorrelate with retrieval failure rate on held-out test set.

**Cost**: Low (on-demand computation); doesn't require labeled data.

---

### 3. **Centroid Recomputation Should Trigger on Information Density Shift**

**Current gap**: After fitting, cone apices are frozen. Incremental ingest appends members but doesn't update centroids. Over time, nodes become "stretched" — the apex no longer represents the member cloud.

**Signal**: Information density = `log2(aperture * member_count)`.

**Action**:
- When a node ingests > 20% new members, recompute centroid via `_lorentz_mean()` on the member embeddings.
- If new centroid is closer to members, shrink aperture; if farther, widen.
- Re-run `close_transitivity()` to maintain containment guarantees.

**Validation**: Post-recomputation, centroid-member distances should decrease; top-k recall should stabilize or improve.

**Cost**: Medium (recomputation is a write; requires upsert + transitive closure). Batch updates to amortize cost.

---

### 4. **Papers Methods Enable Auto-Tuning of Memory Layer Strategies**

**Current gap**: Memory layer allocation is static (`MemorySettings.budget_tokens`, `recency_lambda`, `digest_min_members`). No feedback loop from query performance to memory policy.

**Methods**:
- **Population-Based Training** (Jaderberg et al., 2017): Evaluate N memory configs in parallel; keep top-k by recall@k; mutate survivors.
- **Hyperband** (Li et al., 2018): Bracket-based early stopping; promote winners, kill losers.
- **Bayesian Optimization** (Shahriari et al., 2015): GP-based UCB/EI to select next config efficiently.

**Signal**: `recall@k` under different memory allocation strategies.

**Action**:
- Initialize search space: `fact_budget ∈ [0, 512]`, `recency_lambda ∈ [0.01, 0.3]`, etc.
- Run BO trials; observe recall@k; update GP posterior.
- Every consolidation step, switch to best config seen so far.

**Validation**: Best config should improve recall@k by > 5% vs. baseline in < 20 trials.

**Cost**: High (requires labeled evaluation set and multiple inference runs). Async tuning can mitigate overhead.

---

## Why These Matter

| Hypothesis | Why Now | Triggers When | Fixes |
|-----------|---------|---------------|-------|
| 1. Octave adapt | Cone structure is fixed but real workloads vary | Eval shows per-octave recall diverges | Wide-shot retrieval; over/under-clustering |
| 2. Entropy detect | Failures are often silent (empty top-k, stale nodes) | Coherence drops; ranking rotates; coarse-fine gap widens | Prevents silent degradation; early corrective action |
| 3. Centroid recompute | Incremental ingest stretches old cones; apex drifts | Member count > 1.2x fit count; information density drops | Retrieval creep; members no longer cohere |
| 4. Memory auto-tune | Memory layer is the most opaque feedback loop | Consolidation runs; labeled data available | Suboptimal memory allocation (e.g., too-small working budget) |

---

## Implementation Sequence

**Phase 1 (Low Risk, 1-2 days)**: Expand `diagnose()` to compute per-octave stats and coherence. No behavior change.

**Phase 2 (Low-Med Risk, 2-3 days)**: Add `centroid_coherence()` metric. Enable agents to detect instability.

**Phase 3 (Med-High Risk, 3-4 days)**: Implement octave adaptation logic in `consolidate()`. Gate behind `auto_tune_octaves` flag; agent decides if/when to act.

**Phase 4 (High Risk/Reward, 5-7 days)**: Implement `MemoryTuner` with Bayesian Optimization. Requires labeled queries; async tuning.

---

## Validation via Real Workload

To validate, run on a realistic agent loop:

1. **Baseline**: Standard KB, no learning.
2. **+Phase 1**: Same KB, read `diagnose()` per 100 consolidations. Report: per-octave aperture stats, coherence, flags.
3. **+Phase 2+3**: Enable `auto_tune_centroids=True`. Report: coherence trend, recomputation frequency, recall@k before/after.
4. **+Phase 4**: Provide labeled test set (100-200 queries). Enable `auto_tune_memory=True`. Report: memory config convergence, recall improvement.

Expected outcomes:
- Phase 1: Diagnostics visible; no latency impact.
- Phase 2+3: Coherence correlates with failure rate (R² > 0.6). Recomputation stabilizes recall.
- Phase 4: Best config improves recall@k by 5-15% vs. baseline.

---

## Risk Mitigations

### Phase 1-2 (Low Risk)
- No changes to retrieval paths; read-only.
- Backward compatible; all new fields have defaults.
- Tests verify metrics are computable.

### Phase 3 (Medium Risk)
- Gate behind `auto_tune_octaves: bool = False`.
- Cooldown epoch prevents oscillation.
- Coherence threshold is conservative (only act below 0.5).

### Phase 4 (High Risk)
- Require explicit `set_labeled_queries()` call; no implicit behavior.
- Tuner maintains history; agents can inspect and reject bad configs.
- Best config is applied only after >= 5 trials (stability).
- Memory resets are logged for audit; agents can disable `auto_tune_memory` if unstable.

---

## Next Steps

1. **Implement Phase 1 + 2**: (~3 days)
   - Expand `diagnose()` and add coherence metric.
   - PR: "feat: per-octave diagnostics and centroid coherence" 
   - Acceptance: per-octave stats appear in `diagnose()` output; coherence correlates with manual inspection.

2. **Validate Phase 1 + 2 on internal test set**: (~1-2 days)
   - Run agent loop; inspect `diagnose()` output.
   - Measure: coherence trend, per-octave recall variance.
   - Document findings in `.gm/learning-loop-validation.md`.

3. **Implement Phase 3**: (~3-4 days)
   - Add octave adaptation logic.
   - PR: "feat: adaptive octave boundaries based on retrieval performance"
   - Acceptance: flags appear in `consolidate()` output; A/B test shows no regression.

4. **Implement Phase 4**: (~5-7 days)
   - Add `MemoryTuner` class and integration.
   - PR: "feat: memory layer auto-tuning via Bayesian Optimization"
   - Acceptance: tuner converges in < 20 trials; best config improves recall by >= 5%.

5. **Integrate all phases**: (~2 days)
   - Merge PRs; update docs; create end-to-end example.
   - PR: "docs: learning loop enhancement guide"

---

## References

**Foundational**:
- Ganea et al. (2018): "Hyperbolic Entailment Cones for Learning Hierarchies" (ICML). Core cone fitting and aperture math.

**Tuning & Optimization**:
- Jaderberg et al. (2017): "Population Based Training of Neural Networks" (ICML). PBT meta-learning.
- Li et al. (2018): "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (JMLR). Efficient bandit-based tuning.
- Shahriari et al. (2015): "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (IEEE). BO survey.
- Hansen & Ostermeier (2001): "Completely Derandomized Self-Adaptation in Evolution Strategies" (EA). CMA-ES.

**Information-Theoretic**:
- Cover & Thomas (2006): "Elements of Information Theory" (Wiley). Entropy and divergence.
- Kullback & Leibler (1951): "On Information and Sufficiency" (AMS). KL divergence.

---

## Glossary

- **Octave**: A Matryoshka embedding level; prefix dimension (64, 128, 256, 512, 1024 dims).
- **Cone**: Hyperbolic entailment cone; a node with apex, aperture, and members.
- **Aperture (ψ)**: Half-angle of a cone; controls containment region.
- **Tension**: Overlap with asymmetric containment; high tension = redundancy/contradiction.
- **Centroid**: The fitted apex of a cone; ideally represents its member embeddings.
- **Coherence**: 1 - normalized variance of member-to-centroid distances; high = tight cluster.
- **Recall@k**: Fraction of relevant results in top-k hits; standard IR metric.
- **Consolidate**: Scan tension and apply remediation (merge, reparent, summarize).
- **Dispel**: Tension remediation operation.

---

## Files Affected (If All Hypotheses Implemented)

```
core/
  cone_engine.py           +centroid_coherence(), density-aware recompute
  agent_api.py             +per-octave diagnostics, recomputation trigger, memory tuner integration
  eval.py                  +recall_at_k_per_octave()
  semiotic_memory.py       (consumed by MemoryTuner; no changes)
  settings.py              +auto_tune_* flags, thresholds
  interfaces.py            +ConeNode.last_fit_member_count, last_recompute_ts
  memory_tuner.py          NEW: MemoryTuner, MemoryConfig, Bayesian BO
  test_diagnostics.py      NEW: per-octave stats, coherence, adaptation tests
```

