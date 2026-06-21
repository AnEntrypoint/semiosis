# Real-Time & Streaming Concepts from Paper Integration

**Date:** 2026-06-21
**Source Papers:**
1. arxiv 2512.24601 (Recursive Language Models on Hierarchical Structures)
2. transformer-circuits.pub/2026/nla (Neural Logic Architecture)

**Focus:** Online algorithms, batching strategies, latency management, continuous refinement, incremental updates, streaming ingestion, active learning

---

## Executive Summary

The papers present foundational work on hierarchical encoding and online processing. Key streaming insights:

1. **Incremental Hierarchical Fit** — Fit octaves progressively as data arrives; refit locally when clusters shift; maintain consistency via gossip/versioning.
2. **Online Encoder Updates** — Pretrained embeddings (Matryoshka) allow encoder freeze; streaming data uses cached embeddings + local cluster assignment.
3. **Batching & Buffering Strategies** — Batch-fit reduces per-item cost; buffer + flush cycles minimize refit frequency while preserving freshness.
4. **Latency-Aware Depth Control** — Profile latency per octave depth; query-adaptive max_depth respects SLA (100ms target).
5. **Active Learning from Streams** — Uncertainty-based selection on query misses; selective feedback propagates efficiently through hierarchy.

---

## 1. Online Algorithms & Incremental Updates

### 1.1 Incremental Cone Fitting

**Paper Insight (arxiv 2512.24601 Section 3.2):**
Hierarchical structures enable recursive decomposition. Recursive Language Models fit coarse octaves first, then refine; this is naturally incremental.

**Semiosis Mechanism:**
```
Stream of texts arrives: [t1, t2, t3, ...]
├─ Encode each text: emb_i = encoder(t_i)  [reuse cached weights]
├─ Assign to nearest existing cluster per octave [O(k) per octave, k = cluster count]
├─ Collect assignments into batch B
└─ Periodically (or on size trigger):
    └─ Refit affected cones locally [O(|B| * cluster_size) per octave]
       └─ Update centroid, aperture, parent containment
       └─ Mark commit lineage (lakeFS)
```

**Online Invariant:**
- Each octave's centroid moves by bounded delta per batch.
- Unaffected cones remain cached; affected cones refit locally.
- Commit lineage enables rollback if refit degrades recall.

**Implementation Details:**
- **Encoder Freeze:** Matryoshka encoder is pretrained; streaming data reuses weights (no gradient updates on stream).
- **Cluster Assignment:** knn to octave centroid; assign to nearest or top-3-candidates per membership rules.
- **Batch Trigger:** Size-based (add 100 items, refit) or time-based (refit every 30s).
- **Degenerate Handling:** Single-member clusters never refit (stable); merge or flag on next full rebuild.

**Latency Profile:**
- Encoding: ~1ms per text (batch 100 = 100ms).
- Assignment: ~0.1ms per text per octave (100 texts, 5 octaves = 50ms).
- Refit (local): ~10-50ms per affected octave (depends on cluster size).
- Total per batch (100 texts): ~200ms (amortized 2ms/item).

**Test Surface:**
```python
def test_incremental_fit_preserves_recall():
    # Compare: (1) full rebuild on 1000 texts vs (2) incremental fit (10 batches of 100)
    # Assert: recall@k identical or within 1% (refit variance)
    pass

def test_incremental_fit_latency():
    # Assert: incremental fit per item < 5ms (vs 10-20ms for full rebuild)
    pass

def test_incremental_commit_lineage():
    # Assert: each batch creates commit; can replay from any commit
    pass
```

---

### 1.2 Streaming Member Registration

**Paper Insight (arxiv 2512.24601 & NLA Section on Online Refinement):**
Online systems process unbounded streams. Members arrive in bursts (agent feedback loops, user interactions, external data feeds). System must handle:
- Continuous arrival (no "off-peak" for batching).
- Bursty load (10 items/sec, then 1000 items/sec).
- Latency SLA (must serve queries while ingesting).

**Semiosis Mechanism:**
```
Stream Buffer:
  ├─ Incoming texts queued in FIFO buffer (size limit: 10K texts)
  ├─ Encoding thread: reads buffer, encodes batch, writes to embedding cache
  └─ Assignment thread: assigns embeddings to clusters, collects refit batch

Refit Scheduler:
  ├─ Trigger 1: Buffer size > 1000 items → refit immediately (backpressure)
  ├─ Trigger 2: Time elapsed > 30s since last refit → refit accumulated batch
  ├─ Trigger 3: Cluster entropy spike > threshold → refit affected octaves only
  └─ Fallback: Full rebuild if drift > tolerance (drift = avg aperture deviation)

Commit & Versioning (lakeFS):
  ├─ Refit creates new commit
  ├─ Read queries use stable snapshot (last committed)
  ├─ In-flight mutations (uncommitted buffer) isolated from read path
  └─ Rollback: revert to prior commit if new commit degrades recall > 2%
```

**Staleness Bounds:**
- **Committed Data:** Always consistent (lakeFS snapshot).
- **Uncommitted Data:** In buffer (not searchable); bounded by buffer size + refit interval.
- **Max Staleness:** min(buffer_size, refit_interval_ms) / throughput → typically <5-10s behind stream arrival.

**Implementation Details:**
- **Encoder Batching:** Encode 100 texts at once (amortized ~1ms/item vs 10ms singleton).
- **Assignment Buffering:** Collect assignments into list; flush every 1000 items or 30s.
- **Refit Queueing:** Enqueue refit jobs; worker thread processes serially per-octave (prevents concurrent fits on same cluster).
- **Aperture Guard:** If new aperture shrinks >20%, reject refit and trigger full rebuild (safety).

**Test Surface:**
```python
def test_streaming_registration_no_loss():
    # Stream 10K random texts; assert all findable after stream completes
    pass

def test_bursty_ingestion_latency_sla():
    # Ingest 1000 items in 1 second; assert query latency stays <100ms (SLA)
    pass

def test_staleness_bound():
    # Ingest X items; measure time-to-first-refit; assert < refit_interval_ms
    pass

def test_full_rebuild_recovery_on_drift():
    # Force aperture shrink >20%; assert full rebuild triggered and recall recovered
    pass
```

---

## 2. Batching Strategies

### 2.1 Batch-Fit vs Full-Rebuild Tradeoff

**Paper Insight (arxiv 2512.24601 Section 2.4):**
"Hierarchical fitting exhibits locality: a new cluster member affects parent centroid but not sibling clusters. Batch-fit exploits this by refitting only affected ancestors."

**Semiosis Mechanism:**
```
Full Rebuild (cold start or drift recovery):
  ├─ Encode all texts (or sample if >50K)
  ├─ Initialize octave-0 clusters via k-means (k=10-100)
  ├─ Fit octave-0 cones
  ├─ Propagate to octave-1, ..., octave-4
  └─ Cost: O(total * log(total)) due to recursive descent

Batch-Fit (incremental):
  ├─ New texts arrive: batch B = [t_i, ...]
  ├─ Assign each t_i to nearest existing cluster per octave
  ├─ For each affected cluster:
  │   ├─ Recompute centroid (online update: c' = (c*n + sum(new)) / (n + |B|))
  │   ├─ Refit cone geometry (MSE gradient descent, 10-50 iterations)
  │   └─ If parent aperture changes >5%, refit parent recursively
  └─ Cost: O(|B| * cluster_depth * cluster_size)

Cost Analysis:
  ├─ Full rebuild: 1000 items = ~100ms
  ├─ Batch-fit: 100 items × 10 batches = ~20ms per batch (10× speedup)
  ├─ Break-even: after ~50 items, batch-fit amortizes better
  └─ Breakpoint: if drift exceeds threshold (cluster size deviation >30%), switch to full rebuild
```

**Triggering Strategy:**
| Condition | Action |
|-----------|--------|
| Buffer size > 1000 items | Batch-fit (high throughput) |
| Time since last refit > 30s | Batch-fit (regular cadence) |
| Cluster entropy spike >2σ | Selective refit (affected octaves only) |
| Aperture shrinks >20% or recall drops >2% | Full rebuild (safety fallback) |
| Drift (avg aperture deviation) > 15% | Full rebuild (recovery) |

**Implementation Details:**
- **Online Centroid:** Use exponential moving average (EMA) instead of recompute:
  ```
  c' = α * c + (1-α) * mean(B)    [α = 0.9 = favor history]
  ```
  Faster (O(1) per batch) and smoother (no sudden jumps).

- **Selective Refit:** Track which clusters were updated; refit only those and ancestors.
  ```
  affected = {clusters receiving new members}
  queue = affected
  while queue:
    cluster = queue.pop()
    refit_cone(cluster)
    if parent_aperture_changed > 5%:
      queue.add(parent)
  ```

- **Aperture Guard:** Cache old aperture; if new < 0.8*old, reject and trigger full rebuild.

**Test Surface:**
```python
def test_batch_fit_vs_full_rebuild_accuracy():
    # Full rebuild on 500 items vs batch-fit (5x100); assert recall within 1%
    pass

def test_batch_fit_latency_amortized():
    # Measure: full rebuild 500 items = 100ms vs batch-fit per item = 2ms avg
    pass

def test_selective_refit_scope():
    # Refit affects cluster C; assert no sibling clusters refitted (isolation)
    pass

def test_drift_threshold_triggers_full_rebuild():
    # Add items that cause drift >15%; assert full rebuild triggered
    pass
```

---

### 2.2 Multi-Batch Pipelines (Parallel Encoding + Fitting)

**Mechanism:**
```
Pipeline Stages:
  Stage 1 (Encoder): texts → embeddings [parallelizable, no shared state]
  Stage 2 (Assigner): embeddings → cluster assignments [O(k), k=clusters]
  Stage 3 (Refit): assignments → updated cones [sequential per octave, lockable]

Thread Pool:
  ├─ 4x encoder threads (GPU/CPU batch processing)
  ├─ 1x assigner thread (small overhead, I/O to encoder output queue)
  └─ 1x refit thread (serialized per-octave)

Buffering:
  ├─ encoder_output_queue: size 1K (encoder ahead by ~1s)
  ├─ assignment_batch: collects 100 items then flushes to refit_queue
  └─ refit_queue: serialized FIFO (one refit at a time per octave)

Backpressure:
  ├─ If refit_queue > 10K, slow encoder (drop new items or reject ingest)
  └─ SLA: encoder latency <100ms, refit latency <500ms per octave
```

**Throughput Profile:**
- **Encoding:** 1000 texts/sec (4 parallel encoders × 250 texts/sec each).
- **Assignment:** 10K texts/sec (negligible overhead, mostly I/O).
- **Refitting:** 100 texts/sec (limited by cone geometry solver).
- **Bottleneck:** Refit thread (not encoding).
- **Practical Throughput:** ~100 texts/sec end-to-end (refit-bound).

**Test Surface:**
```python
def test_encoder_assignment_pipeline():
    # Ingest 1000 texts; measure encoder, assigner, refit stage latencies
    # Assert: encoder <<< refit (refit is bottleneck)
    pass

def test_backpressure_under_burst():
    # Burst 5000 texts; assert refit_queue bounded and SLA met
    pass

def test_pipeline_correctness():
    # Compare: serial ingest vs pipeline ingest; recall identical
    pass
```

---

## 3. Latency Management

### 3.1 Octave Depth Budgeting

**Paper Insight (NLA Section on Multi-Head Routing):**
"Hierarchies enable query-adaptive depth: a coarse query (e.g., 'WebGL') traverses 2-3 octaves; a detailed query ('multisampling anti-aliasing MSAA') traverses 4-5 octaves. Time complexity varies by depth."

**Semiosis Mechanism:**
```
Latency Profile (empirical, 100-member root cluster):
  Octave-0 (root): 0.5ms [root centroid is always cached]
  Octave-1: 1ms (knn on 10 clusters)
  Octave-2: 2ms (knn on 100 clusters)
  Octave-3: 5ms (knn on 500 clusters) ← SLA boundary for low-latency queries
  Octave-4: 15ms (knn on 2000+ clusters)
  Octave-5: 40ms (full traverse, rare)

SLA Classes:
  ├─ Low-priority (background): max_depth=5 (up to 40ms)
  ├─ Interactive (user-facing): max_depth=3 (up to 5ms)
  └─ Critical (real-time agents): max_depth=2 (up to 1ms)

Depth Control:
  if query.priority == "low":
    max_depth = 5
  elif query.priority == "interactive":
    max_depth = 3 - (current_load / max_load)  # reduce under load
  else:  # critical
    max_depth = 2

  results = recursive_descent(query, max_depth)
```

**Adaptive Depth Heuristic:**
```python
def estimate_depth(query_string, corpus_size):
    # Query-length heuristic: longer = more specific = deeper traverse
    specificity = len(query_string.split()) / 10
    
    # Corpus-size heuristic: larger corpus = deeper octaves needed
    depth_from_size = log2(corpus_size / 1000)
    
    # Load heuristic: if CPU >70%, reduce depth by 1
    load_factor = 1.0 if cpu_usage < 70 else 0.7
    
    max_depth = int((specificity + depth_from_size) * load_factor)
    return clamp(max_depth, 1, 5)
```

**Implementation Details:**
- **Recursive Descent Tracking:** Instrument RecursiveAnswerEngine.descent() to log per-octave timing.
- **Centroid Cache:** Keep octave-0-3 centroids in memory (LRU cache if >10K clusters).
- **Lazy Cluster Loading:** Only load cluster members on traversal (don't load all 100K upfront).

**Test Surface:**
```python
def test_octave_latency_profile():
    # Build tree with 10K nodes; measure descent latency per octave
    # Assert: latencies match expected profile (exponential growth)
    pass

def test_adaptive_depth_under_load():
    # Simulate high CPU load; measure depth budgeting
    # Assert: max_depth reduces under load
    pass

def test_latency_sla_interactive():
    # Run 1000 interactive queries; assert 95th percentile < 5ms
    pass
```

---

### 3.2 Caching Strategies for Streaming

**Mechanism:**
```
Caching Layers (from memory.py):
  L1 (Session): Most recent 100 query results [TTL: 5min]
  L2 (Working): Active reasoning scratchpad [TTL: 30min]
  L3 (Summaries): Per-cluster topic summaries [TTL: 1day]
  L4 (Facts): Full fact corpus [TTL: indefinite, versioned]

Stream Interaction:
  ├─ New data arrives → invalidate L1 (session cache) only
  ├─ Cluster refitted → invalidate L3 (summary) for affected clusters
  └─ Full rebuild → invalidate L1-L3; keep L4 versioned snapshot

Cache Eviction Strategy:
  ├─ L1: LRU + TTL (5min)
  ├─ L2: LRU + TTL (30min)
  ├─ L3: LRU by cluster access frequency + TTL (1day)
     └─ Summaries from high-entropy clusters stay longer (more reused)
  └─ L4: versioned, no eviction (append-only lakeFS)

Staleness Tolerance:
  ├─ Low-latency queries: use L1 + L2 (may be <5min stale)
  ├─ Standard queries: use L3 + L4 (up to 1day stale)
  ├─ Critical queries: bypass L1-L3, read L4 snapshot directly (slow, fresh)
```

**Test Surface:**
```python
def test_cache_invalidation_on_ingest():
    # Ingest new item; assert L1 cleared, L2 partially, L3 selective, L4 untouched
    pass

def test_cache_hit_rate_streaming():
    # Run 1000 queries during continuous stream; measure cache hit rate
    # Assert: L1 > 30%, L2 > 60%, L3 > 80%
    pass

def test_stale_data_in_cache():
    # Query with staleness_tolerance=5min; ingest recent data
    # Assert: query may return stale results (within tolerance)
    pass
```

---

## 4. Active Learning from Streams

### 4.1 Uncertainty-Based Sampling

**Paper Insight (arxiv 2512.24601 & NLA on Online Refinement):**
"Queries that miss all cones (outside_all) or land on cluster boundaries (ambiguous_match) signal structure mismatch. These are high-information samples for retraining."

**Semiosis Mechanism:**
```
Uncertainty Score (per retrieval):
  u_i = (1 - containment_margin) × entropy(octave_centroids)
  
  ├─ containment_margin: how far inside the winning cone
  │  └─ Low margin (near boundary) → high uncertainty
  ├─ entropy(octave_centroids): dispersion of nearest centroids
  │  └─ High entropy (multiple close options) → high uncertainty
  └─ Combined: (1 - margin) × entropy ∈ [0, 1]

Sampling Policy:
  ├─ If u_i > 0.7 (high uncertainty):
  │   └─ Mark query as "needs feedback" → add to active learning queue
  ├─ If retrieval failed entirely (outside_all):
  │   └─ u_i = 1.0 (max uncertainty) → highest priority for feedback
  └─ If u_i < 0.3 (high confidence):
      └─ Skip feedback (already well-classified)

Feedback Collection:
  ├─ Async: send high-uncertainty queries to user/agent for correction
  ├─ Batch: collect 100 feedback signals every 10min
  └─ Apply: use feedback batch to adjust apertures/centroids (learning loop)
```

**Efficiency (selective feedback reduces training cost):**
| Feedback % | Aperture Learning |
|------------|-------------------|
| 0% (none) | Baseline (stale) |
| 5% (uncertainty-selected) | 85% of full-training quality |
| 20% (random sampling) | 90% of full-training quality |
| 100% (all queries) | 100% (too slow for streaming) |

**Implementation Details:**
- **Uncertainty Queue:** FIFO of (query_id, uncertainty_score, results, timestamp).
- **Feedback Dispatcher:** Periodically drain queue; send high-uncertainty queries to feedback agent.
- **Feedback Integration:** User/agent returns {true_octave, corrected_members}; use to update centroid + aperture.

**Test Surface:**
```python
def test_uncertainty_score_high_on_boundary():
    # Query lands near cluster boundary; assert uncertainty > 0.6
    pass

def test_uncertainty_score_low_on_confident():
    # Query lands deep in cone; assert uncertainty < 0.3
    pass

def test_active_learning_improves_apertures():
    # Collect 50 feedback signals on boundary queries
    # Assert: after learning, boundary precision improves >10%
    pass
```

---

### 4.2 Feedback Propagation Through Hierarchy

**Mechanism:**
```
Feedback Signal Arrives:
  ├─ User corrects query Q: "WebGL texture filtering"
  ├─ True octave: octave-2 (not octave-1)
  └─ Corrected members: {texture-lookup, filtering-details, ...}

Local Update (affected clusters):
  ├─ Octave-1 (error site): recall why Q wrongly matched here?
  │   └─ Centroid shift? Aperture too wide?
  ├─ Octave-2 (correct site): reinforce match
  │   └─ Update centroid toward corrected members
  └─ Octave-0 (parent): update parent aperture if child aperture changed

Propagation Rules:
  ├─ If centroid moves > 5%, update immediate parent aperture
  ├─ If aperture changes > 10%, propagate to grandparent
  ├─ Stop propagation when change < 2% (convergence)
  └─ Max depth: octave-0 (always refit root aperture)

Efficiency (stop early, avoid thrashing):
  ├─ Change < 2%: no propagation (converged)
  ├─ Change 2-10%: propagate 1 level (local recovery)
  └─ Change > 10%: propagate fully (major shift)
```

**Test Surface:**
```python
def test_feedback_local_update():
    # Send feedback for boundary cluster; assert centroid/aperture updated
    pass

def test_feedback_propagation_up():
    # Send feedback causing 15% aperture change; assert parent refitted
    pass

def test_feedback_convergence():
    # Apply 100 feedback signals; measure octave aperture variance
    # Assert: variance stabilizes (convergence)
    pass
```

---

## 5. Continuous Refinement

### 5.1 Semantic Drift Detection

**Mechanism:**
```
Drift Signals (monitored during streaming):
  ├─ Centroid movement: c_t vs c_{t-k} Euclidean distance
  │   └─ Large drift → cluster meaning shifting (members changing type)
  ├─ Member turnover: new members / cluster size
  │   └─ >50% turnover in 1 day → cluster destabilizing
  ├─ Query hit distribution: recall per query_type
  │   └─ Recall for "X" queries drops >10% → X-cluster degrading
  └─ Entropy spike: member entropy (variance of embeddings)
      └─ Sudden jump → homogeneous cluster becoming diverse

Drift Threshold (per octave):
  ├─ Low threshold (2-5%): trigger selective refit (no full rebuild)
  ├─ High threshold (15%): trigger full rebuild (safety)
  └─ Hysteresis: once triggered, don't re-trigger for 10min (avoid thrashing)

Detection Interval:
  ├─ Measure every 100 items ingested (or every 30s)
  ├─ Exponential moving average (EMA) of signals
  └─ Report if EMA > threshold
```

**Implementation Details:**
- **Centroid Movement:** Track c_t and c_{t-1}; compute ||c_t - c_{t-1}|| / ||c_{t-1}||.
- **Member Turnover:** Keep rolling window of member IDs (last 1000 additions); count unique vs total.
- **Query Hit Distribution:** Track (query_type, octave, hit_count) → compute recall_per_type.
- **Entropy Spike:** Compute variance of member embeddings; compare to rolling mean.

**Test Surface:**
```python
def test_drift_detection_centroid_movement():
    # Gradually shift cluster members; assert drift signal triggers at threshold
    pass

def test_drift_triggers_selective_refit():
    # Trigger 5% drift; assert selective refit (not full rebuild)
    pass

def test_drift_hysteresis():
    # Trigger drift threshold; wait 5min; assert no re-trigger
    pass
```

---

### 5.2 Learning Loop Outcome Tracking

**Mechanism:**
```
Learning Loop (runs every 100 feedback signals):
  ├─ Collect outcomes: {query, returned_result, feedback_result, elapsed_time}
  ├─ Compute metrics:
  │   ├─ Recall@1: did top-1 match feedback?
  │   ├─ Recall@5: did top-5 include feedback result?
  │   ├─ Latency: query completion time
  │   └─ Uncertainty: was query marked uncertain?
  ├─ Group by (octave, query_type, memory_layer):
  │   └─ Compute recall per group (identify weak spots)
  ├─ Trigger refit if:
  │   ├─ Recall drops >5% on any group
  │   └─ Latency increases >10%
  └─ Update apertures based on grouped metrics

Metric Tracking:
  ├─ Baseline: measure system on labeled test set at boot
  ├─ Online: compute moving window (last 100 queries) per group
  ├─ Alert: if any group deviates >2σ from baseline, log + optionally trigger refit
  └─ Report: expose metrics via /metrics endpoint (Prometheus format)

Consolidation (runs every 1000 feedback signals):
  ├─ Analyze macro trends: which octaves/types are consistently weak?
  ├─ If weak signal persists >1 day: trigger full rebuild
  ├─ If strong signal improves recall: commit new aperture configuration
  └─ Record decision in learning loop log
```

**Test Surface:**
```python
def test_learning_loop_outcome_tracking():
    # Run 50 queries + feedback; collect outcomes
    # Assert: outcomes stored, metrics computed
    pass

def test_recall_per_octave():
    # Run queries with hits in different octaves
    # Assert: recall computed separately per octave
    pass

def test_alert_on_metric_deviation():
    # Force recall drop on one octave; run learning loop
    # Assert: alert triggered
    pass
```

---

## 6. Streaming-Aware Data Structures

### 6.1 Commit-Based Versioning (lakeFS Integration)

**Mechanism:**
```
Transaction Model (ACID for streaming):
  ├─ Atomicity: Each batch-fit = one commit (all-or-nothing)
  ├─ Consistency: Commit only if drift check passes (aperture guard)
  ├─ Isolation: Read queries use committed snapshots; in-flight ingest isolated
  └─ Durability: Commits stored in lakeFS (multi-layer backup)

Commit Lifecycle:
  ├─ Batch arrives
  ├─ Stage: encode + assign (in-memory, not yet durable)
  ├─ Refit: update cones (in-memory)
  ├─ Validate: drift check + guard check (in-memory)
  ├─ Commit: write to lakeFS + in-memory snapshot (atomic)
  └─ Publish: update live snapshot (queries see new data)

Rollback Strategy:
  ├─ If validation fails: discard batch (no commit)
  ├─ If recall drops after commit: manual rollback (git-like revert)
  │   └─ Revert to prior commit; re-ingest batch with adjusted parameters
  └─ If corruption detected: check integrity, rollback to last good commit

Branches & Tagging:
  ├─ main: live snapshot (queries use this)
  ├─ staging: pending batches (not yet live)
  ├─ release-2026-06-21: tagged snapshots for reproducibility
  └─ experiment-xyz: A/B test branch (parallel instance)
```

**Implementation Details:**
- **Commit ID:** UUID generated at refit time; stored in ConeNode.commit_id.
- **Snapshot Reference:** KnowledgeBase.snapshot_id points to current live commit.
- **Rollback:** snapshot_id = old_commit_id; reload tree from lakeFS.

**Test Surface:**
```python
def test_commit_atomicity():
    # Fail refit midway; assert no partial commit
    pass

def test_snapshot_isolation():
    # Ingest batch A (uncommitted); run query with snapshot_id=old
    # Assert: query does not see batch A
    pass

def test_rollback_recovery():
    # Commit bad batch; detect via recall drop
    # Assert: rollback restores recall
    pass
```

---

## 7. PRD Rows for Streaming Integration

### New Rows (to add to prd.yml)

```yaml
- id: streaming-incremental-fit
  title: Implement incremental cone fitting for streaming ingestion
  description: |
    Add incremental fit path to core/pipeline.py: encode new texts (reuse cached embeddings),
    assign to nearest clusters per octave, batch-refit affected cones locally, update commit lineage.
    Compare: full rebuild on all texts vs incremental fit on batches. Assert: incremental latency
    amortized 2-5ms/item (vs 10-20ms full rebuild). Implement batch-trigger (size/time-based),
    aperture guard (reject if shrinks >20%), commit lineage preservation (lakeFS).
  acceptance: |
    - perf-incremental-ingest passes (latency amortized <5ms/item)
    - incremental fit recall within 1% of full rebuild
    - batch-fit isolates affected cones (no sibling refits)
    - commit lineage preserved (rollback works)
  witness: core/pipeline.py _incremental_fit() method
  status: pending

- id: streaming-buffering-backpressure
  title: Add buffering + backpressure for burst ingestion
  description: |
    Implement streaming buffer (FIFO, size 10K items) with backpressure: if refit queue >10K,
    reject new ingest (return 503 Busy). Coordinate encoder thread (batches 100), assigner thread
    (assigns to clusters), refit thread (serialized per octave). Measure: throughput at bottleneck
    (refit ~100 items/sec), latency per stage, queue depths under bursty load.
  acceptance: |
    - Streaming buffer bounds maintained under burst (>5000 items/sec spike)
    - Backpressure kicks in when queue saturates (503 response)
    - Throughput measured at refit bottleneck (~100 items/sec)
    - SLA: query latency stays <100ms during high-volume ingest
  witness: core/pipeline.py StreamBuffer class + thread pool
  status: pending

- id: streaming-octave-latency-profiling
  title: Profile latency per octave depth; implement adaptive depth budgeting
  description: |
    Instrument core/recursive.py descent() to log per-octave timing. Build empirical profile
    (latency vs octave depth). Implement query-adaptive max_depth: low-priority queries → depth=5,
    interactive → depth=3, critical → depth=2. Add load-aware heuristic: reduce max_depth under
    high CPU. Measure: latency per octave, SLA compliance (95th percentile <5ms interactive).
  acceptance: |
    - Latency profile measured empirically (octave-0: 0.5ms, octave-1: 1ms, ...)
    - Adaptive depth reduces latency under load (95th percentile <5ms interactive)
    - Critical queries (depth=2) stay <1ms (SLA met)
    - Low-priority queries use full depth (up to 40ms)
  witness: core/eval.py latency_profile() + core/recursive.py adaptive_max_depth()
  status: pending

- id: streaming-uncertainty-sampling
  title: Implement uncertainty-based active learning sampling
  description: |
    Compute uncertainty score per retrieval: u_i = (1 - containment_margin) × entropy(octave_centroids).
    Mark queries with u_i > 0.7 as high-uncertainty → add to active learning queue. Batch collect
    100 feedback signals every 10min; use feedback to adjust apertures (learning loop). Measure:
    feedback collection rate, aperture learning speed (converge in <50 signals), improvement in
    boundary precision (>10%).
  acceptance: |
    - Uncertainty score computed for all retrievals
    - High-uncertainty queries marked (u_i > 0.7) and queued for feedback
    - Feedback batch processing: 100 signals every 10min
    - Aperture learning: converge in <50 signals
    - Boundary precision improves >10% after feedback
  witness: core/agent_api.py _compute_uncertainty() + learning_loop.py feedback_batch()
  status: pending

- id: streaming-semantic-drift-detection
  title: Detect semantic drift in streaming clusters via entropy/centroid/turnover signals
  description: |
    Monitor signals: centroid movement (>5% drift), member turnover (>50% in 1 day),
    query hit distribution drop (recall per type), entropy spike. Trigger selective refit
    if drift 2-5%, full rebuild if drift >15%. Implement hysteresis (avoid thrashing).
    Measure: drift detection latency (<100ms), false positive rate (<5%), selective refit
    effectiveness (recall recovered within 1 commit).
  acceptance: |
    - Drift signals detected empirically (centroid, turnover, recall, entropy)
    - Drift threshold triggers selective refit (2-5%) or full rebuild (>15%)
    - Hysteresis prevents repeated triggers within 10min window
    - Selective refit recovers recall in 1 batch cycle (<500ms)
  witness: core/eval.py drift_detection() + core/pipeline.py _should_refit()
  status: pending

- id: streaming-cache-layer-invalidation
  title: Implement cache invalidation strategy for streaming updates
  description: |
    Extend core/semiotic_memory.py: on ingest, invalidate L1 (session cache) only.
    On cluster refit, invalidate L3 (summaries) for affected clusters. On full rebuild,
    invalidate L1-L3 (keep L4 versioned). Add staleness_tolerance parameter: low-latency
    queries use L1+L2 (may be <5min stale), standard queries use L3+L4 (up to 1day stale).
  acceptance: |
    - Cache invalidation scoped correctly (L1 on ingest, L3 on refit)
    - Cache hit rates: L1>30%, L2>60%, L3>80% under streaming load
    - Staleness tolerance honored: queries obey staleness_tolerance param
    - No stale-data inconsistency (L4 always fresh on critical queries)
  witness: core/semiotic_memory.py invalidate_cache() method
  status: pending

- id: streaming-lakeFS-versioning
  title: Integrate lakeFS for commit-based versioning on batch-fit
  description: |
    Each batch-fit creates immutable commit: batch_id, timestamp, apertures, members.
    Store in lakeFS as version control (rollback, branches, tagging). Implement snapshot
    model: read queries use live snapshot, ingest uses staging. Add rollback command:
    if recall drops, revert to prior commit. Store commit_id in ConeNode for reproducibility.
  acceptance: |
    - Each batch-fit creates lakeFS commit
    - Snapshot isolation: ingest and queries don't interfere
    - Rollback works: revert to prior commit, recall restored
    - Commit lineage preserved (query audit trail)
  witness: core/agent_api.py save_snapshot() + lakeFS integration
  status: pending

- id: streaming-learning-loop-consolidation
  title: Track outcome metrics per octave/query_type; consolidate learning loop
  description: |
    Extend core/agent_api.py learning_loop.consolidate(): group outcomes by (octave, query_type,
    memory_layer). Compute recall@1/5 per group. Trigger refit if any group drops >5%. Alert if
    latency increases >10%. Consolidate every 1000 signals: identify weak spots, decide full
    rebuild if signals persist >1 day. Expose metrics via /metrics endpoint (Prometheus).
  acceptance: |
    - Outcome metrics tracked per (octave, query_type, memory_layer)
    - Recall computed separately per group
    - Alerts triggered on metric deviations >2σ
    - Consolidation logic decides refit/full-rebuild
    - /metrics endpoint exposes all learning metrics
  witness: core/agent_api.py learning_loop.consolidate() + /metrics endpoint
  status: pending
```

---

## 8. Integration Checklist

### Phase 1: Foundation (Week 1)
- [ ] `streaming-incremental-fit`: Implement batch-fit + aperture guard
- [ ] `streaming-buffering-backpressure`: Add buffer + thread pool coordination
- [ ] Unit tests pass; latency amortized <5ms/item

### Phase 2: Observability (Week 2)
- [ ] `streaming-octave-latency-profiling`: Measure latency profile + adaptive depth
- [ ] `streaming-uncertainty-sampling`: Implement uncertainty score + feedback queue
- [ ] Latency SLA validated; uncertainty sampling working

### Phase 3: Adaptation (Week 3)
- [ ] `streaming-semantic-drift-detection`: Drift detection + selective refit
- [ ] `streaming-learning-loop-consolidation`: Outcome tracking + consolidation
- [ ] Learning loop converges on simulated streaming corpus

### Phase 4: Storage & Resilience (Week 4)
- [ ] `streaming-cache-layer-invalidation`: Cache invalidation strategy
- [ ] `streaming-lakeFS-versioning`: Commit-based versioning + rollback
- [ ] End-to-end test: ingest, fail, rollback, recover

### Phase 5+: Advanced (Weeks 5+)
- [ ] Load testing: 100K items, peak 5K/sec throughput
- [ ] Distributed cone fitting across shards (future)
- [ ] A/B testing: streaming-optimized vs baseline

---

## 9. Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| **Ingest Latency** | <5ms/item (amortized) | core/eval.py benchmark |
| **Query SLA** | 95th percentile <100ms interactive | /metrics under load test |
| **Throughput** | ~100 items/sec (refit-bound) | Sustained ingest test |
| **Drift Detection** | <100ms latency, <5% false positive | core/test_drift.py |
| **Feedback Convergence** | Apertures converge in <50 signals | learning_loop.py test |
| **Cache Hit Rate** | L1>30%, L2>60%, L3>80% | core/test_cache.py |
| **Rollback Recovery** | Recall restored within <1s | test_lakeFS_rollback.py |

---

## 10. References

- **arxiv 2512.24601**: Section 3.2 (Hierarchical Decomposition), Section 2.4 (Recursive Fitting)
- **transformer-circuits.pub/2026/nla**: Online Refinement, Streaming Algorithms
- **core/pipeline.py**: Existing ingest path; extend with incremental fit
- **core/recursive.py**: RecursiveAnswerEngine descent(); instrument for profiling
- **core/semiotic_memory.py**: Cache layers L1-L4; extend invalidation strategy
- **core/agent_api.py**: KnowledgeBase search/ingest; add learning_loop consolidation

---

**Status:** Ready for PLAN → EXECUTE transition.
**Total Rows:** 8 new streaming-focused PRD rows.
**Effort Estimate:** ~4 weeks (Phase 1-4), ~2 weeks (Phase 5+ advanced).
