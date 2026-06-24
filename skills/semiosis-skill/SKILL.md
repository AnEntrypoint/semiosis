---
name: semiosis-skill
description: >-
  Drive the semiosis KnowledgeBase: hyperbolic entailment cones over Matryoshka
  octaves. Use it whenever a task requires semantic search, hierarchy navigation,
  context packing, or KB health management. The KB is the single source of truth
  for meaning; state carried only in prose is lost on restart.
allowed-tools: Skill, Read, Write, Bash(pytest *), Bash(python *)
---

# semiosis

**The KB is the state; prose is not.** Retrieval, ingest, memory, and health ops
route through the KnowledgeBase API (`core/agent_api.py`), never narrated.

## Boot

```python
from core.agent_api import KnowledgeBase, QueryPriority, FailureMode
kb = KnowledgeBase()
kb.ingest(texts)        # build the cone hierarchy
```

Reload from disk: `KnowledgeBase.load(path)`.

## Orient before acting

`kb.diagnose()` -> `DiagnoseReport`. Read `failure_mode` first:

- `NONE`: healthy, proceed.
- `OUTSIDE_CONE`: ingest focused texts on target domain, re-diagnose.
- `BOUNDARY_AMBIGUOUS`: call `kb.consolidate()` before search.
- `OVER_COMPRESSED`: use `kb.deep_search()`, ingest more diverse texts.
- `OCTAVE_MISMATCH`: use `kb.deep_search()` for cross-octave queries.

`recovery_suggestions` carries the action string verbatim.

## Retrieval

```python
hits = kb.search(query, k=5, priority=QueryPriority.MEDIUM)
# priority=HIGH: multi-octave, diversity-biased MMR, slower
# priority=LOW:  single coarse octave, fast routing pass
```

Read hits in confidence order:
- `evidence_path_count >= 3` + `uncertainty_score < 0.2`: high-confidence claim.
- `evidence_path_count == 1` + `uncertainty_score > 0.6`: weak, verify or re-query.
- `aperture < 0.3`: specific concept; `aperture > 1.0`: broad prior.
- `local_entropy > 1.0`: ambiguous concept, decompose query.

Multi-hop / causal chain: `kb.deep_search(query, k=5)` (RLM octave descent).

Token-budgeted context: `kb.build_context_pack(query, max_tokens=2000)`.

Adaptive pattern:
1. `search(k=3, priority=LOW)` -- route/identify domain.
2. If `uncertainty_score > 0.5` on top hit, retry with `priority=HIGH`.
3. If still `> 0.5`, ingest more texts (KB gap).

## Memory

```python
kb.remember("key fact")          # pin to long-term facts layer
kb.forget("key fact")            # remove from facts
kb.recall("topic")               # retrieve pinned facts
```

Facts survive save/load; session metadata does not.

## Health and self-improvement

```python
report = kb.consolidate()        # returns ConsolidateReport
# report.merges: redundant pairs resolved
# report.changed: False -> KB is coherent, no action needed
```

Call consolidate when `diagnose().redundant_pairs > nodes // 4`.

Learning loop:
```python
kb.record_outcome(query, useful_texts, useless_texts)  # reinforce retrieval signal
```

Call after every agent interaction that produced a judgment on hit quality.

## Navigation

```python
kb.navigate(node_id, direction="flow_out")  # downstream implications
kb.scan_tension(top_n=10)                   # TensionPair list: boundary conflicts
```

Use `scan_tension` to find semantic conflicts before synthesizing an answer.

## Persistence

```python
kb.save(path)
kb = KnowledgeBase.load(path)
m = kb.metrics()   # queries, ingests, nodes, n_texts, n_facts, consolidations
```

Reproducibility key: Settings snapshot x lakeFS CommitId (see CLAUDE.md).

## Semantic Navigation

Distance, direction, and trajectory in embedding space at any dimensionality.

```python
# Distance between two texts
d = kb.semantic_distance(text_a, text_b, octave=128)  # scalar
dists = kb.semantic_distance(text_a, text_b)           # dict[octave -> distance]
p = kb.best_octave(text_a, text_b)                     # sharpest discriminating prefix

# Direction between two cone nodes
sd = kb.compute_direction(node_id_a, node_id_b, octave=128)
# sd.direction_vec: unit vector; sd.magnitude: centroid separation

# Search in a direction from anchor
results = kb.direction_search(anchor_text, sd.direction_vec, k=5)

# Map sub-concepts from a parent node
dirs = kb.fold_directions(node_id)  # list[{child_id, direction_vec, magnitude, semantic_label}]

# Agentic inference: reflect on low-confidence queries
result = kb.search_with_reflection(query, reflect_fn=lambda q: llm_rephrase(q))
# result["reflected"] is populated only when top hit uncertainty_score > 0.5
```

Adaptive pattern:
1. semantic_distance(a, b) across all octaves -> call best_octave to find sharpest signal.
2. compute_direction(node_a, node_b) -> direction_search to find what lies in that direction.
3. uncertainty_score > 0.5 on search -> call search_with_reflection with a reflect_fn.

## Advanced Primitives

```python
compress_hierarchy(query, max_nodes=10)     # info-bottleneck prune before context packing
sense_complexity(query)                     # TwoNN intrinsic dim; mc.suggested_octave auto-selects
fold_budget(query, max_tokens, texts)       # greedy relevance selection under token budget
sparse_search(query, k, sparsity=0.9)      # NLA sparse attention; zero low-activation nodes
contrastive_direction(text_a, text_b)      # direction vec: what separates A from B
find_analogy(text_a, text_b, text_c)       # A:B::C:X direction arithmetic
concept_boundary(node_a, node_b)           # decision surface; margin < 0.1 -> BOUNDARY_AMBIGUOUS
entropy_dispel(entropy_ceiling=2.0)        # auto-prune high-entropy noise nodes
build_digest_chain(summarizer=None)        # bottom-up hierarchy summarization
attention_score(node_id, query)            # NLA scaled dot-product attention weight
optimal_octave(query, entropy_budget=1.5)  # minimize energy s.t. entropy <= budget
information_content(node_id)               # IC = -log2(aperture/pi); high=specific
agentic_reflect(query, llm_fn=None, max_rounds=3)   # LLM-in-loop recursive search refinement -> list[ReflectStep]
categorical_parent_score(query, k=5)                # summarize+embed parent nodes, rank by similarity -> list[CategoricalParentHit]
activation_embed(text)                              # NLA activation-prediction routing or standard encode -> list[float]
energy_gradient_search(query, max_steps=10)         # minimum-energy path descent through hierarchy -> dict{steps, terminal_node_id, total_energy_drop}
```

Decision additions:
- high local_entropy hits -> entropy_dispel then re-search
- vague query -> sense_complexity -> use mc.suggested_octave
- compound query -> decompose_query -> parallel sub-searches
- analogy/transfer queries -> find_analogy
- context overflow -> fold_budget to prune candidates
- node pair ambiguity -> concept_boundary; low margin -> expect BOUNDARY_AMBIGUOUS
- query is vague, needs iterative focus -> agentic_reflect
- need best parent category for a concept -> categorical_parent_score
- need activation-prediction or standard embedding -> activation_embed
- need efficient concept localization via energy -> energy_gradient_search

## Invariants

- Manifold: Lorentz/hyperboloid; `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor.
- Octave ids are prefix-namespaced (`root@64`); all Matryoshka octaves coexist in store.
- Env override: `SC_ENCODER__MODEL=...` (prefix `SC_`, delimiter `__`).
- Tests: `pytest core/` (requires `torch` + `geoopt`; auto-skip if absent).
