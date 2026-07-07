# semiosis

Hyperbolic entailment-cone semantic memory over Matryoshka octaves.
Manages energy and entropy in embedding space by folding semantics to lower dimensions,
creating hierarchies that give agents measurable direction, trajectory, and distance
in embedding space at any chosen dimensionality.

## What it does

- Encodes knowledge as cones on the Lorentz/hyperboloid manifold (k=1); cone containment
  models entailment: parent cone contains child cones, aperture encodes specificity.
- Matryoshka octave prefixes (64, 128, 256, ...) coexist in one store; the sharpest
  discriminating dimension is found automatically via best_octave().
- Semantic direction, trajectory, and distance are first-class operations at any octave.
- Agentic inference loop: search -> reflect (rephrase/decompose/expand) -> re-search,
  optionally with LLM+SBERT hybrid reranking.

## Install

```
pip install -e ".[hyperbolic,dev]"
```

Requires Python 3.11+, `torch`, `geoopt`.

## Quick start

```python
from core.agent_api import KnowledgeBase

kb = KnowledgeBase()
kb.ingest([
    "instanced drawing cuts draw calls",
    "VAO binds all attributes in one call",
    "compressed textures stay compressed on GPU",
    "CPU frustum culling cuts GPU work before the rasterizer",
])

hits = kb.search("draw call optimization", k=3)
print(hits[0].text, hits[0].score)

d = kb.semantic_distance("draw calls", "GPU rasterizer", octave=64)
print(f"distance: {d:.4f}")

# compute_direction/compute_trajectory operate on store node ids, resolved via search;
# distinct node ids require the two concepts to land in different clusters, so a
# small/degenerate corpus (e.g. 2 texts) can collapse both into the same cluster id.
a_id = kb.search("VAO binds all attributes")[0].node_id
b_id = kb.search("compressed textures on GPU")[0].node_id
direction = kb.compute_direction(a_id, b_id)
print(direction.direction_vec[:4])

traj = kb.compute_trajectory("WebGL performance", answer_node_id=b_id)
print([(s.octave, s.distance_from_prev) for s in traj.steps])
```

## Key primitives

| Method | Description |
|---|---|
| `search(query, k)` | MMR-ranked semantic search |
| `semantic_distance(a, b, octave)` | Cosine or Lorentz geodesic distance |
| `compute_direction(a, b)` | Direction vector between two concepts |
| `compute_trajectory(query, octaves)` | Meaning drift across Matryoshka levels |
| `energy_gradient_search(query)` | Follow tension gradient through cone tree |
| `agentic_reflect(query, llm_fn)` | Iterative reflect-and-refine retrieval |
| `hybrid_score(query, texts, llm_fn)` | SBERT + LLM weighted reranking |
| `compress_hierarchy(query, max_nodes)` | Fold hierarchy to max_nodes leaves (info bottleneck) |
| `sense_complexity(query)` | Estimate manifold complexity near query |
| `entropy_dispel()` | Find and remove high-entropy nodes |
| `remember(fact, key)` / `recall(query)` | Pinned long-term fact store |

## Architecture

```
core/
  interfaces.py          -- Protocols + dataclasses (ConeNode, ClusterTree, ...)
  cone_engine.py         -- HyperbolicConeEngine: fit, batch_contains, tension, flow
  encoder.py             -- Matryoshka encoder + recursive Ward clusterer (depth grows with corpus)
  store.py               -- InMemoryStore + InMemoryQuery; leaf-scoped centroid knn over the tree
  serialization.py       -- JSON cone_node_to_dict / cone_node_from_dict
  markdown_store.py      -- primary persistence: browsable markdown folder tree + _meta companion
  manifold_ops.py        -- frechet_mean, twonn_intrinsic_dim, lorentz_project
  semiotic_memory.py     -- 4-layer memory: facts/summaries/working/session
  context_pack.py        -- Token-budgeted, overlap-deduped context packing
  recursive.py           -- beam descent through the within-octave cone tree
  agent_api.py           -- KnowledgeBase: all search, direction, agentic methods
  settings.py            -- Pydantic-settings (prefix SC_, delimiter __)
  eval.py                -- Retrieval quality harness: recall@k, MRR
  api.py                 -- FastAPI: /health /ready + /tools manifest
```

## Persistence: markdown knowledge base

`kb.save("some_dir")` writes the structure as a human-readable, grep-searchable
markdown tree: folders are internal cones, leaf `.md` files carry frontmatter
(name + one-line description) and the member texts; every parent folder has a
`README.md` linking its children (progressive disclosure). The `_meta/` companion
holds one JSON per cone node (every octave), so `KnowledgeBase.load("some_dir")`
restores the fitted structure verbatim -- no re-encoding, no refit.
`kb.save("snapshot.json")` keeps the single-file snapshot path.

Intelligence tasks that need a mind (labeling clusters, adjudicating contradictions)
are delegated to the calling agent: `kb.structure_directives()` emits Directive
objects; the caller answers via `kb.apply_label(node_id, label)`.

## Settings

All settings use prefix `SC_` with nested delimiter `__`:

```
SC_ENCODER__MODEL=nomic-ai/nomic-embed-text-v1.5
SC_CONE__EPOCHS=10
SC_STORE__HILBERT_PARTITIONS=32
```

Sub-settings (`EncoderSettings`, `ConeSettings`, `StoreSettings`) are `BaseModel`, not
`BaseSettings` -- only the root `Settings` reads from env.

## Test

```
pytest core/
```

21 tests, single file (`core/test_manifold_invariants.py`). Requires `torch`
and `geoopt`; tests auto-skip if absent.

## Invariants

- Manifold: Lorentz/hyperboloid (not Poincare ball -- no boundary blowup).
- Stability: `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor,
  `_MAX_GRAD_NORM=1.0` tangent-space clip, `stabilize=10` on RiemannianAdam.
- Hierarchy is real: recursive Ward tree, depth and node count grow with the corpus
  (`SC_CLUSTER__BRANCHING_FACTOR`, `SC_CLUSTER__MAX_LEAF_SIZE`); ingest routes new
  texts to leaves, splits locally on overflow, and only rebuilds globally past
  `SC_CLUSTER__REBALANCE_TENSION`.
- Reproducibility: any state = `Settings` snapshot x uuid `CommitId` (no versioned
  backend yet); the markdown tree restores fitted cones verbatim.
- No Unicode box-drawing or arrow symbols anywhere in source (ASCII only).
