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

Requires Python 3.10+, `torch`, `geoopt`.

## Quick start

```python
from core.agent_api import KnowledgeBase

kb = KnowledgeBase()
kb.ingest(["instanced drawing cuts draw calls", "VAO binds all attributes in one call"])

hits = kb.search("draw call optimization", k=3)
print(hits[0].text, hits[0].score)

d = kb.semantic_distance("draw calls", "GPU rasterizer", octave=64)
print(f"distance: {d:.4f}")

direction = kb.compute_direction("VAO", "bufferSubData")
print(direction.direction_vec[:4])

traj = kb.compute_trajectory("WebGL performance", octaves=[64, 128, 256])
print([(t.octave, t.complexity_estimate) for t in traj.steps])
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
| `compress_hierarchy(k)` | Fold hierarchy to k leaves (info bottleneck) |
| `sense_complexity(query)` | Estimate manifold complexity near query |
| `entropy_dispel()` | Find and remove high-entropy nodes |
| `remember(fact, key)` / `recall(query)` | Pinned long-term fact store |

## Architecture

```
core/
  interfaces.py          -- Protocols + dataclasses (ConeNode, ClusterTree, ...)
  cone_engine.py         -- HyperbolicConeEngine: fit, batch_contains, tension, flow
  encoder.py             -- Matryoshka encoder (octave-prefix namespaced)
  store.py               -- InMemoryStore + InMemoryQuery (HNSW-compatible interface)
  serialization.py       -- JSON cone_node_to_dict / cone_node_from_dict
  manifold_ops.py        -- frechet_mean, twonn_intrinsic_dim, lorentz_project
  activation_predictor.py -- NLA ActivationPredictor + sparse activation routing
  semiotic_memory.py     -- 4-layer memory: facts/summaries/working/session
  context_pack.py        -- Token-budgeted, overlap-deduped context packing
  recursive.py           -- RLM octave-descent query decomposition
  agent_api.py           -- KnowledgeBase: all search, direction, agentic methods
  settings.py            -- Pydantic-settings (prefix SC_, delimiter __)
  eval.py                -- Retrieval quality harness: recall@k, MRR
  api.py                 -- FastAPI: /health /ready + /tools manifest
```

## Settings

All settings use prefix `SC_` with nested delimiter `__`:

```
SC_ENCODER__MODEL=sentence-transformers/all-MiniLM-L6-v2
SC_CONE__EPOCHS=10
SC_STORE__MAX_NODES=50000
```

Sub-settings (`EncoderSettings`, `ConeSettings`, `StoreSettings`) are `BaseModel`, not
`BaseSettings` -- only the root `Settings` reads from env.

## Test

```
pytest core/
```

16 tests, single file (`core/test_manifold_invariants.py`, 195 lines). Requires `torch`
and `geoopt`; tests auto-skip if absent.

## Invariants

- Manifold: Lorentz/hyperboloid (not Poincare ball -- no boundary blowup).
- Stability: `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor,
  `_MAX_GRAD_NORM=1.0` tangent-space clip, `stabilize=10` on RiemannianAdam.
- Reproducibility: any state = `Settings` snapshot x lakeFS `CommitId`.
- No Unicode box-drawing or arrow symbols anywhere in source (ASCII only).
