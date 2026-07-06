@AGENTS.md

# semiosis

Hyperbolic entailment-cone semantic structure over Matryoshka octaves.

## Module layout

```
core/
  __init__.py
  interfaces.py          -- Protocols: Encoder, HierarchicalClusterer, ConeEmbedder,
                            Store, Labeler, Query + all shared dataclasses
  cone_engine.py         -- HyperbolicConeEngine + tension/flow/energy/dispel ops
  semiotic_memory.py     -- SemioticMemory: ChatGPT 4-layer memory (facts/summaries/working/session)
  context_pack.py        -- ContextPackBuilder: token-budgeted, overlap-deduped, distance-collapsed
  recursive.py           -- RecursiveAnswerEngine: RLM octave-descent + query decomposition
  agent_api.py           -- KnowledgeBase: typed search/deep_search/recall/navigate/scan_tension/
                            build_context_pack/compress_context/remember/forget + learning loop
                            (record_outcome/consolidate/diagnose/metrics) + save/load + batch_search
  eval.py                -- retrieval-quality harness (recall@k, MRR); measure, do not assume
  api.py                 -- FastAPI serving: /health /ready + agent endpoints + /tools manifest
  settings.py            -- Pydantic-settings Settings; sub-models are BaseModel
  test_manifold_invariants.py  -- property-based + integration tests
```

Meaning-flow layer (maps three sources onto the cone structure): ChatGPT memory
(`semiotic_memory.py`), context rot (`context_pack.py`), Recursive Language Models
(`recursive.py`); tension/flow/energy primitives live on `cone_engine.py`.

Agent integration: `docs/agent-guide.md`. Retrieval ranks the query embedding against
per-node embedding centroids (ConeNode.centroid), not the cone apex -- the cone math
drives containment/tension/flow, embedding centroids drive relevance. Octave cluster ids
are prefix-namespaced (root@64) so all Matryoshka octaves coexist in the store.

## Build order (hardest node first)

1. `core/cone_engine.py` + `core/interfaces.py` -- done
2. Encoder + HierarchicalClusterer (real Matryoshka model) -- done
3. Store (in-memory, Hilbert-bucketed) -- done; HNSW/versioned backend still open
4. Query impl -- done
5. Serving (FastAPI /health /ready) + Dagster DAG + observability -- API done, DAG stubbed
6. Optional NLA Labeler last

## Test

```
pytest core/
```

18 tests. Requires `torch` + `geoopt` (install `.[hyperbolic,dev]`); tests auto-skip if absent.

## Key invariants

- Manifold: Lorentz/hyperboloid (not Poincare ball -- no boundary blowup).
- Stability guards: `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor,
  `_MAX_GRAD_NORM=1.0` tangent-space clip, `stabilize=10` on RiemannianAdam.
- Settings env vars: prefix `SC_`, nested delimiter `__`
  (e.g. `SC_ENCODER__MODEL=...` overrides `settings.encoder.model`).
- Sub-settings (`EncoderSettings`, `ConeSettings`, `StoreSettings`) are `BaseModel`,
  not `BaseSettings` -- only the root `Settings` loads from env.
- Reproducibility: any state = `Settings` snapshot x `CommitId` (uuid handle today; lakeFS-backed versioning not yet implemented).

## Rules

- No Unicode box-drawing glyphs or arrow symbols anywhere in source; use ASCII.
- No multi-paragraph docstrings; one line max.
- Every code/file/symbol lookup goes through the gm spool (`codesearch`), not platform search.
- Memory lives in `.gm/` via `memorize-fire`, not in platform memory dirs.
