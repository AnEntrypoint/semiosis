@AGENTS.md

# semiosis

Hyperbolic entailment-cone semantic structure over Matryoshka octaves: take any amount
of unstructured text and fold it into an energy-balanced, human-readable hierarchy.

## Module layout

```
core/
  __init__.py
  interfaces.py          -- Protocols: Encoder, HierarchicalClusterer, ConeEmbedder,
                            Store, Labeler, Query + all shared dataclasses
  cone_engine.py         -- HyperbolicConeEngine + tension/flow/energy/dispel ops
  encoder.py             -- Matryoshka encoders + recursive Ward AgglomerativeClusterer
                            (depth/node count grow with the corpus; SC_CLUSTER__*)
  pipeline.py            -- encode -> recursive-cluster -> fit cones -> store; leaf-routed
                            incremental ingest, local splits, tension-gated global rebuild
  store.py               -- InMemoryStore (leaf-scoped centroid knn, children_of/leaves_at)
                            + InMemoryQuery (containment, centroid-arithmetic analogy)
  markdown_store.py      -- primary persistence: browsable markdown folder tree
                            (frontmatter + per-folder README links) + _meta JSON companion
  semiotic_memory.py     -- SemioticMemory: ChatGPT 4-layer memory (facts/summaries/working/session)
  context_pack.py        -- ContextPackBuilder: token-budgeted, overlap-deduped, distance-collapsed
  recursive.py           -- RecursiveAnswerEngine: beam descent through the within-octave tree
  agent_api.py           -- KnowledgeBase: typed search/deep_search/recall/navigate/scan_tension/
                            build_context_pack/compress_context/remember/forget + learning loop
                            (record_outcome/consolidate/diagnose/metrics) + save/load + batch_search
                            + structure_directives/apply_label (caller-delegated intelligence)
  research_loop.py       -- instruction-emitting propose/experiment/observe/refine loop
  eval.py                -- retrieval-quality harness (recall@k, MRR); measure, do not assume
  api.py                 -- FastAPI serving: /health /ready (503 on RandomEncoder degrade)
                            + agent endpoints + /tools manifest
  settings.py            -- Pydantic-settings Settings; sub-models are BaseModel
  test_manifold_invariants.py  -- property-based + integration tests
```

Meaning-flow layer (maps three sources onto the cone structure): ChatGPT memory
(`semiotic_memory.py`), context rot (`context_pack.py`), Recursive Language Models
(`recursive.py`); tension/flow/energy primitives live on `cone_engine.py`.

Agent integration: `docs/agent-guide.md`. Retrieval ranks the query embedding against
per-node embedding centroids (ConeNode.centroid) over the LEAF pool -- the cone math
drives containment/tension/flow, embedding centroids drive relevance. Redundancy checks
everywhere (MMR, context_pack, overlap_nodes) use `centroid_overlap`, never cone
`overlap_score` (seed noise for sibling pairs). Octave cluster ids are prefix-namespaced
(root@64, root.2.1@64) so all Matryoshka octaves coexist in the store; `ConeNode.parent`
carries the tree edge.

## Persistence

`kb.save(dir)` writes the markdown knowledge base (primary, human-readable, grep-able);
`kb.save(x.json)` writes a single-file snapshot. Both restore fitted cones verbatim --
`KnowledgeBase.load` never re-encodes or refits (`pipeline.rebuild_count` stays 0).

## Test

```
pytest core/
```

21 tests. Requires `torch` + `geoopt` (install `.[hyperbolic,dev]`); tests auto-skip if absent.

## Key invariants

- Manifold: Lorentz/hyperboloid (not Poincare ball -- no boundary blowup).
- Stability guards: `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor,
  `_MAX_GRAD_NORM=1.0` tangent-space clip, `stabilize=10` on RiemannianAdam.
- Settings env vars: prefix `SC_`, nested delimiter `__`
  (e.g. `SC_ENCODER__MODEL=...` overrides `settings.encoder.model`).
- Sub-settings (`EncoderSettings`, `ClusterSettings`, `ConeSettings`, `StoreSettings`)
  are `BaseModel`, not `BaseSettings` -- only the root `Settings` loads from env.
- Reproducibility: any state = `Settings` snapshot x `CommitId` (uuid handle; no
  versioned backend yet).
- Encoder degrade is loud: RandomEncoder fallback sets `diagnose().degraded`,
  `/ready` returns 503.

## Rules

- No Unicode box-drawing glyphs or arrow symbols anywhere in source; use ASCII.
- No multi-paragraph docstrings; one line max.
- Every code/file/symbol lookup goes through the gm spool (`codesearch`), not platform search.
- Memory lives in `.gm/` via `memorize-fire`, not in platform memory dirs.
