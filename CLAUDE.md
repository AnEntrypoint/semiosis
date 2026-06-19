@AGENTS.md

# semiosis

Hyperbolic entailment-cone semantic structure over Matryoshka octaves.

## Module layout

```
core/
  __init__.py
  interfaces.py          -- Protocols: Encoder, HierarchicalClusterer, ConeEmbedder,
                            Store, Labeler, Query + all shared dataclasses
  cone_engine.py         -- HyperbolicConeEngine (ConeEmbedder impl, geoopt.Lorentz)
  settings.py            -- Pydantic-settings Settings; sub-models are BaseModel
  test_manifold_invariants.py  -- property-based + integration tests
```

Root files `cone_engine.py`, `interfaces.py`, `settings.py`, `test_manifold_invariants.py`
are stale copies superseded by `core/`; delete them.

## Build order (hardest node first)

1. `core/cone_engine.py` + `core/interfaces.py` -- done
2. Encoder + HierarchicalClusterer (real Matryoshka model)
3. Store (tangent-projection HNSW + lakeFS versioning)
4. Query impl
5. Serving (FastAPI /health /ready) + Dagster DAG + observability
6. Optional NLA Labeler last

## Test

```
pytest core/
```

Requires `torch` + `geoopt` (install `.[hyperbolic,dev]`); tests auto-skip if absent.

## Key invariants

- Manifold: Lorentz/hyperboloid (not Poincare ball -- no boundary blowup).
- Stability guards: `_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor,
  `_MAX_GRAD_NORM=1.0` tangent-space clip, `stabilize=10` on RiemannianAdam.
- Settings env vars: prefix `SC_`, nested delimiter `__`
  (e.g. `SC_ENCODER__MODEL=...` overrides `settings.encoder.model`).
- Sub-settings (`EncoderSettings`, `ConeSettings`, `StoreSettings`) are `BaseModel`,
  not `BaseSettings` -- only the root `Settings` loads from env.
- Reproducibility: any state = `Settings` snapshot x lakeFS `CommitId`.

## Rules

- No Unicode box-drawing glyphs or arrow symbols anywhere in source; use ASCII.
- No multi-paragraph docstrings; one line max.
- Every code/file/symbol lookup goes through the gm spool (`codesearch`), not platform search.
- Memory lives in `.gm/` via `memorize-fire`, not in platform memory dirs.
