# Semantic-Cones -- Production System Architecture

A mature, modular-monolith implementation of the slimmed semantic-structure design:
Matryoshka prefix-octaves + a single hyperbolic entailment-cone structure that
unifies hierarchy, overlap, and relations. Built as swappable protocol-typed
modules (arxiv 2506.06946 v2/v3 maturity arc -- not a v1 big-ball-of-mud script).

## Module map (each is a Protocol in `core/interfaces.py`)

```
            +-------------+
  texts --> |  Encoder    |  Matryoshka model, prefix-slice -> octaves
            +------+------+
                   v
            +-------------+
            | Clusterer   |  hierarchical tree per octave (BERTopic/TaxoGen-style)
            +------+------+
                   v
            +-------------+
            | ConeEmbedder|  HARDEST NODE -- geoopt.Lorentz + RiemannianAdam
            +------+------+  parent-contains-child = hierarchy; intersection = overlap
                   v
            +-------------+      +--------------+
            |   Store     |<--->|  Labeler     |  optional NLA, fully decoupled
            | HNSW+graph  |      +--------------+
            | lakeFS ver. |
            +------+------+
                   v
            +-------------+
            |   Query     |  knn . containment . analogy . overlap
            +-------------+
```

## Why each production decision (web-witnessed)

| Concern | Decision | Source |
|---|---|---|
| Manifold | Lorentz/hyperboloid, **not** Poincare ball (no boundary blowup) | geoopt, arxiv 2005.02819 |
| Optimizer | RiemannianAdam, tangent-space grad clip, `stabilize=10` | geoopt |
| Octaves | Matryoshka prefix-slicing (nested => consistency free) | arxiv 2205.13147 |
| One structure | entailment cones = hierarchy + overlap + relations | Ganea 2018 (1804.01882), HypCBM 2026 |
| Vector store | Qdrant/pgvector HNSW; hyperbolic pts -> tangent projection | lakefs.io, firecrawl |
| Versioning | lakeFS/Deep Lake commit-IDs; plan index rebuilds; CDC | arxiv 2601.05270 |
| Serving | FastAPI+Pydantic, /health+/ready, canary+tested rollback | emitechlogic |
| Orchestration | Dagster data-aware DAG, selective re-runs | datadef.io |
| Observability | structured logs, drift via distance-over-distributions | arxiv 2108.13557 |
| Eval | recall@k, containment acc, hierarchy precision, MLflow | -- |

## Numerical-stability guarantees (`core/cone_engine.py`)
epsilon-guarded `arccos`, aperture floor (`_MIN_APERTURE`), tangent-space gradient
clipping (`_MAX_GRAD_NORM`), boundary-safe expmap/logmap, periodic manifold
re-projection. Verified by property-based tests asserting expmap/logmap = id and
on-manifold invariance over random inputs (`core/test_manifold_invariants.py`).

## Reproducibility
Any system state = (typed `Settings` snapshot) x (lakeFS `CommitId`). Embeddings,
index manifests, encoder params, and cone params are all versioned; a rebuild is
deterministic from those two handles.

## Build order (hardest node first)
1. `core/cone_engine.py` -- prove Riemannian cone-fit + stability (done).
2. `core/interfaces.py` -- freeze the contracts (done).
3. Encoder + Clusterer against real Matryoshka model.
4. Store with tangent-projection HNSW + lakeFS versioning.
5. Query interface.
6. Serving + Dagster orchestration + observability.
7. Optional NLA labeler last (system is correct without it).

## When to escalate to the heavy NLA/SMACOF/GW stack
Only for interpretability audits that must read **activation internals** -- the one
capability this slim system trades away.
