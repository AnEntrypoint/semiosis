# semantic-cones (semiosis)

[![ci](https://github.com/AnEntrypoint/semiosis/actions/workflows/ci.yml/badge.svg)](https://github.com/AnEntrypoint/semiosis/actions/workflows/ci.yml)

Hyperbolic entailment-cone semantic structure over Matryoshka octaves. A
protocol-typed, swappable-module knowledge base: Matryoshka prefix-octaves plus a
single hyperbolic entailment-cone structure that unifies hierarchy, overlap, and
relations, with an agent-facing retrieval and memory layer on top.

## Install

```
pip install -e .                       # core (numpy + pydantic only)
pip install -e '.[all]'                # full stack
pip install -e '.[hyperbolic,encoder,serving,dev]'   # common dev set
```

Extras: `hyperbolic` (torch + geoopt cone fitting), `encoder`
(sentence-transformers + scipy), `store`, `serving` (FastAPI), `orchestration`
(Dagster), `eval`, `labels`, `dev`.

## Quickstart

```python
from core import KnowledgeBase

kb = KnowledgeBase()
kb.ingest([
    "Hyperbolic space embeds trees with low distortion.",
    "Matryoshka embeddings nest coarse-to-fine in prefix slices.",
    "Entailment cones encode hierarchy as containment.",
])

for hit in kb.search("how is hierarchy represented?", k=3):
    print(hit.score, hit.octave, hit.text)

pack = kb.build_context_pack("hierarchy", max_tokens=256)
print(pack.render())

kb.remember("The project manifold is Lorentz, not Poincare.")
print(kb.recall("which manifold?"))

kb.save("kb.json")
kb2 = KnowledgeBase.load("kb.json")
```

Without `torch`/`sentence-transformers` installed, the pipeline falls back to a
`RandomEncoder` so the API still runs end-to-end (relevance is degraded).

## Surfaces

- `core/agent_api.py` -- `KnowledgeBase`: search, deep_search, recall, navigate,
  scan_tension, build_context_pack, compress_context, remember/forget, learning
  loop (record_outcome/consolidate/diagnose/metrics), save/load, batch_search.
- `core/api.py` -- FastAPI serving (`/health`, `/ready`, agent endpoints, `/tools`).
- `core/dag.py` -- Dagster assets (encode -> cluster -> fit cones -> store).
- `core/cone_engine.py` -- Lorentz manifold cone fitting plus tension/flow/energy ops.

## Configuration

Settings load from env with prefix `SC_` and nested delimiter `__`
(e.g. `SC_ENCODER__MODEL=sentence-transformers/all-MiniLM-L6-v2`).

## Develop

```
make install        # core + ruff, mypy, pytest
make install-full   # adds torch (cpu) + all extras for the full suite
make lint type test # ruff check, mypy strict, pytest
make ci             # everything CI runs, including ruff format --check
```

`pytest core/` is the test command directly. Tests `importorskip` heavy deps
(`torch`, `geoopt`, ...) and auto-skip when those are absent, so the core suite
runs on a numpy-only install. See `CONTRIBUTING.md` for the full workflow and
project rules.

## See also

- `ARCHITECTURE.md` -- production design and per-decision rationale.
- `docs/agent-guide.md` -- agent integration guide.
- `CONTRIBUTING.md` -- dev setup, gates, and hard project rules.
