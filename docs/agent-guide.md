# semiosis agent guide

How an agent uses semiosis as a smart knowledgebase. All entrypoints hide the
manifold; import `from core import KnowledgeBase`.

## Lifecycle

```python
from core import KnowledgeBase
kb = KnowledgeBase()            # defaults; or KnowledgeBase(Settings(...))
kb.ingest(["fact one", "fact two", ...])   # incremental: reuses cached embeddings
```

`ingest` reuses cached embeddings, so adding facts in a loop encodes only the new
texts (not the whole corpus). All Matryoshka octaves are stored distinctly.

## Retrieval

- `search(query, k) -> list[SearchHit]` -- diversified (MMR), scored, with provenance
  (text, score, node_id, octave, members). `search_texts(query, k)` for plain strings.
- `batch_search(queries, k)` -- one model call for many queries.
- `deep_search(query, k) -> DeepSearchResult` -- recursive octave-descent (RLM-style).
- `build_context_pack(query, max_tokens) -> ContextPack` -- token-budgeted, dedup, distanced.
- `explain_retrieval(query, k) -> list[RetrievalStep]` -- why each hit surfaced.
- `recall(query, budget_tokens) -> str` -- layered memory block (facts, summaries, working, session).

Retrieval ranks the query embedding against each node's embedding centroid (not the
cone apex), so relevance is real; the cone math drives containment/tension/flow.

## Learning loop

- `record_outcome(query, useful_texts, useless_texts)` -- feed back what helped; usage
  counts steer future ranking when `Settings.agent.usage_weight > 0`.
- `remember(fact, id)` / `forget(id)` -- pin explicit long-term facts.
- `consolidate()` -- self-improve: merge redundant cones above the tension threshold.
- `diagnose() -> DiagnoseReport` -- health snapshot (nodes, octaves, mean tension, energy).
- `metrics()` -- usage counters.

## Meaning flow

- `navigate(focus_query, k) -> list[FlowNeighbor]` -- neighbors by entailment gradient (up/down).
- `scan_tension(top_n) -> list[TensionPair]` -- worst redundancy/contradiction pairs.
- `compress_context(query, k) -> CompressResult` -- energy-minimizing representative set.

## Continuity

- `kb.save(path)` / `KnowledgeBase.load(path)` -- texts, usage, facts, session, Settings
  snapshot; cones rebuilt deterministically from texts (state = Settings x CommitId).

## Serving

`from core.api import create_app` exposes POST /ingest /search /recall /context_pack
/navigate /deep_search /tension, GET /diagnose /tools (capability manifest) /health
/ready. Install the `serving` extra.

## Tuning

Env prefix `SC_`, nested `__`. Agent knobs under `SC_AGENT__`: `usage_weight`,
`mmr_lambda`, `octave_fusion`, `incremental_ingest`, `consolidate_tension`,
`max_query_chars`. Measure changes with `core.eval.evaluate(kb, labeled, k)`.
