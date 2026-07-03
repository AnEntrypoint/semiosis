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

## Open-ended research

For research rather than a single lookup, `ResearchLoop(kb).run(answer)` hands you one
question at a time about something in the KB that is loose or unsettled; you go look,
bring back what you found and how sure it leaves you, and the next question arrives shaped
by what just held. `answer(question) -> Observation(evidence=..., success_signal=...)` is
the only thing you supply; no model loads (sub-4GB by construction). The full framing,
the callback contract, and the stopping conditions are in `research-loop-skill`
(`Skill(skill="research-loop-skill")`); the developer-facing architecture is in
docs/auto-research.md.

## Tuning

Env prefix `SC_`, nested `__`. Agent knobs under `SC_AGENT__`: `usage_weight`,
`mmr_lambda`, `octave_fusion`, `hybrid_lexical`, `incremental_ingest`, `consolidate_tension`,
`max_query_chars`. Measure changes with `core.eval.evaluate(kb, labeled, k)`.

`hybrid_lexical=True` fuses a BM25 lexical ranking over ingested texts into the same
RRF accumulator `octave_fusion` uses across Matryoshka prefixes -- exact-keyword and
embedding-similarity signals combine into one ranked list, no separate lexical API.
Store knobs under `SC_STORE__`: `hilbert_partitions` (Hilbert-curve bucket count per
octave, prunes `knn`/`knn_scored` candidate scans once a store octave holds enough
nodes), `catapult_cache_size` (LRU-bounded query-locality shortcut cache size),
`bm25_k1`/`bm25_b` (standard BM25 term-frequency saturation / length-normalization).
