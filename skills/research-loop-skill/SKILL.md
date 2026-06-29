---
name: research-loop-skill
description: >-
  Drive the semiosis auto-research loop (Karpathy-style self-improving research):
  propose hypothesis -> design experiment -> observe -> refine, with no in-process
  LLM. The ResearchLoop emits Directives YOU execute against the KnowledgeBase; the
  refined instruction set it returns IS the trained artifact. Use whenever a task is
  open-ended research over the cone KB rather than a single retrieval.
allowed-tools: Skill, Read, Write, Bash(pytest *), Bash(python *)
---

# research-loop

**You are the LLM the loop calls.** `ResearchLoop` (core/research_loop.py) runs no
generative model in-process; it emits a `Directive` each step and you, the agent in the
harness, execute it and return an `Observation`. Only the ~90MB embedding encoder loads,
so the whole stack stays sub-4GB by construction. Drive it through the semiosis-skill KB
primitives; this skill is the outer research harness, semiosis-skill is the inner KB API.

## Boot

```python
from core.agent_api import KnowledgeBase
from core.research_loop import ResearchLoop
from core.kb_types import Observation

kb = KnowledgeBase()
kb.ingest(corpus_texts)          # build the cone hierarchy first; empty KB converges at once
loop = ResearchLoop(kb)          # reads Settings.research; pass settings to override
result = loop.run(observe_fn)    # observe_fn is YOU; see below
```

Reload an improving loop: set `SC_RESEARCH__INSTRUCTION_PERSIST_PATH=<file>` so the
refined instructions survive across sessions and the next run starts smarter.

## You are observe_fn

`run(observe_fn)` calls `observe_fn(directive) -> Observation` for each emitted Directive.
Read `directive.stage` and act, then return an `Observation`:

```python
def observe_fn(directive):
    if directive.stage == "experiment":
        hits = kb.search(directive.target_query, k=3)          # or deep_search for multi-hop
        return Observation(
            directive_stage=directive.stage,
            result_text="<your judgment of what the hits show>",
            evidence=tuple(h.text for h in hits),
            success_signal=0.85,                                # in [0,1]; >= min_support_score == supported
        )
    return Observation(directive_stage=directive.stage)         # propose/refine: no evidence needed
```

`directive.instruction_text` is the prose to act on; `target_query`/`target_octave` name
the experiment; `context` carries the frontier region. Returning an empty Observation
(no result_text, no evidence) signals "could not act" -- the loop bounds these by
`max_no_observation` then stops.

## Stage -> action

| directive.stage | what you do                                  | KB primitive                       |
| --------------- | -------------------------------------------- | ---------------------------------- |
| propose         | form a hypothesis about the named region     | (loop picks frontier via diagnose) |
| experiment      | run the named query, gather bearing evidence | kb.search / kb.deep_search         |
| observe         | judge support, set success_signal            | kb.record_outcome (loop calls it)  |
| refine          | restate the sharpest open question           | kb.consolidate (loop calls it)     |

The loop owns frontier selection (highest-aperture nodes, deduped against tried) and the
record_outcome/consolidate calls; you supply only the judgment in each Observation.

## Convergence and the trained artifact

`run()` returns a `ResearchResult`:
- `converged`: True on energy-delta convergence or empty KB; False on max_cycles or
  no-observation cutoff.
- `hypotheses`: list of `Hypothesis` (text, support_score, status supported|refuted|open).
- `steps`: list of `ResearchStep` (cycle, directive, observation, energy_delta).
- `refined_instructions`: the dict that IS the training -- sharpened toward confirmed
  regions, away from refuted ones, persisted if a path is set.

Knobs under `SC_RESEARCH__`: `max_cycles`, `convergence_energy_delta`, `frontier_top_k`,
`min_support_score`, `max_no_observation`, `instruction_persist_path`.

## Degenerate and failure paths

- Empty KB: `run()` returns `converged=True, steps=()` -- ingest before researching.
- No observation returned: bounded by `max_no_observation`, then `converged=False`.
- Non-convergence: `max_cycles` caps the walk, `converged=False` set explicitly (never
  an infinite loop).
- Corrupt/missing persist sidecar: loop falls back to default instructions, never crashes.

## One-shot reflection without the full loop

For a single low-confidence query, skip the loop: `kb.reflect_directive(query)` returns
one observe-stage Directive you execute and feed back, no llm_fn needed.

## Invariants

- No in-process generative model; encoder only -> sub-4GB by construction.
- Instructions are the weights: persist `refined_instructions`, do not re-derive from prose.
- Drive KB ops through semiosis-skill (search, deep_search, diagnose, consolidate).
- Tests: `pytest core/` (research-loop coverage in core/test_manifold_invariants.py).
- See docs/auto-research.md for the architecture and VRAM budget.
