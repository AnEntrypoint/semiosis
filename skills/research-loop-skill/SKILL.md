---
name: research-loop-skill
description: >-
  Open-ended research over the semiosis cone KB. You are handed questions you cannot
  answer from where you sit; you go look, bring back what you actually found, and say
  how sure it leaves you. Doing this turn after turn is the work. Use whenever a task is
  exploration rather than a single lookup.
allowed-tools: Skill, Read, Write, Bash(pytest *), Bash(python *)
---

# research-loop

You do not get to decide what to look into. A question arrives, already sharp, about
something in the knowledge base that is loose or unsettled. You cannot resolve it by
thinking -- only by going to the KB and pulling what actually bears on it. You bring back
what you saw and how sure that leaves you. The next question arrives shaped by what just
held and what just fell through. You keep going until nothing pulls anymore.

A prior answer -- a report, an audit, a number someone wrote down -- is not what you saw.
It is one more claim to test, and the conditions it was taken under may no longer hold (or
never matched the real ones). When a question lands on something a prior already "settled",
go take the reading yourself; trust the prior only once your own look agrees with it. The
honest end of a chase is sometimes "the thing I was sent to fix was never the problem" --
bring that back too, it is worth more than an invented change.

The KB primitives you reach for are in `semiosis-skill`; this is the harness that keeps
handing you the next question.

## Run it

```python
from core.agent_api import KnowledgeBase
from core.research_loop import ResearchLoop
from core.kb_types import Observation

kb = KnowledgeBase()
kb.ingest(corpus_texts)          # nothing to research over an empty KB; ingest first
loop = ResearchLoop(kb)
result = loop.run(answer)        # `answer` is the callback below
```

Persist across sessions with `SC_RESEARCH__INSTRUCTION_PERSIST_PATH=<file>`: the questions
that arrive next time start already shaped by everything that held or fell through before.

## The callback

`run` hands your callback one question at a time and waits for what you bring back:

```python
def answer(question):
    if question.target_query:                       # it sent you somewhere to look
        hits = kb.search(question.target_query, k=3)  # or deep_search for a chain
        return Observation(
            directive_stage=question.stage,
            result_text="<what you actually saw>",
            evidence=tuple(h.text for h in hits),
            success_signal=0.85,                      # 0 = it did not hold, 1 = certain
        )
    return Observation(directive_stage=question.stage)  # nothing to fetch; just answer
```

`question.instruction_text` is the question, in plain words -- read it and do what it
asks. `target_query` and `target_octave` name where to look when there is somewhere to
look. Bring back real evidence and an honest `success_signal`; that honesty is what shapes
the next question. Hand back nothing (no `result_text`, no `evidence`) only when you truly
could not look -- a few of those in a row and the run gives up.

## When it stops

`run` returns a `ResearchResult`:
- `converged`: it ran out of pull (or the KB was empty). `False` means it hit the cycle
  cap or you went quiet too many times.
- `hypotheses`: what got staked, each with where it ended up (held, fell through, open).
- `steps`: the trail -- each question, what you brought back, how much the picture moved.
- `refined_instructions`: the questions, reshaped by this run. This is what persists; do
  not regenerate it from scratch, let it accumulate.

Tune the appetite under `SC_RESEARCH__`: `max_cycles`, `convergence_energy_delta`,
`frontier_top_k`, `min_support_score`, `max_no_observation`, `instruction_persist_path`.

## When there is nothing to chase

- Empty KB: `run` returns at once, `converged=True`. Ingest before researching.
- You keep going quiet: bounded, then it stops with `converged=False`.
- It never runs out of pull: the cycle cap stops it, `converged=False`. Never an infinite
  loop.
- A saved trail is unreadable: it starts fresh from the default questions, never crashes.

For a single unsettled query without the full chase, `kb.reflect_directive(query)` hands
you one question to answer the same way.

## Invariants

- No model loads to run this; only the embedding encoder -- sub-4GB by construction.
- The reshaped questions are what carries forward; persist them, do not rebuild them.
- Reach for the KB through `semiosis-skill` (search, deep_search, diagnose, consolidate).
- Tests: `pytest core/` (coverage in core/test_manifold_invariants.py).
- Architecture and budget: docs/auto-research.md.
