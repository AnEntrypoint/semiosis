---
key: mem-ab8ee3fde188d9bc-539
ns: default
created: 1782755179850
updated: 1782755179850
---

## Resolved mutable: mut-karpathy-autoresearch-shape

core/research_loop.py ResearchLoop emits Directives (propose/experiment/observe/refine) for the calling agent; agent returns Observation; loop calls record_outcome+consolidate; refined_instructions persist as the trained artifact. Witnessed by exec (converged True, instructions refined) and pytest core/ 18 passed. gm-method lesson: an instruction-emitting loop substitutes the agent-in-the-harness for an in-process model, making the emitted-then-refined prose the persisted weights.
