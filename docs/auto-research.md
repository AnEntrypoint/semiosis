# Auto-research over semiosis (sub-4GB VRAM)

Combines Karpathy-style auto-research (a self-improving loop: propose hypothesis ->
design experiment -> run -> observe -> refine) with the semiosis hyperbolic-cone
KnowledgeBase, under a hard sub-4GB VRAM budget.

## Core idea: instructions are the model

No generative LLM runs in-process. The `ResearchLoop` (core/research_loop.py) drives
the KB by EMITTING structured `Directive` objects -- prose instructions for the calling
agent (the harness already in the loop) to execute. The agent returns an `Observation`;
the loop folds it back via the KB learning surface. The refined instruction set
(`ResearchResult.refined_instructions`) IS the trained artifact: each run sharpens the
propose/experiment instructions toward confirmed regions and away from refuted ones, and
the set persists across sessions. The agent plays the role the LLM would; the persisted
instructions play the role of the weights.

This is what makes sub-4GB trivial: the only model in process is the embedding encoder.

## Stage -> primitive map

| Karpathy stage    | ResearchLoop stage | semiosis primitive used                          |
| ----------------- | ------------------ | ------------------------------------------------ |
| propose hypothesis| propose            | diagnose + store frontier (max-aperture nodes)   |
| design experiment | experiment         | target query/octave -> deep_search / search      |
| run + observe     | observe            | record_outcome (usage feedback), success scoring |
| refine            | refine             | consolidate (merge/dispel) + instruction rewrite |
| converge          | run loop           | diagnose().total_energy delta < threshold        |

Frontier = highest-aperture (most under-explored) nodes, deduped against tried
hypotheses so the loop never spins. Convergence = the KB total-energy delta between
cycles falling under `convergence_energy_delta`, capped by `max_cycles` (fail loud,
`converged=False`, never an infinite loop).

## Failure paths (explicit)

- Empty KB: run() returns `ResearchResult(converged=True, steps=())` immediately.
- No observation from agent: bounded by `max_no_observation`, then stop with
  `converged=False`.
- Non-convergence: `max_cycles` hard cap, `converged=False` set explicitly.
- Corrupt/missing instruction sidecar: fall back to default instructions, never crash.

## VRAM / RAM budget (component sum)

No generative weights. VRAM is the encoder only; the KB store is host RAM (numpy).

| Component                       | Precision | Footprint        | Location |
| ------------------------------- | --------- | ---------------- | -------- |
| all-MiniLM-L6-v2 (22M params)   | fp32      | ~90 MB           | VRAM/RAM |
| all-MiniLM-L6-v2 (22M params)   | fp16      | ~45 MB           | VRAM     |
| encode activations (batch 128)  | fp32      | ~50-150 MB       | VRAM     |
| ResearchLoop + Directives       | n/a       | < 1 MB           | RAM      |
| KB store (cones, centroids)     | n/a       | ~MBs per 1k docs | host RAM |
| generative LLM                  | none      | 0 MB             | --       |
| TOTAL VRAM ceiling              |           | < 250 MB         | well under 4096 MB |

Set `SC_ENCODER__DEVICE=cpu` for a zero-VRAM run, or `SC_ENCODER__FP16=true` to halve
encoder VRAM on GPU. Headroom under 4GB is ~16x even with fp32 on GPU.

## Settings (SC_RESEARCH__)

`max_cycles`, `convergence_energy_delta`, `frontier_top_k`, `min_support_score`,
`max_no_observation`, `instruction_persist_path`.
