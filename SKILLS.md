# SKILLS.md

Available skills for this project. Read this file, then read every plausibly relevant
skill file before starting any task.

## semiosis-skill

Path: `skills/semiosis-skill/SKILL.md`

Drive the semiosis KnowledgeBase: semantic search, hierarchy navigation, context packing,
KB health management. Use whenever a task reads from or writes to the cone KB, runs
retrieval, or manages agent memory.

Invoke: `Skill(skill="semiosis-skill")`

## research-loop-skill

Path: `skills/research-loop-skill/SKILL.md`

Drive the auto-research loop (Karpathy-style propose/experiment/observe/refine,
instruction-emitting, sub-4GB by construction). Use whenever a task is open-ended research
over the cone KB rather than a single retrieval; the agent executes emitted Directives and
the refined instruction set is the trained artifact.

Invoke: `Skill(skill="research-loop-skill")`
