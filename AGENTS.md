# AGENTS.md -- hard project rules for semiosis

## Encoding
- No Unicode box-drawing glyphs or arrow symbols anywhere in source; use ASCII only.
- No emoji or decorative non-ASCII symbols.

## Documentation
- No multi-paragraph docstrings; one line max per function/class.
- No filler comments explaining what code does; only WHY when non-obvious.

## Package structure
- Source lives under `core/`; root-level `cone_engine.py`, `interfaces.py`, `settings.py`,
  `test_manifold_invariants.py` are stale copies -- delete them.
- Sub-settings (`EncoderSettings`, `ConeSettings`, `StoreSettings`) are `BaseModel`,
  not `BaseSettings`; only root `Settings` loads from env.

## Memory
- Memory routes through `memorize-fire` (stored in `.gm/rs-learn.db`), never in
  platform dirs (`~/.claude/`, `~/.codex/`, `~/.cursor/`).
- AGENTS.md carries hard rules only; soft preferences go to `memorize-fire`.

## Settings
- Env prefix `SC_`, nested delimiter `__` (e.g. `SC_ENCODER__MODEL=...`).
- Any state = `Settings` snapshot x lakeFS `CommitId` for reproducibility.

## Invariants
- Manifold: Lorentz/hyperboloid (not Poincare ball); numerical guards in rs-learn key `semiosis-stability-invariants`.

## Search
- All code/file/symbol lookups go through the gm spool (`codesearch`), not platform
  search agents, grep, or Glob.

## Tests
- `pytest core/` is the test command; requires `torch` + `geoopt`.
- Tests auto-skip if deps absent; no mocks for integration paths.

## Skills
This project has task-specific skills available.
Before starting any task, read `SKILLS.md` and invoke every relevant skill.
1. Read `SKILLS.md` to discover available skills.
2. Read every skill file plausibly relevant to the task.
3. Invoke with `Skill(skill="semiosis-skill")` before any KB manipulation.

`semiosis-skill` is the primary harness for all KnowledgeBase operations:
search, ingest, diagnose, consolidate, navigate, recall, and memory management.
Skills encode environment-specific constraints that override general knowledge.

@.gm/next-step.md
