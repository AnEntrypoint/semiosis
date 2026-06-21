# Contributing to semiosis

## Development setup

```
make install        # core + dev tooling (ruff, mypy, pytest)
make install-full   # adds torch (cpu), geoopt, encoder, serving, eval extras
```

## Before you push

`make ci` runs the same gates CI enforces:

```
make lint    # ruff check .
make type    # mypy core (strict)
make test    # pytest core/
make ci      # all of the above + ruff format --check
```

Optional: `pre-commit install` wires `.pre-commit-config.yaml` so the same
checks run on every commit.

## Hard rules (enforced)

- ASCII source only: no Unicode box-drawing, arrows, bullets, or emoji. Use
  `->`, `-`, `[x]`/`[ ]` instead.
- One-line docstrings: no multi-paragraph docstrings; one line max per
  function/class. Comments explain WHY, not WHAT.
- Source lives under `core/`. Sub-settings (`EncoderSettings`, `ConeSettings`,
  `StoreSettings`, ...) are `BaseModel`, not `BaseSettings`; only the root
  `Settings` loads from env.
- Tests are integration-real, not mock-heavy: no mocks on integration paths.
  Heavy-dep tests `pytest.importorskip` their deps and auto-skip when absent.

## Configuration

Settings load from env with prefix `SC_` and nested delimiter `__`
(e.g. `SC_ENCODER__MODEL=sentence-transformers/all-MiniLM-L6-v2`).

## Manifold invariants

The manifold is Lorentz/hyperboloid (not Poincare ball). Numerical guards:
`_EPS=1e-7` arccos clamp, `_MIN_APERTURE=0.1` rad floor, `_MAX_GRAD_NORM=1.0`
tangent-space clip, `stabilize=10` on RiemannianAdam. Do not weaken these.
