"""Integration tests for the Dagster DAG wiring (core/dag.py)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core import dag  # noqa: E402

FACTS = [
    "hyperbolic space embeds trees with low distortion",
    "matryoshka embeddings nest coarse to fine in prefix slices",
    "entailment cones encode hierarchy as containment",
    "lorentz manifold avoids poincare ball boundary blowup",
]


def test_dag_helpers_materialize_store_snapshot() -> None:
    from core.settings import Settings
    settings = Settings()
    encoder = dag.build_encoder(settings)
    import numpy as np
    vecs = np.asarray(encoder.encode(FACTS), dtype=np.float32)
    nodes = dag.fit_octave_nodes(vecs, FACTS, list(encoder.dims), settings)
    assert nodes
    # every octave should contribute at least one node
    prefixes = {int(n.prefix) for n in nodes}
    assert prefixes == set(encoder.dims)


def test_dag_assets_run_end_to_end() -> None:
    pytest.importorskip("dagster")
    from dagster import materialize
    result = materialize(
        [dag.embeddings, dag.cone_nodes, dag.store_snapshot],
        run_config={"ops": {"embeddings": {"config": {"texts": FACTS}}}},
    )
    assert result.success
    snap = result.output_for_node("store_snapshot")
    assert snap["node_count"] > 0
    assert snap["commit_id"]
