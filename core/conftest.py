"""Shared pytest fixtures for the semiosis core test suite."""

from __future__ import annotations

import pytest

from core.cone_engine import ConeFitConfig
from core.interfaces import ClusterTree, NodeId, PhraseId, Prefix


@pytest.fixture()
def two_node_tree() -> ClusterTree:
    return ClusterTree(
        edges=((NodeId("root"), NodeId("child")),),
        assignments={PhraseId("p1"): NodeId("root"), PhraseId("p2"): NodeId("child")},
        prefix=Prefix(64),
    )


@pytest.fixture()
def default_cfg() -> ConeFitConfig:
    return ConeFitConfig(epochs=20, dim=4, seed=0)
