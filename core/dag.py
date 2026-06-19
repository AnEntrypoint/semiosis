"""Dagster DAG skeleton -- encode -> cluster -> fit -> store pipeline."""
from __future__ import annotations

try:
    from dagster import Definitions, asset
    _HAS_DAGSTER = True
except ImportError:  # pragma: no cover
    _HAS_DAGSTER = False

if _HAS_DAGSTER:
    @asset
    def embeddings():
        """Encode texts to Matryoshka vectors; wire a real Encoder here."""
        raise NotImplementedError("wire RandomEncoder or real Matryoshka encoder")

    @asset
    def cluster_tree(embeddings):  # noqa: F811 -- shadowed intentionally as asset input
        """Cluster embeddings into a ClusterTree; wire HierarchicalClusterer here."""
        raise NotImplementedError("wire HierarchicalClusterer.fit here")

    @asset
    def cone_nodes(cluster_tree):  # noqa: F811
        """Fit hyperbolic cones; wire HyperbolicConeEngine.fit here."""
        raise NotImplementedError("wire HyperbolicConeEngine.fit here")

    @asset
    def store_snapshot(cone_nodes):  # noqa: F811
        """Write cones to Store at a new CommitId; wire Store.write here."""
        raise NotImplementedError("wire Store.write and CommitId here")

    defs = Definitions(assets=[embeddings, cluster_tree, cone_nodes, store_snapshot])
