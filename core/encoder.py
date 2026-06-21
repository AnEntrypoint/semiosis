"""Encoder implementations: RandomEncoder stub + SentenceTransformerEncoder + AgglomerativeClusterer."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .interfaces import ClusterTree, EuclideanVec, NodeId, PhraseId, Prefix


class RandomEncoder:
    """Satisfies the Encoder protocol with deterministic per-text unit vectors; for testing only."""

    def __init__(self, octaves: tuple[int, ...] = (64, 128, 256, 512, 1024), seed: int = 0) -> None:
        self._octaves = octaves
        self._base_seed = seed

    @property
    def dims(self) -> Sequence[Prefix]:
        return [Prefix(d) for d in self._octaves]

    def encode(self, texts: Sequence[str]) -> EuclideanVec:
        dim = max(self._octaves)
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # deterministic per text: hash combines base seed with text content
            rng = np.random.default_rng(self._base_seed ^ (hash(text) & 0xFFFF_FFFF_FFFF_FFFF))
            v = rng.standard_normal(dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            out[i] = v / norm if norm > 0 else v
        return out

    def slice(self, vec: EuclideanVec, prefix: Prefix) -> EuclideanVec:
        return vec[..., :prefix]


class FixedClusterer:
    """HierarchicalClusterer that wraps a pre-built ClusterTree; for pipeline testing."""

    def __init__(self, tree: ClusterTree) -> None:
        self._tree = tree

    def fit(self, vecs: EuclideanVec, prefix: Prefix) -> ClusterTree:
        """Return the fixed tree, ignoring vecs (vecs are unused in stub)."""
        return self._tree


class SentenceTransformerEncoder:
    """Encoder wrapping a sentence-transformers model; renormalizes Matryoshka sub-vectors on slice."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        octaves: tuple[int, ...] = (64, 128, 256, 512, 1024),
        device: "str | None" = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required; install with: pip install sentence-transformers"
            )
        self._model = _ST(model_name, device=device)
        self._octaves = octaves
        self._normalize = normalize

    @property
    def dims(self) -> "list[Prefix]":
        return [Prefix(d) for d in self._octaves]

    def encode(self, texts: "Sequence[str]") -> EuclideanVec:
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        full = max(self._octaves)
        return np.asarray(vecs, dtype=np.float32)[..., :full]

    def slice(self, vec: EuclideanVec, prefix: Prefix) -> EuclideanVec:
        """Slice to prefix dims and re-normalize (Matryoshka sub-vectors need renorm)."""
        sliced = np.asarray(vec)[..., :prefix].astype(np.float32)
        if self._normalize:
            norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
            sliced = np.where(norms > 0, sliced / norms, sliced)
        return sliced


class AgglomerativeClusterer:
    """HierarchicalClusterer via scipy Ward agglomeration; builds a root-over-k-clusters star tree."""

    def __init__(self, n_clusters: int = 16, linkage: str = "ward") -> None:
        self._n_clusters = n_clusters
        self._linkage = linkage

    def fit(self, vecs: EuclideanVec, prefix: Prefix) -> ClusterTree:
        try:
            from scipy.cluster.hierarchy import fcluster, linkage as _linkage
        except ImportError:
            raise RuntimeError("scipy is required for AgglomerativeClusterer; pip install scipy")
        X = np.asarray(vecs, dtype=np.float64)[..., :prefix]
        n = len(X)
        k = min(self._n_clusters, max(1, n - 1))
        Z = _linkage(X, method=self._linkage)
        labels = fcluster(Z, k, criterion="maxclust")
        cluster_ids = {int(c): NodeId(f"cluster_{c}") for c in np.unique(labels)}
        root = NodeId("root")
        edges = tuple((root, cid) for cid in cluster_ids.values())
        assignments = {
            PhraseId(f"doc_{i}"): cluster_ids[int(labels[i])] for i in range(n)
        }
        return ClusterTree(edges=edges, assignments=assignments, prefix=prefix)
