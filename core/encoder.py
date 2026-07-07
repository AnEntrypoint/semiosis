"""Encoder implementations: RandomEncoder stub + SentenceTransformerEncoder + AgglomerativeClusterer."""
from __future__ import annotations

from typing import Sequence

import hashlib

import numpy as np

from .interfaces import ClusterTree, EuclideanVec, NodeId, PhraseId, Prefix


def _stable_text_hash(text: str) -> int:
    """Process-independent text hash; Python's builtin hash() is salted per-process (PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


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
            rng = np.random.default_rng(self._base_seed ^ (_stable_text_hash(text) & 0xFFFF_FFFF_FFFF_FFFF))
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
        fp16: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required; install with: pip install sentence-transformers"
            )
        self._model = _ST(model_name, device=device)
        if fp16:  # halve VRAM for the sub-4GB target on GPU
            try:
                self._model = self._model.half()
            except Exception:
                pass
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
    """HierarchicalClusterer via recursive scipy Ward splits; depth and node count grow with the corpus."""

    def __init__(self, branching_factor: int = 8, max_leaf_size: int = 16,
                 max_depth: int = 32, linkage: str = "ward") -> None:
        self._branching = max(2, branching_factor)
        self._max_leaf = max(2, max_leaf_size)
        self._max_depth = max(1, max_depth)
        self._linkage = linkage

    def fit(self, vecs: EuclideanVec, prefix: Prefix) -> ClusterTree:
        X = np.asarray(vecs, dtype=np.float64)[..., :prefix]
        n = len(X)
        root = NodeId(f"root@{int(prefix)}")
        edges: list[tuple[NodeId, NodeId]] = []
        assignments: dict[PhraseId, NodeId] = {}
        if n == 0:
            return ClusterTree(edges=(), assignments={}, prefix=prefix)
        self._split(X, np.arange(n), root, 0, int(prefix), edges, assignments)
        return ClusterTree(edges=tuple(edges), assignments=assignments, prefix=prefix)

    def _split(self, X: np.ndarray, idxs: np.ndarray, node_id: NodeId, depth: int,
               prefix: int, edges: list, assignments: dict) -> None:
        if len(idxs) <= self._max_leaf or depth >= self._max_depth or len(idxs) < 2:
            for i in idxs:
                assignments[PhraseId(f"doc_{int(i)}")] = node_id
            return
        try:
            from scipy.cluster.hierarchy import fcluster, linkage as _linkage
        except ImportError:
            raise RuntimeError("scipy is required for AgglomerativeClusterer; pip install scipy")
        k = min(self._branching, len(idxs) - 1)
        Z = _linkage(X[idxs], method=self._linkage)
        labels = fcluster(Z, k, criterion="maxclust")
        uniq = np.unique(labels)
        if len(uniq) < 2:  # degenerate split; stop here as a leaf
            for i in idxs:
                assignments[PhraseId(f"doc_{int(i)}")] = node_id
            return
        base = str(node_id).split("@")[0]
        for c in uniq:
            child = NodeId(f"{base}.{int(c)}@{prefix}")
            edges.append((node_id, child))
            self._split(X, idxs[labels == c], child, depth + 1, prefix, edges, assignments)
