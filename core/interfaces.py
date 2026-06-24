"""Core protocol interfaces -- contracts every module implements."""
from __future__ import annotations

import types
from typing import Mapping, Protocol, runtime_checkable, Sequence, NewType
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

PhraseId = NewType("PhraseId", str)
NodeId = NewType("NodeId", str)
Prefix = NewType("Prefix", int)          # Matryoshka octave = embedding prefix length
CommitId = NewType("CommitId", str)      # lakeFS/Deep Lake reproducibility handle

EuclideanVec = npt.NDArray[np.float32]
LorentzVec = npt.NDArray[np.float64]


def phrase_to_text_index(phrase_id: "PhraseId | str", n_texts: int) -> int | None:
    """Decode a PhraseId's trailing integer to a source-text index, or None if out of range."""
    tail = str(phrase_id).rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    idx = int(tail)
    return idx if 0 <= idx < n_texts else None


@dataclass(frozen=True, slots=True)
class Phrase:
    id: PhraseId
    text: str
    span: tuple[int, int]
    granularity: str


@dataclass(frozen=True, slots=True)
class ConeNode:
    """Apex (Lorentz manifold point) + half-aperture; parent-contains-child is the hierarchy."""
    id: NodeId
    apex: LorentzVec
    aperture: float
    prefix: Prefix
    members: tuple[PhraseId, ...]
    label: str | None = None
    digest: str | None = None      # lightweight summary standing in for members at distance
    pinned: bool = False           # explicit long-term fact, exempt from summary-collapse/eviction
    centroid: tuple[float, ...] | None = None  # embedding-space member mean; retrieval ranks on this

    def __repr__(self) -> str:
        return f"ConeNode(id={self.id!r}, aperture={self.aperture:.4f}, members={len(self.members)})"


@dataclass(frozen=True, slots=True)
class ClusterTree:
    edges: tuple[tuple[NodeId, NodeId], ...]
    assignments: Mapping[PhraseId, NodeId]
    prefix: Prefix

    def __post_init__(self) -> None:
        # Wrap mutable dict in a read-only proxy; frozen=True only blocks reassignment.
        object.__setattr__(self, "assignments", types.MappingProxyType(dict(self.assignments)))


@runtime_checkable
class Encoder(Protocol):
    @property
    def dims(self) -> Sequence[Prefix]: ...
    def encode(self, texts: Sequence[str]) -> EuclideanVec: ...
    def slice(self, vec: EuclideanVec, prefix: Prefix) -> EuclideanVec: ...


@runtime_checkable
class HierarchicalClusterer(Protocol):
    def fit(self, vecs: EuclideanVec, prefix: Prefix) -> ClusterTree: ...


@runtime_checkable
class ConeEmbedder(Protocol):
    """Fits hyperbolic entailment cones from a cluster tree."""
    def fit(self, tree: ClusterTree) -> Sequence[ConeNode]: ...
    def contains(self, parent: ConeNode, child: ConeNode) -> float: ...


@runtime_checkable
class Store(Protocol):
    """Versioned persistence: HNSW over tangent-space projections, exact cone math on retrieval."""
    def write(self, nodes: Sequence[ConeNode], at: CommitId) -> CommitId: ...
    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]: ...
    def upsert(self, node: ConeNode) -> None: ...


@runtime_checkable
class Labeler(Protocol):
    """Optional NLA post-hoc labeler -- the system is correct without it."""
    def label(self, node: ConeNode, members: Sequence[Phrase]) -> str: ...


@runtime_checkable
class Query(Protocol):
    """Unified query surface: knn, containment, analogy, overlap."""
    def knn(self, q: EuclideanVec, k: int, prefix: Prefix) -> Sequence[NodeId]: ...
    def containment_score(self, parent: NodeId, child: NodeId) -> float: ...
    def analogy(self, a: NodeId, b: NodeId, c: NodeId) -> Sequence[NodeId]: ...
    def overlap_nodes(self, node: NodeId, threshold: float) -> Sequence[NodeId]: ...


@runtime_checkable
class Summarizer(Protocol):
    """Generate an opinionated summary of a cone node's member texts for direction grounding."""
    def summarize(self, node_id: str, member_texts: list[str]) -> str: ...
