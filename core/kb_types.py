"""Types and enums for the KnowledgeBase public API."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureMode(Enum):
    NONE = "none"
    OUTSIDE_CONE = "outside_cone"
    BOUNDARY_AMBIGUOUS = "boundary_ambiguous"
    OVER_COMPRESSED = "over_compressed"
    OCTAVE_MISMATCH = "octave_mismatch"


@dataclass(frozen=True, slots=True)
class SearchHit:
    text: str
    score: float
    node_id: str
    octave: int
    members: tuple[str, ...] = ()
    aperture: float = 0.0
    local_entropy: float = 0.0
    evidence_path_count: int = 1
    uncertainty_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ConsolidateReport:
    changed: bool
    nodes_before: int
    nodes_after: int
    merges: int
    aperture_updates: int
    dispel_count: int


@dataclass(frozen=True, slots=True)
class FlowNeighbor:
    text: str
    gradient: float
    direction: str


@dataclass(frozen=True, slots=True)
class TensionPair:
    text_a: str
    text_b: str
    tension: float
    kind: str


@dataclass(frozen=True, slots=True)
class DeepSearchResult:
    texts: tuple[str, ...]
    evidence: tuple[str, ...]
    trace: tuple[tuple, ...]


@dataclass(frozen=True, slots=True)
class CompressResult:
    texts: tuple[str, ...]
    energy_reduction: float


@dataclass(frozen=True, slots=True)
class DiagnoseReport:
    nodes: int
    octaves: int
    texts: int
    facts: int
    mean_aperture: float
    mean_tension: float
    total_energy: float
    redundant_pairs: int
    entropy_divergence: float = 0.0
    failure_mode: FailureMode = FailureMode.NONE
    recovery_suggestions: tuple[str, ...] = ()
    degraded: bool = False                 # True when retrieval runs on the RandomEncoder fallback
    fallback_reason: str | None = None     # why the real encoder failed to load


@dataclass(frozen=True, slots=True)
class RetrievalStep:
    text: str
    score: float
    node_id: str
    octave: int
    containment_to_top: float
    tension: float


@dataclass(frozen=True, slots=True)
class SemanticDirection:
    from_node: str
    to_node: str
    octave: int
    direction_vec: tuple[float, ...]
    magnitude: float
    cosine_alignment: float


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    node_id: str
    octave: int
    distance_from_prev: float
    direction_vec: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SemanticTrajectory:
    steps: tuple[TrajectoryStep, ...]
    total_distance: float
    coherence_score: float
    energy_cost: float


@dataclass(frozen=True, slots=True)
class DirectionSearchResult:
    hits: tuple["SearchHit", ...]
    alpha: float
    alignment: float


@dataclass(frozen=True, slots=True)
class CompressedHierarchy:
    """Info-bottleneck pruned node set."""
    retained_nodes: tuple
    dropped_nodes: tuple
    info_retained_ratio: float


@dataclass(frozen=True, slots=True)
class RecursiveAnswerResult:
    """Result of recursive LLM-driven octave descent."""
    answer_nodes: tuple
    depth_reached: int
    energy_total: float
    sub_queries: tuple


@dataclass(frozen=True, slots=True)
class ManifoldComplexity:
    """Intrinsic dimensionality of query neighborhood."""
    intrinsic_dim: float
    suggested_octave: int
    complexity_label: str


@dataclass(frozen=True, slots=True)
class FoldBudgetResult:
    """Energy-aware greedy candidate selection under token budget."""
    included: tuple
    excluded: tuple
    tokens_used: int
    energy_cost: float


@dataclass(frozen=True, slots=True)
class SparseSearchResult:
    """SearchHit with sparse mask weight."""
    hit: SearchHit
    sparse_score: float


@dataclass(frozen=True, slots=True)
class IngestStreamResult:
    """Result of incremental streaming ingest."""
    ingested_count: int
    new_nodes: int
    rebalanced: bool
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ContrastiveDirection:
    """Direction vector representing what separates two concepts."""
    direction_vec: tuple
    contrast_score: float
    octave: int


@dataclass(frozen=True, slots=True)
class QueryDecomposition:
    """Compound query split into sub-queries with octave assignments."""
    sub_queries: tuple
    octave_assignments: tuple
    compound_score: float


@dataclass(frozen=True, slots=True)
class AttentionScore:
    """NLA-style attention weight for a node given a query."""
    node_id: str
    weight: float
    octave: int
    temperature: float


@dataclass(frozen=True, slots=True)
class AnalogyResult:
    """word2vec-style A:B::C:X analogy result."""
    hits: tuple
    direction_used: tuple
    analogy_score: float


@dataclass(frozen=True, slots=True)
class ConceptBoundary:
    """Hyperplane separating two concept clusters."""
    midpoint: tuple
    normal_vec: tuple
    margin: float
    octave: int


@dataclass(frozen=True, slots=True)
class DispelReport:
    """Result of entropy-triggered dispel."""
    dispelled_ids: tuple
    entropy_before: float
    entropy_after: float


@dataclass
class ReflectStep:
    round: int
    query: str
    hits: list
    observation: str


@dataclass
class CategoricalParentHit:
    node_id: str
    summary: str
    embedding_sim: float
    confidence_ratio: float = 1.0


@dataclass
class EnergyStep:
    node_id: str
    energy: float
    octave: int


class SemanticDirectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Directive:
    """Instruction emitted for the calling agent to execute; the agent IS the LLM."""
    stage: str
    instruction_text: str
    cycle: int = 0
    context: tuple[str, ...] = ()
    target_query: str = ""
    target_octave: int = 0
    expected: str = ""


@dataclass(frozen=True, slots=True)
class Observation:
    """Calling agent's reply to a Directive; success_signal in [0,1]."""
    directive_stage: str
    result_text: str = ""
    evidence: tuple[str, ...] = ()
    success_signal: float = 0.0


@dataclass
class Hypothesis:
    """A proposed claim about an under-explored KB region; status set on observe."""
    text: str
    support_score: float = 0.0
    octave: int = 0
    status: str = "open"  # open | supported | refuted


@dataclass
class ResearchStep:
    cycle: int
    directive: Directive
    observation: "Observation | None"
    energy_delta: float = 0.0


@dataclass
class ResearchResult:
    """Outcome of a ResearchLoop run; refined_instructions is the trained artifact."""
    hypotheses: list = None
    steps: list = None
    converged: bool = False
    refined_instructions: dict = None

    def __post_init__(self) -> None:
        if self.hypotheses is None:
            self.hypotheses = []
        if self.steps is None:
            self.steps = []
        if self.refined_instructions is None:
            self.refined_instructions = {}
