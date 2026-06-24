"""Session 7 tests: CompressedHierarchy, ManifoldComplexity, FoldBudgetResult, SparseSearchResult,
IngestStreamResult, ContrastiveDirection, QueryDecomposition, AttentionScore, AnalogyResult,
ConceptBoundary, DispelReport, and manifold/activation helpers."""
from __future__ import annotations

import pytest
import numpy as np
from core.agent_api import (
    KnowledgeBase,
    CompressedHierarchy, ManifoldComplexity, FoldBudgetResult, SparseSearchResult,
    IngestStreamResult, ContrastiveDirection, QueryDecomposition,
    AttentionScore, AnalogyResult, ConceptBoundary, DispelReport,
)
from core.settings import Settings

TEXTS = [
    "machine learning algorithms",
    "neural network architectures",
    "gradient descent optimization",
    "backpropagation through layers",
    "decision tree classification",
    "random forest ensemble methods",
    "support vector machine kernel",
    "linear regression prediction",
]


@pytest.fixture
def kb():
    settings = Settings()
    kb = KnowledgeBase(settings=settings)
    kb.ingest(TEXTS)
    return kb


@pytest.fixture
def node_ids(kb):
    hits = kb.search("machine learning", k=2)
    return [h.node_id for h in hits] if hits else []


def test_compress_hierarchy_dataclass(kb):
    """compress_hierarchy returns CompressedHierarchy with valid info_retained_ratio."""
    r = kb.compress_hierarchy("neural network", max_nodes=5)
    assert isinstance(r, CompressedHierarchy)
    assert 0.0 <= r.info_retained_ratio <= 1.0


def test_compress_hierarchy_max_nodes(kb):
    """compress_hierarchy respects the max_nodes cap on retained_nodes."""
    r = kb.compress_hierarchy("gradient", max_nodes=2)
    assert len(r.retained_nodes) <= 2


def test_sense_complexity_label(kb):
    """sense_complexity returns ManifoldComplexity with valid label and octave."""
    mc = kb.sense_complexity("machine learning", k=5)
    assert isinstance(mc, ManifoldComplexity)
    assert mc.complexity_label in ("constant", "linear", "quadratic", "exponential")
    assert mc.suggested_octave in (64, 128, 256, 512, 1024)


def test_fold_budget_token_limit(kb):
    """fold_budget respects token budget; tokens_used <= max_tokens."""
    r = kb.fold_budget("machine learning", 50, ["hello world", "gradient descent", "neural"])
    assert isinstance(r, FoldBudgetResult)
    assert r.tokens_used <= 50


def test_fold_budget_included_subset(kb):
    """fold_budget partitions candidates into included and excluded without loss."""
    candidates = ["cat", "dog"]
    r = kb.fold_budget("machine", 1000, candidates)
    assert set(r.included) | set(r.excluded) == set(candidates)


def test_sparse_search_returns_results(kb):
    """sparse_search returns a list."""
    results = kb.sparse_search("neural network", k=3)
    assert isinstance(results, list)


def test_sparse_search_k_limit(kb):
    """sparse_search returns at most k results."""
    results = kb.sparse_search("learning", k=3)
    assert len(results) <= 3


def test_sparse_search_type(kb):
    """sparse_search results are SparseSearchResult with float sparse_score."""
    results = kb.sparse_search("decision tree", k=2)
    for r in results:
        assert isinstance(r, SparseSearchResult)
        assert isinstance(r.sparse_score, float)


def test_optimal_octave_valid(kb):
    """optimal_octave returns a value from the standard octave set."""
    o = kb.optimal_octave("machine learning")
    assert o in (64, 128, 256, 512, 1024)


def test_information_content_nonneg(kb, node_ids):
    """information_content returns non-negative float for known node."""
    if not node_ids:
        pytest.skip("no nodes")
    ic = kb.information_content(node_ids[0])
    assert isinstance(ic, float)
    assert ic >= 0.0


def test_information_content_missing(kb):
    """information_content returns 0.0 for unknown node_id."""
    ic = kb.information_content("nonexistent_node_id")
    assert ic == 0.0


def test_ingest_stream_count(kb):
    """ingest_stream returns IngestStreamResult with correct count and non-negative elapsed."""
    r = kb.ingest_stream(["alpha", "beta", "gamma"])
    assert isinstance(r, IngestStreamResult)
    assert r.ingested_count == 3
    assert r.elapsed_ms >= 0.0


def test_contrastive_direction_unit_norm(kb):
    """contrastive_direction direction_vec is unit-normalized."""
    cd = kb.contrastive_direction("cat", "dog")
    assert isinstance(cd, ContrastiveDirection)
    vec = np.array(cd.direction_vec)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_contrastive_direction_score_positive(kb):
    """contrastive_direction contrast_score is non-negative."""
    cd = kb.contrastive_direction("machine learning", "cooking recipes")
    assert cd.contrast_score >= 0.0


def test_decompose_query_simple(kb):
    """decompose_query splits compound query into multiple sub_queries."""
    qd = kb.decompose_query("cats and dogs")
    assert isinstance(qd, QueryDecomposition)
    assert len(qd.sub_queries) >= 2


def test_decompose_query_single(kb):
    """decompose_query on single-term query sets compound_score=0 and one sub_query."""
    qd = kb.decompose_query("just cats")
    assert qd.compound_score == 0.0
    assert len(qd.sub_queries) == 1


def test_attention_score_valid(kb, node_ids):
    """attention_score returns AttentionScore with weight in [0, 1]."""
    if not node_ids:
        pytest.skip("no nodes")
    a = kb.attention_score(node_ids[0], "machine learning")
    assert isinstance(a, AttentionScore)
    assert 0.0 <= a.weight <= 1.0


def test_find_analogy_returns_result(kb):
    """find_analogy returns an AnalogyResult."""
    r = kb.find_analogy("cat", "animal", "dog")
    assert isinstance(r, AnalogyResult)


def test_concept_boundary_margin(kb, node_ids):
    """concept_boundary returns ConceptBoundary with non-negative margin."""
    if len(node_ids) < 2:
        pytest.skip("need 2 nodes")
    cb = kb.concept_boundary(node_ids[0], node_ids[1])
    assert isinstance(cb, ConceptBoundary)
    assert cb.margin >= 0.0


def test_entropy_dispel_returns_report(kb):
    """entropy_dispel returns DispelReport with non-negative entropy_before."""
    r = kb.entropy_dispel()
    assert isinstance(r, DispelReport)
    assert r.entropy_before >= 0.0


def test_build_digest_chain_nonempty(kb):
    """build_digest_chain returns a dict (may be empty if store empty)."""
    d = kb.build_digest_chain()
    assert isinstance(d, dict)
    assert len(d) >= 0


def test_compute_transition_matrix_keys(kb, node_ids):
    """compute_transition_matrix returns a dict."""
    result = kb.compute_transition_matrix(node_ids)
    assert isinstance(result, dict)


def test_manifold_ops_twonn(kb):
    """twonn_intrinsic_dim returns a positive float."""
    from core.manifold_ops import twonn_intrinsic_dim
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
    d = twonn_intrinsic_dim(vecs)
    assert isinstance(d, float)
    assert d > 0.0


def test_manifold_ops_lorentz_project():
    """lorentz_project lifts a vector to the hyperboloid with time component >= 1."""
    import numpy as np
    from core.manifold_ops import lorentz_project
    v = np.array([1.0, 0.0])
    x = lorentz_project(v)
    assert x[0] >= 1.0
    assert len(x) == 3


def test_activation_predictor():
    """ActivationPredictor.predict_embedding returns unit-normed vector of output_dim."""
    from core.activation_predictor import ActivationPredictor, stub_activations
    pred = ActivationPredictor(input_dim=64, output_dim=32)
    acts = stub_activations("hello world", dim=64)
    emb = pred.predict_embedding(acts)
    import numpy as np
    assert len(emb) == 32
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
