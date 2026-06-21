"""Dagster DAG: encode -> cluster -> fit cones -> store, wired to real core components."""
import dataclasses
import uuid

import numpy as np

from .cone_engine import ConeFitConfig, HyperbolicConeEngine
from .encoder import AgglomerativeClusterer, RandomEncoder
from .interfaces import CommitId, Prefix, phrase_to_text_index
from .settings import Settings

try:
    from dagster import Config, Definitions, asset
    _HAS_DAGSTER = True
except ImportError:  # pragma: no cover
    _HAS_DAGSTER = False


def build_encoder(settings: Settings):
    """Real Matryoshka encoder when sentence-transformers is present, else RandomEncoder."""
    try:
        from .encoder import SentenceTransformerEncoder
        return SentenceTransformerEncoder(
            model_name=settings.encoder.model,
            octaves=settings.encoder.octaves,
        )
    except RuntimeError:
        return RandomEncoder(octaves=settings.encoder.octaves)


def fit_octave_nodes(vecs: np.ndarray, texts: list[str], dims, settings: Settings):
    """Cluster and fit cones across every Matryoshka octave; attach retrieval centroids."""
    cone_cfg = ConeFitConfig(
        curvature=settings.cone.curvature,
        dim=settings.cone.dim,
        epochs=settings.cone.epochs,
        lr=settings.cone.lr,
        margin=settings.cone.margin,
        neg_samples=settings.cone.neg_samples,
        seed=settings.cone.seed,
    )
    engine = HyperbolicConeEngine(cone_cfg)
    n = max(1, len(texts))
    all_nodes: list = []
    for prefix in dims:
        clusterer = AgglomerativeClusterer(n_clusters=min(n, 16))
        tree = clusterer.fit(vecs, Prefix(prefix))
        for node in engine.fit(tree):
            idxs = [phrase_to_text_index(m, len(texts)) for m in node.members]
            idxs = [i for i in idxs if i is not None]
            if idxs:
                c = vecs[idxs, : int(prefix)].mean(axis=0)
                node = dataclasses.replace(node, centroid=tuple(float(x) for x in c))
            all_nodes.append(node)
    return all_nodes


if _HAS_DAGSTER:
    class CorpusConfig(Config):
        texts: list[str] = []

    @asset
    def embeddings(config: CorpusConfig):
        """Encode the configured corpus to Matryoshka vectors."""
        settings = Settings()
        encoder = build_encoder(settings)
        texts = list(config.texts)
        vecs = encoder.encode(texts) if texts else np.zeros((0, max(encoder.dims)), dtype=np.float32)
        return {"texts": texts, "vecs": np.asarray(vecs, dtype=np.float32), "dims": list(encoder.dims)}

    @asset
    def cone_nodes(embeddings):
        """Cluster and fit hyperbolic cones across every octave."""
        settings = Settings()
        texts = embeddings["texts"]
        if not texts:
            return []
        return fit_octave_nodes(embeddings["vecs"], texts, embeddings["dims"], settings)

    @asset
    def store_snapshot(cone_nodes):
        """Write fitted cones to the store at a fresh CommitId."""
        from .store import InMemoryStore
        store = InMemoryStore()
        commit_id = CommitId(str(uuid.uuid4()))
        store.write(list(cone_nodes), commit_id)
        return {"commit_id": str(commit_id), "node_count": len(cone_nodes)}

    defs = Definitions(assets=[embeddings, cone_nodes, store_snapshot])
