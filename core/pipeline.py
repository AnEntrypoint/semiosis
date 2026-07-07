"""End-to-end knowledge pipeline: encode -> recursive-cluster -> fit cones -> store; leaf-routed incremental ingest."""
from __future__ import annotations

import dataclasses
import uuid
from typing import Sequence

import numpy as np

from .cone_engine import ConeFitConfig, HyperbolicConeEngine
from .encoder import AgglomerativeClusterer, RandomEncoder
from .interfaces import CommitId, ConeNode, NodeId, PhraseId, Prefix, phrase_to_text_index
from .settings import Settings
from .store import InMemoryQuery, InMemoryStore


class KnowledgePipeline:
    """Encode texts -> recursive cluster tree per octave -> fit cones -> store; exposes a Query."""

    def __init__(
        self,
        texts: Sequence[str],
        settings: Settings | None = None,
        prebuilt_nodes: "Sequence[ConeNode] | None" = None,
    ) -> None:
        cfg = settings or Settings()
        self._settings = cfg
        self._texts: list[str] = list(texts)
        self.encoder_fallback_reason: str | None = None
        try:
            from .encoder import SentenceTransformerEncoder
            self._encoder = SentenceTransformerEncoder(
                model_name=cfg.encoder.model,
                octaves=cfg.encoder.octaves,
                device=cfg.encoder.device,
                fp16=cfg.encoder.fp16,
            )
        except RuntimeError as e:
            # deliberate fallback (e.g. sentence-transformers not installed, or model fetch failed)
            self.encoder_fallback_reason = str(e)
            self._encoder = RandomEncoder(octaves=cfg.encoder.octaves)

        self._clusterer = AgglomerativeClusterer(
            branching_factor=cfg.cluster.branching_factor,
            max_leaf_size=cfg.cluster.max_leaf_size,
            max_depth=cfg.cluster.max_depth,
        )
        self._engine = HyperbolicConeEngine(ConeFitConfig.from_settings(cfg.cone))
        self._store = InMemoryStore(
            n_partitions=cfg.store.hilbert_partitions,
            catapult_size=cfg.store.catapult_cache_size,
        )
        self._query = InMemoryQuery(self._store, self._engine)
        self._vec_cache: dict[str, np.ndarray] = {}
        self._commit: CommitId | None = None
        self.rebuild_count = 0

        if prebuilt_nodes is not None:
            self._commit = self._store.write(list(prebuilt_nodes), CommitId(str(uuid.uuid4())))
        elif texts:
            self._rebuild()

    def _encode_cached(self, texts: list[str]) -> np.ndarray:
        """Encode only texts absent from the cache; assemble the full matrix in order."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec_cache]
        if missing:
            vecs = self._encoder.encode(missing)
            for t, v in zip(missing, vecs):
                self._vec_cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._vec_cache[t] for t in texts])

    def _rebuild(self) -> CommitId:
        """Rebuild all octave trees over the full corpus, reusing cached embeddings."""
        self.rebuild_count += 1
        vecs = self._encode_cached(self._texts)
        self._store = InMemoryStore(
            n_partitions=self._settings.store.hilbert_partitions,
            catapult_size=self._settings.store.catapult_cache_size,
        )
        self._query = InMemoryQuery(self._store, self._engine)
        commit_id = CommitId(str(uuid.uuid4()))
        all_nodes: list = []
        for prefix in self._encoder.dims:
            tree = self._clusterer.fit(vecs, Prefix(prefix))
            fitted = self._engine.fit_and_close(tree)
            all_nodes.extend(self._finalize_octave(fitted, tree, vecs, Prefix(prefix)))
        self._store.write(all_nodes, commit_id)
        self._commit = commit_id
        return commit_id

    def _finalize_octave(self, fitted, tree, vecs: np.ndarray, prefix: Prefix) -> list:
        """Attach parent edges, transitive parent membership, centroids, and digests."""
        parent_of = {c: p for p, c in tree.edges}
        by_id = {n.id: n for n in fitted}
        # bottom-up transitive member closure: deepest path first
        members: dict = {nid: list(n.members) for nid, n in by_id.items()}
        for nid in sorted(by_id, key=lambda i: -str(i).count(".")):
            p = parent_of.get(nid)
            if p in members:
                seen = set(members[p])
                members[p].extend(m for m in members[nid] if m not in seen)
        out = []
        for nid, node in by_id.items():
            node = dataclasses.replace(
                node, members=tuple(members[nid]), parent=parent_of.get(nid))
            out.append(self._centroid(self._digest(node), vecs, prefix))
        return out

    def _centroid(self, node, vecs: np.ndarray, prefix: Prefix):
        """Attach the embedding-space mean of a node's members (prefix-sliced) for retrieval."""
        idxs = [phrase_to_text_index(m, len(self._texts)) for m in node.members]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return node
        c = vecs[idxs, :int(prefix)].mean(axis=0)
        return dataclasses.replace(node, centroid=tuple(float(x) for x in c))

    def _digest(self, node):
        """Backfill a lightweight digest for multi-member cones; lazy member-text fallback."""
        mem = self._settings.memory
        if len(node.members) <= mem.digest_min_members:
            return node
        parts: list[str] = []
        for m in sorted(node.members):
            idx = phrase_to_text_index(m, len(self._texts))
            if idx is not None and self._texts[idx] not in parts:
                parts.append(self._texts[idx])
            if len(parts) >= 1:
                break
        head = parts[0] if parts else ""
        extra = len(node.members) - 1
        text = f"{head} (+{extra} more)" if extra > 0 else head
        return dataclasses.replace(node, digest=text[:mem.summary_max_chars])

    @property
    def texts(self) -> list[str]:
        return list(self._texts)

    @property
    def commit(self) -> "CommitId | None":
        return self._commit

    def ingest(self, texts: Sequence[str]) -> CommitId:
        """Route new texts to nearest leaves (local splits on overflow); full rebuild only past tension threshold."""
        texts = list(texts)
        if not texts:
            return self._commit or CommitId(str(uuid.uuid4()))
        start = len(self._texts)
        self._texts.extend(texts)
        if not self._store.all_nodes() or not self._settings.agent.incremental_ingest:
            return self._rebuild()
        self._encode_cached(texts)
        for j, t in enumerate(texts):
            vec = self._vec_cache[t]
            pid = PhraseId(f"doc_{start + j}")
            for prefix in self._encoder.dims:
                self._route(pid, vec, Prefix(prefix))
        if self._tension_high():
            return self._rebuild()
        self._commit = CommitId(str(uuid.uuid4()))
        return self._commit

    def _route(self, pid: PhraseId, vec: np.ndarray, prefix: Prefix) -> None:
        """Assign one new phrase to its nearest leaf; update centroids up the ancestor chain."""
        leaves = self._store.leaves_at(prefix)
        if not leaves:
            return
        v = np.asarray(vec[:int(prefix)], dtype=np.float64)
        vn = np.linalg.norm(v) or 1.0

        def sim(node) -> float:
            c = np.asarray(node.centroid, dtype=np.float64) if node.centroid else None
            if c is None or not len(c):
                return -1e18
            return float(np.dot(c, v[:len(c)]) / ((np.linalg.norm(c) or 1.0) * vn))

        best = max(leaves, key=sim)
        nid: "NodeId | None" = best.id
        while nid is not None:
            node = self._store.get(nid)
            new_members = (*node.members, pid)
            k = len(new_members)
            if node.centroid:
                c = np.asarray(node.centroid, dtype=np.float64)
                c = c + (v[:len(c)] - c) / k
                centroid = tuple(float(x) for x in c)
            else:
                centroid = tuple(float(x) for x in v)
            node = dataclasses.replace(node, members=new_members, centroid=centroid)
            self._store.upsert(node)
            nid = node.parent
        leaf = self._store.get(best.id)
        if len(leaf.members) > self._settings.cluster.max_leaf_size:
            self._split_leaf(leaf, prefix)

    def _split_leaf(self, leaf, prefix: Prefix) -> None:
        """Local Ward-2 split of an oversize leaf; children inherit cone geometry until next rebuild."""
        idx_pairs = [(m, phrase_to_text_index(m, len(self._texts))) for m in leaf.members]
        idx_pairs = [(m, i) for m, i in idx_pairs if i is not None]
        if len(idx_pairs) < 4:
            return
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
        except ImportError:
            return
        X = np.stack([self._vec_cache[self._texts[i]][:int(prefix)] for _, i in idx_pairs]).astype(np.float64)
        labels = fcluster(linkage(X, method="ward"), 2, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            return
        base = str(leaf.id).split("@")[0]
        for c in np.unique(labels):
            sub = [m for (m, _), lb in zip(idx_pairs, labels) if lb == c]
            sub_x = X[labels == c]
            child = ConeNode(
                id=NodeId(f"{base}.s{int(c)}@{int(prefix)}"),
                apex=leaf.apex, aperture=leaf.aperture, prefix=prefix,
                members=tuple(sub),
                centroid=tuple(float(x) for x in sub_x.mean(axis=0)),
                parent=leaf.id,
            )
            self._store.upsert(child)

    def _tension_high(self) -> bool:
        """Sampled mean tension over finest-octave leaves against the rebalance threshold."""
        finest = Prefix(int(self._encoder.dims[-1]))
        leaves = self._store.leaves_at(finest)[:16]
        if len(leaves) < 2:
            return False
        total, pairs = 0.0, 0
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                total += max(0.0, self._engine.tension(leaves[i], leaves[j]))
                pairs += 1
        return (total / pairs) > self._settings.cluster.rebalance_tension if pairs else False

    @property
    def query(self) -> InMemoryQuery:
        return self._query

    @property
    def store(self) -> InMemoryStore:
        return self._store

    @property
    def engine(self) -> HyperbolicConeEngine:
        return self._engine
