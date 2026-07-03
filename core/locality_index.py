"""Locality-preserving indexing over embedding centroids -- Hilbert-curve bucketing + SimHash prefilter + query-locality catapult cache.

Ideas adapted from: ELID (docs.rs/elid, SimHash/Hilbert encoding of vectors),
VStream (VLDB'25, dynamic space-filling-curve partitioning templates),
CatapultDB (arxiv 2603.02164, LSH-bucketed query-locality shortcut cache).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .interfaces import EuclideanVec, NodeId, Prefix

_HILBERT_ORDER = 10          # bits per axis; matches ELID's Hilbert10x10 granularity
_HILBERT_SIDE = 1 << _HILBERT_ORDER
_SIMHASH_BITS = 64


def _hilbert_d2xy_free(order: int, x: int, y: int) -> int:
    """Convert 2D grid coords to 1D Hilbert distance; standard bit-rotation algorithm."""
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_key(vec: np.ndarray, seed: int = 0) -> int:
    """Project a vector to 2D via two fixed random hyperplanes, then encode as a Hilbert index."""
    rng = np.random.default_rng(seed)
    dim = len(vec)
    proj = rng.standard_normal((2, dim)).astype(np.float64)
    xy = proj @ np.asarray(vec, dtype=np.float64)
    # map to [0, side) via a stable sigmoid-style squash -- keeps locality without unbounded range
    x = int((1.0 / (1.0 + np.exp(-xy[0]))) * (_HILBERT_SIDE - 1))
    y = int((1.0 / (1.0 + np.exp(-xy[1]))) * (_HILBERT_SIDE - 1))
    return _hilbert_d2xy_free(_HILBERT_ORDER, x, y)


def simhash(vec: np.ndarray, bits: int = _SIMHASH_BITS, seed: int = 1) -> int:
    """Signed random-projection fingerprint (ELID Mini128-style); Hamming-comparable."""
    rng = np.random.default_rng(seed)
    dim = len(vec)
    planes = rng.standard_normal((bits, dim)).astype(np.float64)
    signs = planes @ np.asarray(vec, dtype=np.float64) >= 0
    h = 0
    for b in signs:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class HilbertBucketIndex:
    """Per-octave Hilbert-bucketed candidate index; VStream-style dynamic partitioning templates.

    Falls back to full scan when bucket count is too small to prune meaningfully.
    """

    def __init__(self, n_partitions: int = 16) -> None:
        self._n_partitions = max(1, n_partitions)
        self._by_prefix: dict[Prefix, dict[NodeId, int]] = {}   # nid -> hilbert key
        self._templates: dict[Prefix, list[int]] = {}           # sorted key boundaries (dynamic partitioning template)
        self._simhash: dict[Prefix, dict[NodeId, int]] = {}

    def upsert(self, nid: NodeId, prefix: Prefix, vec: np.ndarray) -> None:
        """Insert/update one node's locality keys; invalidates cached partition template for this octave."""
        keys = self._by_prefix.setdefault(prefix, {})
        keys[nid] = hilbert_key(vec)
        sh = self._simhash.setdefault(prefix, {})
        sh[nid] = simhash(vec)
        self._templates.pop(prefix, None)

    def remove(self, nid: NodeId, prefix: Prefix) -> None:
        self._by_prefix.get(prefix, {}).pop(nid, None)
        self._simhash.get(prefix, {}).pop(nid, None)
        self._templates.pop(prefix, None)

    def _template(self, prefix: Prefix) -> list[int]:
        """Rebuild partition boundaries from current key distribution (VStream Sec 3.2)."""
        if prefix in self._templates:
            return self._templates[prefix]
        keys = sorted(self._by_prefix.get(prefix, {}).values())
        if not keys:
            self._templates[prefix] = []
            return []
        p = min(self._n_partitions, max(1, len(keys)))
        step = max(1, len(keys) // p)
        boundaries = [keys[i] for i in range(0, len(keys), step)][1:]
        self._templates[prefix] = boundaries
        return boundaries

    def candidates(self, prefix: Prefix, q_vec: np.ndarray, simhash_prefilter: bool = True) -> set[NodeId] | None:
        """Return a pruned candidate set for this octave, or None if bucketing isn't warranted yet."""
        keys = self._by_prefix.get(prefix)
        if not keys or len(keys) < self._n_partitions * 2:
            return None  # too few nodes to prune; caller should fall back to full scan
        boundaries = self._template(prefix)
        qk = hilbert_key(q_vec)
        import bisect
        bucket = bisect.bisect_left(boundaries, qk)
        # include the query's bucket plus its two neighbors -- boundary vectors can land either side
        target_buckets = {bucket - 1, bucket, bucket + 1}
        sorted_items = sorted(keys.items(), key=lambda kv: kv[1])
        n_buckets = len(boundaries) + 1
        out: set[NodeId] = set()
        idx = 0
        cur_bucket = 0
        for nid, k in sorted_items:
            while cur_bucket < len(boundaries) and k >= boundaries[cur_bucket]:
                cur_bucket += 1
            if cur_bucket in target_buckets:
                out.add(nid)
        if not out:
            return None
        if simhash_prefilter and prefix in self._simhash:
            qsh = simhash(q_vec)
            sh_map = self._simhash[prefix]
            budget = max(len(out) // 2, 8)
            ranked = sorted(out, key=lambda n: hamming(sh_map.get(n, 0), qsh))
            out = set(ranked[:budget]) if len(ranked) > budget else out
        return out


class CatapultCache:
    """LRU-bounded query-locality shortcut cache: bucket -> best-known entry-point node (CatapultDB)."""

    def __init__(self, max_size: int = 512) -> None:
        self._max_size = max(1, max_size)
        self._cache: OrderedDict[tuple[Prefix, int], NodeId] = OrderedDict()

    def get(self, prefix: Prefix, q_vec: np.ndarray) -> NodeId | None:
        key = (prefix, hilbert_key(q_vec))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def record(self, prefix: Prefix, q_vec: np.ndarray, best_nid: NodeId) -> None:
        key = (prefix, hilbert_key(q_vec))
        self._cache[key] = best_nid
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)
