"""KnowledgeBase mixin: semantic distance, direction, trajectory, and octave geometry."""
from __future__ import annotations

from .interfaces import Prefix
from .kb_types import (
    SemanticDirection, SemanticDirectionError, DirectionSearchResult, SearchHit,
    SemanticTrajectory, TrajectoryStep, ContrastiveDirection, AttentionScore,
    AnalogyResult, ConceptBoundary, ManifoldComplexity,
)


class GeometryMixin:
    """Distance, direction, trajectory, and attention/analogy ops over the Matryoshka octaves."""

    def explain_hierarchy(self, query: str) -> dict:
        """Return cone hierarchy info for the top node matching query."""
        if self._pipeline is None or not query:
            return {}
        node = self._resolve_top(query)
        if node is None:
            return {}
        return {
            "node_id": str(node.id), "aperture": node.aperture,
            "members": [str(m) for m in node.members], "label": node.label,
            "digest": node.digest,
        }

    def containment(self, parent_query: str, child_query: str) -> float:
        """Soft containment score >0 means parent entails child in hyperbolic space."""
        if self._pipeline is None:
            return 0.0
        p, c = self._resolve_top(parent_query), self._resolve_top(child_query)
        if p is None or c is None:
            return 0.0
        return self._pipeline.query.containment_score(p.id, c.id)

    def semantic_distance(self, text_a: str, text_b: str, octave: int | None = None,
                          use_hyperbolic: bool = False) -> "float | dict[int, float]":
        """Cosine or hyperbolic geodesic distance between two texts at one or all octave prefixes."""
        if self._pipeline is None:
            return 0.0 if octave is not None else {}
        enc = self._pipeline._encoder
        va, vb = enc.encode([text_a, text_b])
        import numpy as np
        prefixes = [int(octave)] if octave is not None else [int(d) for d in enc.dims]
        results: dict[int, float] = {}
        for p in prefixes:
            a = va[:p].astype(np.float64)
            b = vb[:p].astype(np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                results[p] = 1.0
                continue
            a, b = a / na, b / nb
            if use_hyperbolic:
                # Lorentz lift: x=(sqrt(2),v); Minkowski <xa,xb>_L = -2+dot(a,b)
                # geodesic = arccosh(-<xa,xb>_L) = arccosh(2-dot); requires arg>=1
                dot_ab = float(np.clip(np.dot(a, b), -1.0, 1.0))
                results[p] = float(np.arccosh(max(2.0 - dot_ab, 1.0)))
            else:
                cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
                results[p] = float(1.0 - cos)
        return results[prefixes[0]] if octave is not None else results

    def compute_direction(self, node_id_a: str, node_id_b: str, octave: int | None = None) -> SemanticDirection:
        """Direction vector from node A centroid to node B centroid in octave subspace."""
        if self._pipeline is None:
            raise SemanticDirectionError("KB not initialized")
        enc = self._pipeline._encoder
        store = self._pipeline.store
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        import numpy as np
        def _centroid(nid: str) -> np.ndarray:
            node = store.get(nid)
            if node.centroid is not None:
                return np.array(node.centroid[:prefix], dtype=np.float64)
            texts = self._member_texts(node)
            if not texts:
                raise SemanticDirectionError(f"node {nid} has no members in this octave")
            vecs = enc.encode(texts)[:, :prefix].astype(np.float64)
            return vecs.mean(axis=0)
        ca = _centroid(node_id_a)
        cb = _centroid(node_id_b)
        diff = cb - ca
        mag = float(np.linalg.norm(diff))
        if mag < 1e-9:
            raise SemanticDirectionError(f"zero direction: nodes {node_id_a} and {node_id_b} are identical in octave {prefix}")
        direction = diff / mag
        na = np.linalg.norm(ca)
        nb = np.linalg.norm(cb)
        cos = float(np.dot(ca / max(na, 1e-9), cb / max(nb, 1e-9)))
        return SemanticDirection(
            from_node=node_id_a, to_node=node_id_b, octave=prefix,
            direction_vec=tuple(float(x) for x in direction),
            magnitude=mag, cosine_alignment=cos,
        )

    def best_octave(self, text_a: str, text_b: str) -> int:
        """Return the octave prefix where distance signal is sharpest (max second derivative)."""
        dists = self.semantic_distance(text_a, text_b)
        if not isinstance(dists, dict) or len(dists) < 3:
            if isinstance(dists, dict):
                return next(iter(dists)) if dists else 64
            return 64
        import math
        prefixes = sorted(dists.keys())
        log_p = [math.log(p) for p in prefixes]
        d = [dists[p] for p in prefixes]
        best_p, best_dd = prefixes[1], -1.0
        for i in range(1, len(prefixes) - 1):
            dd = abs((d[i+1] - d[i]) / max(log_p[i+1] - log_p[i], 1e-9)
                     - (d[i] - d[i-1]) / max(log_p[i] - log_p[i-1], 1e-9))
            if dd > best_dd:
                best_dd, best_p = dd, prefixes[i]
        return best_p

    def direction_search(self, anchor_text: str, direction_vec: "tuple[float, ...] | list[float]",
                         k: int = 5, octave: int | None = None) -> list[DirectionSearchResult]:
        """Find concepts in a given semantic direction from anchor; returns hits per alpha step."""
        if self._pipeline is None or not anchor_text:
            return []
        import numpy as np
        enc = self._pipeline._encoder
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        av = enc.encode([anchor_text])[0][:prefix].astype(np.float64)
        dv = np.array(direction_vec[:prefix], dtype=np.float64)
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-9:
            return []
        dv = dv / dv_norm
        results = []
        for alpha in (0.1, 0.5, 1.0, 2.0):
            probe = av + alpha * dv
            pn = np.linalg.norm(probe)
            if pn > 1e-9:
                probe = probe / pn
            probe32 = probe.astype(np.float32)
            store = self._pipeline.store
            ids = store.knn(probe32, k, Prefix(prefix))
            hits = []
            for nid in ids:
                node = store.get(nid)
                texts = self._member_texts(node)
                if not texts:
                    continue
                nv = enc.encode([texts[0]])[0][:prefix].astype(np.float64)
                nn = np.linalg.norm(nv)
                alignment = float(np.dot(nv / max(nn, 1e-9), dv)) if nn > 1e-9 else 0.0
                hits.append(SearchHit(
                    text=texts[0], score=alignment, node_id=str(node.id),
                    octave=prefix, members=tuple(texts),
                    aperture=float(getattr(node, 'aperture', 0.0)),
                    local_entropy=0.0, evidence_path_count=1,
                    uncertainty_score=float(max(0.0, 1.0 - alignment)),
                ))
            if hits:
                avg_align = sum(h.score for h in hits) / len(hits)
                results.append(DirectionSearchResult(hits=tuple(hits), alpha=alpha, alignment=avg_align))
        return results

    def fold_directions(self, node_id: str, octave: int | None = None) -> list[dict]:
        """Direction vectors from a parent node to each child node; maps 'downward intuition'."""
        if self._pipeline is None:
            return []
        store = self._pipeline.store
        parent = store.get(node_id)
        enc = self._pipeline._encoder
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        import numpy as np
        def _centroid_vec(node) -> "np.ndarray | None":
            if node.centroid is not None:
                return np.array(node.centroid[:prefix], dtype=np.float64)
            texts = self._member_texts(node)
            if not texts:
                return None
            return enc.encode(texts)[:, :prefix].astype(np.float64).mean(axis=0)
        pc = _centroid_vec(parent)
        if pc is None:
            return []
        all_nodes = [n for n in store.all_nodes() if n.prefix == Prefix(prefix) and n.id != parent.id and n.members]
        results = []
        for child in all_nodes:
            cc = _centroid_vec(child)
            if cc is None:
                continue
            diff = cc - pc
            mag = float(np.linalg.norm(diff))
            if mag < 1e-9:
                continue
            direction = diff / mag
            label = self._summarizer.summarize(str(child.id), self._member_texts(child))
            results.append({
                "child_id": str(child.id),
                "direction_vec": tuple(float(x) for x in direction),
                "magnitude": mag,
                "semantic_label": label,
            })
        results.sort(key=lambda r: -r["magnitude"])
        return results

    def compute_trajectory(self, query: str, answer_node_id: str,
                           octave: int | None = None) -> SemanticTrajectory:
        """Trace the octave-descent path from query to answer_node as a SemanticTrajectory."""
        if self._pipeline is None:
            return SemanticTrajectory(steps=(), total_distance=0.0, coherence_score=0.0, energy_cost=0.0)
        import numpy as np
        enc = self._pipeline._encoder
        store = self._pipeline.store
        prefix = int(octave) if octave is not None else int(enc.dims[0])
        q_vec = enc.encode([query])[0][:prefix].astype(np.float64)
        try:
            answer = store.get(answer_node_id)
        except KeyError:
            return SemanticTrajectory(steps=(), total_distance=0.0, coherence_score=0.0, energy_cost=0.0)
        # build path: query -> intermediate nodes at each octave -> answer
        steps: list[TrajectoryStep] = []
        prev_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
        all_dims = [int(d) for d in enc.dims if int(d) <= prefix]
        for dim in all_dims:
            ids = store.knn(q_vec[:dim].astype(np.float32), k=1, prefix=Prefix(dim))
            if not ids:
                continue
            node = store.get(ids[0])
            texts = self._member_texts(node)
            if not texts:
                continue
            nv = enc.encode([texts[0]])[0][:dim].astype(np.float64)
            nn = float(np.linalg.norm(nv))
            nv_n = nv / max(nn, 1e-9)
            cos = float(np.clip(np.dot(prev_vec[:dim], nv_n), -1.0, 1.0))
            dist = float(1.0 - cos)
            diff = nv_n - prev_vec[:dim]
            dm = float(np.linalg.norm(diff))
            direction = tuple(float(x) for x in (diff / dm if dm > 1e-9 else diff))
            steps.append(TrajectoryStep(
                node_id=str(node.id), octave=dim,
                distance_from_prev=dist, direction_vec=direction,
            ))
            prev_vec = nv_n
        total = sum(s.distance_from_prev for s in steps)
        import math
        if len(steps) > 1:
            dists = [s.distance_from_prev for s in steps]
            mean_d = total / len(dists)
            var = sum((d - mean_d) ** 2 for d in dists) / len(dists)
            entropy = math.sqrt(var) / max(mean_d, 1e-9)
            coherence = max(0.0, 1.0 - entropy / math.log(len(steps) + 1))
        else:
            coherence = 1.0
        energy = sum(s.distance_from_prev * (2 ** i) for i, s in enumerate(steps))
        return SemanticTrajectory(
            steps=tuple(steps), total_distance=total,
            coherence_score=coherence, energy_cost=energy,
        )

    def best_octave_trajectory(self, text_a: str, text_b: str) -> dict:
        """Semantic drift across all Matryoshka octaves; best = sharpest signal per best_octave()."""
        import numpy as _np
        if self._pipeline is None:
            return {"octaves": [], "distances": [], "best_octave": 64}
        enc = self._pipeline._encoder
        octaves = [int(d) for d in enc.dims]
        va_full, vb_full = _np.array(enc.encode([text_a, text_b]), dtype=float)
        distances = []
        for oct in octaves:
            va, vb = va_full[:oct], vb_full[:oct]
            na, nb = _np.linalg.norm(va) + 1e-9, _np.linalg.norm(vb) + 1e-9
            cos = float(_np.dot(va / na, vb / nb))
            distances.append(1.0 - cos)
        # single best-octave definition project-wide: max second derivative (see best_octave)
        return {"octaves": octaves, "distances": distances,
                "best_octave": self.best_octave(text_a, text_b)}

    def multi_octave_direction(self, node_a: str, node_b: str) -> list:
        """Direction vectors from node_a to node_b at each Matryoshka octave."""
        import numpy as _np
        if self._pipeline is None:
            return []
        enc = self._pipeline._encoder
        octaves = [int(d) for d in enc.dims]
        results = []
        va_full, vb_full = _np.array(enc.encode([node_a, node_b]), dtype=float)
        for oct in octaves:
            va = va_full[:oct]; vb = vb_full[:oct]
            diff = vb - va
            mag = float(_np.linalg.norm(diff))
            if mag > 1e-9:
                unit = (diff / mag).tolist()
            else:
                unit = [0.0] * min(oct, len(va_full))
            na = _np.linalg.norm(va) + 1e-9; nb = _np.linalg.norm(vb) + 1e-9
            cos = float(_np.dot(va / na, vb / nb))
            results.append(SemanticDirection(
                from_node=node_a, to_node=node_b, octave=oct,
                direction_vec=tuple(unit[:8]), magnitude=mag, cosine_alignment=cos,
            ))
        return results

    def contrastive_direction(self, text_a: str, text_b: str, octave: int | None = None) -> "ContrastiveDirection":
        """Direction = normalize(embed(a) - embed(b)); contrast_score = norm of difference."""
        import numpy as _np
        if self._pipeline is None:
            return ContrastiveDirection((), 0.0, octave or 0)
        va, vb = _np.array(self._pipeline._encoder.encode([text_a, text_b]), dtype=float)
        if octave is not None:
            va, vb = va[:octave], vb[:octave]
        diff = va - vb
        score = float(_np.linalg.norm(diff))
        direction = diff / (score + 1e-9)
        return ContrastiveDirection(tuple(direction.tolist()), score, octave or len(va))

    def attention_score(self, node_id: str, query: str, temperature: float = 1.0) -> "AttentionScore":
        """NLA-style scaled dot-product attention weight over all nodes at the finest octave."""
        import numpy as _np
        import math as _math
        if self._pipeline is None:
            return AttentionScore(node_id, 0.0, 64, temperature)
        enc = self._pipeline._encoder
        p = int(enc.dims[0])
        store_nodes = {n.id: n for n in self._pipeline.store.all_nodes()}
        if not store_nodes:
            return AttentionScore(node_id, 0.0, p, temperature)
        q_vec = _np.array(enc.encode([query])[0], dtype=float)
        q_slice = q_vec[:p] / (_np.linalg.norm(q_vec[:p]) + 1e-9)
        raw_scores = {}
        for nid, node in store_nodes.items():
            c_list = list(node.centroid or [])[:p]
            if not c_list:
                continue
            c = _np.array(c_list, dtype=float)
            if len(c) < p:
                c = _np.pad(c, (0, p - len(c)))
            cn = _np.linalg.norm(c)
            c = c / (cn + 1e-9)
            raw_scores[nid] = float(_np.dot(q_slice, c)) / (_math.sqrt(p) * temperature)
        max_s = max(raw_scores.values(), default=0.0)
        exp_scores = {nid: _math.exp(s - max_s) for nid, s in raw_scores.items()}
        total = sum(exp_scores.values()) + 1e-9
        weight = exp_scores.get(node_id, 0.0) / total
        return AttentionScore(node_id, weight, p, temperature)

    def find_analogy(self, text_a: str, text_b: str, text_c: str, k: int = 5) -> "AnalogyResult":
        """word2vec analogy: embed(c) + (embed(b) - embed(a)) -> nearest nodes."""
        import numpy as _np
        if self._pipeline is None:
            return AnalogyResult((), (), 0.0)
        va, vb, vc = _np.array(self._pipeline._encoder.encode([text_a, text_b, text_c]), dtype=float)
        direction = vb - va
        hits = self.direction_search(text_c, direction.tolist(), k=k)
        analogy_score = float(hits[0].alignment) if hits else 0.0
        raw_hits = []
        for r in hits:
            raw_hits.extend(list(r.hits))
        return AnalogyResult(tuple(raw_hits[:k]), tuple(direction.tolist()), analogy_score)

    def concept_boundary(self, node_id_a: str, node_id_b: str, octave: int | None = None) -> "ConceptBoundary":
        """Compute decision boundary between two concept nodes."""
        import numpy as _np
        store_nodes = self._pipeline.store.nodes_by_id() if self._pipeline is not None else {}
        node_a = store_nodes.get(node_id_a)
        node_b = store_nodes.get(node_id_b)
        if octave is not None:
            p = octave
        else:
            # auto-select the common octave: the smaller of the two centroid dims,
            # so neither side gets silently broadcast-mismatched.
            lens = [len(n.centroid) for n in (node_a, node_b) if n is not None and n.centroid]
            p = min(lens) if lens else 256
        def get_centroid(node):
            if node is None or not node.centroid:
                return _np.zeros(p)
            c = _np.array(list(node.centroid)[:p], dtype=float)
            if len(c) < p:
                c = _np.pad(c, (0, p - len(c)))
            return c / (_np.linalg.norm(c) + 1e-9)
        ca = get_centroid(node_a)
        cb = get_centroid(node_b)
        midpoint = (ca + cb) / 2.0
        normal = cb - ca
        margin = float(_np.linalg.norm(normal)) / 2.0
        nn = _np.linalg.norm(normal)
        normal_unit = normal / (nn + 1e-9)
        return ConceptBoundary(tuple(midpoint.tolist()), tuple(normal_unit.tolist()), margin, p)

    def information_content(self, node_id: str) -> float:
        """IC = -log2(aperture/pi); high IC = specific concept."""
        import math as _math
        store_nodes = self._pipeline.store.nodes_by_id() if self._pipeline is not None else {}
        node = store_nodes.get(node_id)
        if node is None:
            return 0.0
        aperture = getattr(node, 'aperture', 0.5) or 0.5
        return max(0.0, -_math.log2(max(aperture, 1e-9) / _math.pi))

    def sense_complexity(self, query: str, k: int = 10) -> "ManifoldComplexity":
        """Estimate intrinsic dim of query neighborhood via TwoNN."""
        from .manifold_ops import twonn_intrinsic_dim
        hits = self.search(query, k=k)
        if len(hits) < 3:
            return ManifoldComplexity(1.0, 64, "constant")
        store_nodes = self._pipeline.store.nodes_by_id() if self._pipeline is not None else {}
        vecs = []
        for h in hits[:k]:
            if h.node_id in store_nodes:
                c = list(store_nodes[h.node_id].centroid or [])
                if c:
                    vecs.append(c)
        if len(vecs) < 3:
            return ManifoldComplexity(1.0, 64, "constant")
        min_len = min(len(v) for v in vecs)
        vecs = [v[:min_len] for v in vecs]
        dim = twonn_intrinsic_dim(vecs)
        if dim < 1.5:
            label, octave = "constant", 64
        elif dim < 2.5:
            label, octave = "linear", 128
        elif dim < 4.0:
            label, octave = "quadratic", 256
        else:
            label, octave = "exponential", 512
        return ManifoldComplexity(dim, octave, label)
