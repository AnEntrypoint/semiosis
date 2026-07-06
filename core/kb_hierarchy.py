"""KnowledgeBase mixin: hierarchy compression, tension/flow navigation, and energy descent."""
from __future__ import annotations

from .interfaces import Prefix
from .kb_types import (
    FlowNeighbor, TensionPair, CompressResult, CompressedHierarchy,
    FoldBudgetResult,
)


class HierarchyMixin:
    """Compression, navigation, tension scanning, and energy-gradient ops over the cone tree."""

    def navigate(self, focus_query: str, k: int = 5) -> list[FlowNeighbor]:
        """Neighbor cones ranked by entailment gradient with up/down direction labels."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not focus_query:
            return []
        focus = self._resolve_top(focus_query)
        if focus is None:
            return []
        engine = self._pipeline.engine
        nodes = [n for n in self._pipeline.store.all_nodes()
                 if n.prefix == focus.prefix and n.members]
        out: list[FlowNeighbor] = []
        for nid, weight, direction in engine.flow_neighbors(focus, nodes, k):
            out.append(FlowNeighbor(self._primary_text(self._pipeline.store.get(nid)),
                                    float(weight), direction))
        return out

    def scan_tension(self, top_n: int = 10) -> list[TensionPair]:
        """Surface the worst redundancy/contradiction pairs as human-readable text pairs."""
        if self._pipeline is None:
            return []
        engine = self._pipeline.engine
        store = self._pipeline.store
        out: list[TensionPair] = []
        for a_id, b_id, tension, kind in engine.tension_scan(store.all_nodes(), top_n=top_n):
            out.append(TensionPair(self._primary_text(store.get(a_id)),
                                   self._primary_text(store.get(b_id)), float(tension), kind))
        return out

    def compress_context(self, query: str, k: int) -> CompressResult:
        """Pick k energy-minimizing representatives over a candidate pool; report energy reduction."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._pipeline is None or not query:
            return CompressResult((), 0.0)
        enc = self._pipeline._encoder
        q_vec = enc.encode([query])[0]
        prefix = Prefix(enc.dims[0])
        ids = self._pipeline.query.knn(q_vec[:prefix], k=max(k * 4, k), prefix=prefix)
        pool = [n for n in (self._pipeline.store.get(i) for i in ids) if n.members]
        engine = self._pipeline.engine
        reps, coverage = engine.select_representatives(pool, k)
        base = engine.select_representatives(pool, 1)[1]
        return CompressResult(tuple(self._primary_text(n) for n in reps),
                              float(max(0.0, base - coverage)))

    def compress_hierarchy(self, query: str, max_nodes: int = 10) -> "CompressedHierarchy":
        """Retain highest-relevance nodes under info-bottleneck criterion."""
        import numpy as _np
        nodes = self._pipeline.store.all_nodes() if self._pipeline is not None else []
        if not nodes:
            return CompressedHierarchy((), (), 1.0)
        q_vec = self._pipeline._encoder.encode([query])[0]
        first_centroid = next((n.centroid for n in nodes if n.centroid), None)
        p = len(first_centroid) if first_centroid else min(256, len(q_vec))
        p = min(p, len(q_vec))
        q_slice = _np.array(q_vec[:p], dtype=float)
        q_slice = q_slice / (_np.linalg.norm(q_slice) + 1e-9)
        scored = []
        for node in nodes:
            c = _np.array(list(node.centroid)[:p], dtype=float) if node.centroid else _np.zeros(p)
            cn = _np.linalg.norm(c)
            c = c / (cn + 1e-9)
            scored.append((float(_np.dot(q_slice, c)), node.id))
        scored.sort(reverse=True)
        retained = tuple(nid for _, nid in scored[:max_nodes])
        dropped = tuple(nid for _, nid in scored[max_nodes:])
        ratio = len(retained) / (len(retained) + len(dropped)) if (retained or dropped) else 1.0
        return CompressedHierarchy(retained, dropped, ratio)

    def fold_budget(self, query: str, max_tokens: int, candidates: list) -> "FoldBudgetResult":
        """Greedy token-budget fold: select highest-relevance candidates under token limit."""
        import numpy as _np
        q_vec = _np.array(self._pipeline._encoder.encode([query])[0], dtype=float)
        q_vec = q_vec / (_np.linalg.norm(q_vec) + 1e-9)
        scored = []
        for text in candidates:
            v = _np.array(self._pipeline._encoder.encode([text])[0], dtype=float)
            vn = _np.linalg.norm(v)
            v = v / (vn + 1e-9)
            scored.append((float(_np.dot(q_vec, v)), text))
        scored.sort(reverse=True)
        included, excluded = [], []
        tokens_used = 0
        energy = 0.0
        for score, text in scored:
            tok = max(1, len(text.split()) * 4 // 3)
            if tokens_used + tok <= max_tokens:
                included.append(text)
                tokens_used += tok
                energy += 1.0 - score
            else:
                excluded.append(text)
        return FoldBudgetResult(tuple(included), tuple(excluded), tokens_used, energy)

    def build_digest_chain(self, summarizer=None) -> dict:
        """Bottom-up hierarchy digest chain: leaf->parent using summarizer."""
        store_nodes = self._pipeline.store.nodes_by_id() if self._pipeline is not None else {}
        if not store_nodes:
            return {}
        digests = {}
        for nid, node in store_nodes.items():
            members = list(getattr(node, 'members', []) or [])
            if members:
                digests[nid] = members[0][:100] if members[0] else nid
            else:
                digests[nid] = nid
        for nid, node in store_nodes.items():
            digest_attr = getattr(node, 'digest', None)
            if digest_attr:
                if summarizer is not None:
                    members = list(getattr(node, 'members', []) or [])
                    try:
                        digests[nid] = summarizer.summarize(nid, members)
                    except Exception:
                        digests[nid] = digest_attr
                else:
                    digests[nid] = digest_attr
        return digests

    def compute_transition_matrix(self, node_ids: list) -> dict:
        """P(octave_j | octave_i): fraction of octave_i nodes whose members resolve into an octave_j node."""
        if self._pipeline is None:
            return {}
        enc = self._pipeline._encoder
        store = self._pipeline.store
        octaves = [int(d) for d in enc.dims]
        counts = {(o1, o2): 0.0 for o1 in octaves for o2 in octaves}
        by_octave: dict[int, list] = {o: [] for o in octaves}
        wanted = set(node_ids) if node_ids else None
        for n in store.all_nodes():
            if wanted is not None and str(n.id) not in wanted:
                continue
            if int(n.prefix) in by_octave:
                by_octave[int(n.prefix)].append(n)
        for oi_idx, oi in enumerate(octaves[:-1]):
            oj = octaves[oi_idx + 1]
            member_map = store.members_to_nodes(Prefix(oj))
            for node in by_octave[oi]:
                targets = {member_map[m] for m in node.members if m in member_map}
                if targets:
                    counts[(oi, oi)] += 0.0  # no self-transition credit; real edge found below
                    for _ in targets:
                        counts[(oi, oj)] += 1.0 / len(targets)
                else:
                    counts[(oi, oi)] += 1.0
        if octaves:
            last = octaves[-1]
            counts[(last, last)] += float(len(by_octave[last]))
        matrix = {}
        for o1 in octaves:
            row_sum = sum(counts[(o1, o2)] for o2 in octaves) + 1e-9
            for o2 in octaves:
                matrix[(o1, o2)] = counts[(o1, o2)] / row_sum
        return matrix

    def energy_gradient_search(self, query: str, max_steps: int = 10) -> dict:
        """Greedy energy-descent from best search hit toward lowest-energy leaf node."""
        import numpy as _np
        from .kb_types import EnergyStep
        try:
            from .manifold_ops import lorentz_project as _lorentz_project
        except ImportError:
            _lorentz_project = None
        if self._pipeline is None or not query:
            return {"steps": [], "terminal_node_id": "", "total_energy_drop": 0.0}
        enc = self._pipeline._encoder
        store = self._pipeline.store
        engine = self._pipeline.engine
        q_raw = _np.array(enc.encode([query])[0], dtype=float)
        if _lorentz_project is not None:
            q_vec = _lorentz_project(q_raw)
        else:
            q_vec = q_raw
        hits = self.search(query, k=1)
        if not hits:
            return {"steps": [], "terminal_node_id": "", "total_energy_drop": 0.0}
        store_nodes = store.nodes_by_id()
        current_id = hits[0].node_id
        steps = []
        start_energy: float | None = None
        for _ in range(max_steps):
            node = store_nodes.get(current_id)
            if node is None:
                break
            all_nodes = [n for n in store_nodes.values()]
            e = float(engine.context_energy([node] + all_nodes[:4]))
            if start_energy is None:
                start_energy = e
            octave = int(node.prefix) if node.prefix else 64
            steps.append(EnergyStep(node_id=current_id, energy=e, octave=octave))
            child_ids = [
                nid for nid, n in store_nodes.items()
                if n.members and nid != current_id and int(n.prefix) >= octave
            ]
            if not child_ids:
                break
            best_child_id = current_id
            best_energy = e
            for cid in child_ids[:8]:
                cn = store_nodes.get(cid)
                if cn is None:
                    continue
                ce = float(engine.context_energy([cn]))
                if ce < best_energy:
                    best_energy = ce
                    best_child_id = cid
            if best_child_id == current_id:
                break
            current_id = best_child_id
        terminal_energy = steps[-1].energy if steps else (start_energy or 0.0)
        drop = float((start_energy or 0.0) - terminal_energy)
        return {"steps": steps, "terminal_node_id": current_id, "total_energy_drop": drop}
