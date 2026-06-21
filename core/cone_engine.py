"""Hyperbolic entailment-cone engine -- fits Lorentz cones via RiemannianAdam."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .settings import ConeSettings

import numpy as np

from .interfaces import ConeNode, ClusterTree, NodeId

# These imports are deferred so the module is importable without torch present
# (e.g. for type-checking or docs builds); the engine validates them on init.
try:  # pragma: no cover - exercised in integration env
    import torch
    import geoopt
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


_EPS = 1e-7              # arccos / sqrt guard
_MIN_APERTURE = 0.1      # radians; cones never collapse to a ray
_MAX_GRAD_NORM = 1.0     # tangent-space gradient clip


@dataclass(frozen=True, slots=True)
class ConeFitConfig:
    curvature: float = 1.0
    dim: int = 8                 # cone-space dim (small; the octave prefix is separate)
    epochs: int = 200
    lr: float = 1e-2
    margin: float = 0.01
    neg_samples: int = 5
    seed: int = 0

    @classmethod
    def from_settings(cls, s: "ConeSettings") -> "ConeFitConfig":
        """Build from env-configurable ConeSettings; single source of truth for hyperparams."""
        return cls(
            curvature=s.curvature,
            dim=s.dim,
            epochs=s.epochs,
            lr=s.lr,
            margin=s.margin,
            neg_samples=s.neg_samples,
            seed=s.seed,
        )


class HyperbolicConeEngine:
    """Fits Ganea 2018-style entailment cones on the Lorentz manifold (ConeEmbedder protocol)."""

    def __init__(self, cfg: ConeFitConfig) -> None:
        if not _HAS_TORCH:
            raise RuntimeError(
                "HyperbolicConeEngine requires torch + geoopt. "
                "Install the 'hyperbolic' extra."
            )
        self.cfg = cfg
        self.manifold = geoopt.Lorentz(k=torch.tensor(cfg.curvature))

    # --- numerically guarded primitives -------------------------------------
    @staticmethod
    def _safe_arccos(x: "torch.Tensor") -> "torch.Tensor":
        return torch.arccos(torch.clamp(x, -1.0 + _EPS, 1.0 - _EPS))

    def _half_aperture(self, apex: "torch.Tensor") -> "torch.Tensor":
        """Ganea 2018 closed-form aperture as a function of apex norm, floored."""
        # spatial component norm on the hyperboloid
        norm = torch.clamp(apex[..., 1:].norm(dim=-1), min=_EPS)
        psi = torch.arcsin(torch.clamp((1.0) / norm, max=1.0 - _EPS))
        return torch.clamp(psi, min=_MIN_APERTURE)

    def _cone_energy(self, parent: "torch.Tensor", child: "torch.Tensor") -> "torch.Tensor":
        """Penalty when `child` lies outside `parent`'s entailment cone (>=0)."""
        xi = self._angle_at(parent, child)
        psi = self._half_aperture(parent)
        return torch.clamp(xi - psi, min=0.0)

    def _angle_at(self, apex: "torch.Tensor", other: "torch.Tensor") -> "torch.Tensor":
        """Angle between the cone axis at `apex` and the geodesic to `other`."""
        # Lorentzian inner product <a,b>_L = -a0 b0 + sum a_i b_i
        ip = -apex[..., 0] * other[..., 0] + (apex[..., 1:] * other[..., 1:]).sum(-1)
        # On the hyperboloid, ip <= -1 always; clamp ensures ip*ip - 1 >= _EPS^2 > 0.
        ip = torch.clamp(ip, max=-(1.0 + _EPS))
        num = other[..., 0] + ip * apex[..., 0]
        den = torch.clamp(
            apex[..., 1:].norm(dim=-1) * torch.sqrt(ip * ip - 1.0),
            min=_EPS,
        )
        return self._safe_arccos(torch.clamp(num / den, -1.0 + _EPS, 1.0 - _EPS))

    # --- fit ----------------------------------------------------------------
    def fit(self, tree: ClusterTree) -> Sequence[ConeNode]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)  # reproducible regardless of how many fit() calls precede this
        ids = sorted({n for e in tree.edges for n in e} | set(tree.assignments.values()))
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)

        # Random init on the manifold, wrapped as Riemannian parameters
        apex = geoopt.ManifoldParameter(
            self.manifold.random_normal((n, cfg.dim + 1), std=0.1),
            manifold=self.manifold,
        )
        opt = geoopt.optim.RiemannianAdam([apex], lr=cfg.lr, stabilize=10)

        pos = torch.tensor([[idx[p], idx[c]] for p, c in tree.edges], dtype=torch.long)
        rng = np.random.default_rng(cfg.seed)

        if pos.shape[0] > 0:  # skip training when no positive edges
            for _ in range(cfg.epochs):
                opt.zero_grad()
                p_apex, c_apex = apex[pos[:, 0]], apex[pos[:, 1]]
                loss_pos = self._cone_energy(p_apex, c_apex).mean()
                neg_raw = rng.integers(0, n, size=(pos.shape[0] * cfg.neg_samples, 2))
                # remove self-loops: a point vs itself has cone_energy=0, biasing the loss
                mask = neg_raw[:, 0] != neg_raw[:, 1]
                neg_raw = neg_raw[mask] if mask.any() else neg_raw
                neg = torch.tensor(neg_raw, dtype=torch.long)
                np_apex, nc_apex = apex[neg[:, 0]], apex[neg[:, 1]]
                loss_neg = torch.clamp(
                    cfg.margin - self._cone_energy(np_apex, nc_apex), min=0.0
                ).mean()
                loss = loss_pos + loss_neg
                loss.backward()
                torch.nn.utils.clip_grad_norm_([apex], _MAX_GRAD_NORM)
                opt.step()

        apex_np = apex.detach().cpu().numpy().astype(np.float64)
        psi_np = self._half_aperture(apex.detach()).cpu().numpy()

        members: dict[str, list[str]] = {nid: [] for nid in ids}
        for pid, nid in tree.assignments.items():
            members[nid].append(pid)

        return tuple(
            ConeNode(
                id=NodeId(nid),
                apex=apex_np[idx[nid]],
                aperture=float(psi_np[idx[nid]]),
                prefix=tree.prefix,
                members=tuple(members[nid]),  # type: ignore[arg-type]
            )
            for nid in ids
        )

    def contains(self, parent: ConeNode, child: ConeNode) -> float:
        """Soft containment margin in [-pi, +psi]; >0 => parent entails child."""
        p = torch.from_numpy(parent.apex).float()
        c = torch.from_numpy(child.apex).float()
        xi = self._angle_at(p.unsqueeze(0), c.unsqueeze(0))
        return float(parent.aperture - xi.item())

    def batch_contains(
        self,
        parents: "Sequence[ConeNode]",
        children: "Sequence[ConeNode]",
    ) -> "np.ndarray":
        """Return [N, M] float32 containment margins for N parents x M children."""
        pa = torch.tensor(np.stack([n.apex for n in parents]), dtype=torch.float32)
        ca = torch.tensor(np.stack([n.apex for n in children]), dtype=torch.float32)
        # broadcast: [N,1,D] vs [1,M,D]
        xi = self._angle_at(pa.unsqueeze(1), ca.unsqueeze(0))
        apertures = torch.tensor([n.aperture for n in parents], dtype=torch.float32)
        margins = apertures.unsqueeze(1) - xi
        return margins.detach().cpu().numpy().astype(np.float32)

    def overlap_score(self, a: ConeNode, b: ConeNode) -> float:
        """Symmetric soft cone overlap; 1.0 means one fully contains the other."""
        return float(min(self.contains(a, b), self.contains(b, a)))

    def find_entailments(
        self,
        nodes: "Sequence[ConeNode]",
        threshold: float = 0.0,
    ) -> "list[tuple[ConeNode, ConeNode]]":
        """Return all (parent, child) pairs where contains(parent, child) > threshold."""
        result = []
        for i, p in enumerate(nodes):
            for j, c in enumerate(nodes):
                if i != j and self.contains(p, c) > threshold:
                    result.append((p, c))
        return result

    # --- semiotic distancing: tension, flow, energy --------------------------
    def containment_asymmetry(self, a: ConeNode, b: ConeNode) -> float:
        """Signed entailment direction contains(a,b) - contains(b,a); >0 means a entails b."""
        return float(self.contains(a, b) - self.contains(b, a))

    def tension(self, a: ConeNode, b: ConeNode) -> float:
        """High overlap with symmetric (ambiguous) containment = redundancy/contradiction tension."""
        return float(self.overlap_score(a, b) - abs(self.containment_asymmetry(a, b)))

    def pair_kind(
        self,
        a: ConeNode,
        b: ConeNode,
        min_overlap: float = 0.0,
        dir_margin: float = 0.05,
    ) -> str:
        """Bucket a pair as entailment, redundancy, contradiction, or independent."""
        overlap = self.overlap_score(a, b)
        if overlap <= min_overlap:
            return "independent"
        if abs(a.aperture - _MIN_APERTURE) < _EPS and abs(b.aperture - _MIN_APERTURE) < _EPS:
            return "aperture_degenerate"
        asym = self.containment_asymmetry(a, b)
        if abs(asym) > dir_margin:
            return "entailment"
        if self.contains(a, b) <= 0.0 and self.contains(b, a) <= 0.0:
            return "contradiction"
        return "redundancy"

    def tension_scan(
        self,
        nodes: "Sequence[ConeNode]",
        top_n: int = 10,
        min_overlap: float = 0.0,
        max_candidates: int = 256,
    ) -> "list[tuple[NodeId, NodeId, float, str]]":
        """Return the top_n worst (a_id, b_id, tension, kind) pairs sharing an octave."""
        pool = list(nodes)[:max_candidates]
        if len(pool) < 2:
            return []
        result: list[tuple[NodeId, NodeId, float, str]] = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                if a.prefix != b.prefix:
                    continue
                ov = self.overlap_score(a, b)
                if ov <= min_overlap:
                    continue
                result.append((a.id, b.id, self.tension(a, b), self.pair_kind(a, b, min_overlap)))
        result.sort(key=lambda t: t[2], reverse=True)
        return result[:top_n]

    def geodesic_distance(self, a: ConeNode, b: ConeNode) -> float:
        """Guarded hyperbolic distance between two cone apices on the Lorentz manifold."""
        if a.apex is b.apex or np.array_equal(a.apex, b.apex):
            return 0.0
        pa = torch.from_numpy(a.apex).double()
        pb = torch.from_numpy(b.apex).double()
        return float(self.manifold.dist(pa, pb).item())

    def flow_weight(self, focus: ConeNode, other: ConeNode) -> float:
        """Entailment gradient: asymmetry per unit distance; sign gives flow direction."""
        return float(self.containment_asymmetry(focus, other) / (self.geodesic_distance(focus, other) + _EPS))

    def flow_neighbors(
        self,
        focus: ConeNode,
        nodes: "Sequence[ConeNode]",
        k: int = 5,
    ) -> "list[tuple[NodeId, float, str]]":
        """Rank neighbors by entailment gradient; direction down=focus-generalizes, up=focus-specializes."""
        scored: list[tuple[NodeId, float, str]] = []
        for n in nodes:
            if n.id == focus.id:
                continue
            w = self.flow_weight(focus, n)
            scored.append((n.id, w, "down" if w >= 0 else "up"))
        scored.sort(key=lambda t: abs(t[1]), reverse=True)
        return scored[:k]

    def context_energy(self, nodes: "Sequence[ConeNode]") -> float:
        """Semiotic spread = sum of pairwise geodesic distances of a context set."""
        pool = list(nodes)
        total = 0.0
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                total += self.geodesic_distance(pool[i], pool[j])
        return float(total)

    def select_representatives(
        self,
        nodes: "Sequence[ConeNode]",
        k: int,
    ) -> "tuple[list[ConeNode], float]":
        """Greedy farthest-point k-center; returns reps and coverage energy (sum min-dist to a rep)."""
        pool = list(nodes)
        if k <= 0 or not pool:
            return [], 0.0
        if k >= len(pool):
            return pool, 0.0
        reps = [pool[0]]
        while len(reps) < k:
            best, best_d = None, -1.0
            for n in pool:
                if any(n.id == r.id for r in reps):
                    continue
                d = min(self.geodesic_distance(n, r) for r in reps)
                if d > best_d:
                    best, best_d = n, d
            if best is None:
                break
            reps.append(best)
        coverage = sum(min(self.geodesic_distance(n, r) for r in reps) for n in pool)
        return reps, float(coverage)

    def energy_contribution(self, node: ConeNode, context: "Sequence[ConeNode]") -> float:
        """Marginal energy a node adds = summed distance to the rest of the context set."""
        return float(sum(self.geodesic_distance(node, o) for o in context if o.id != node.id))

    def _lorentz_mean(self, apices: "Sequence[np.ndarray]") -> "np.ndarray":
        """Euclidean-mean the apices then project back onto the hyperboloid (guarded centroid)."""
        stacked = torch.from_numpy(np.stack(apices)).float()
        mean = stacked.mean(dim=0)
        proj = self.manifold.projx(mean)
        return proj.detach().cpu().numpy().astype(np.float64)

    # --- dispel operations ---------------------------------------------------
    def merge_nodes(self, a: ConeNode, b: ConeNode) -> ConeNode:
        """Collapse a redundant pair to one cone: midpoint apex, widest aperture, union members."""
        import dataclasses
        apex = self._lorentz_mean([a.apex, b.apex])
        return dataclasses.replace(
            a,
            apex=apex,
            aperture=max(a.aperture, b.aperture),
            members=tuple(dict.fromkeys((*a.members, *b.members))),
        )

    def reparent(
        self,
        child: ConeNode,
        candidates: "Sequence[ConeNode]",
    ) -> "NodeId | None":
        """Pick the candidate that most decisively entails child (max containment)."""
        best, best_c = None, 0.0
        for cand in candidates:
            if cand.id == child.id:
                continue
            c = self.contains(cand, child)
            if c > best_c:
                best, best_c = cand.id, c
        return best

    def summarize_cluster(self, nodes: "Sequence[ConeNode]") -> "ConeNode | None":
        """Synthesize one umbrella cone whose aperture is widened to contain every input."""
        import dataclasses
        pool = list(nodes)
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        apex_np = self._lorentz_mean([n.apex for n in pool])
        p = torch.from_numpy(apex_np).float().unsqueeze(0)
        required = _MIN_APERTURE
        for n in pool:
            c = torch.from_numpy(n.apex).float().unsqueeze(0)
            required = max(required, float(self._angle_at(p, c).item()))
        members: tuple = ()
        for n in pool:
            members = (*members, *n.members)
        return dataclasses.replace(
            pool[0],
            apex=apex_np,
            aperture=required + _EPS,
            members=tuple(dict.fromkeys(members)),
            digest=None,
        )

    def dispel_plan(
        self,
        scan_result: "Sequence[tuple[NodeId, NodeId, float, str]]",
    ) -> "list[tuple[str, NodeId, NodeId]]":
        """Map each scanned pair's kind to a remediation op; coherent KB yields an empty plan."""
        op = {"redundancy": "merge", "contradiction": "reparent", "aperture_degenerate": "summarize"}
        return [(op[kind], a, b) for a, b, _t, kind in scan_result if kind in op]

    def fit_and_close(self, tree: ClusterTree) -> "list[ConeNode]":
        """Fit cones then close transitivity in one call; the recommended production API."""
        nodes = list(self.fit(tree))
        return self.close_transitivity(nodes, list(tree.edges))

    def close_transitivity(
        self,
        nodes: "Sequence[ConeNode]",
        edges: "Sequence[tuple[NodeId, NodeId]]",
    ) -> "list[ConeNode]":
        """Widen apertures so containment closes over the transitive hull; apices unchanged."""
        import dataclasses
        from collections import defaultdict, deque

        by_id = {n.id: n for n in nodes}
        children: dict = defaultdict(set)
        for p, c in edges:
            children[p].add(c)

        def _descendants(nid: NodeId) -> set:
            seen: set = set()
            queue: deque = deque(children.get(nid, set()))
            while queue:
                d = queue.popleft()
                if d not in seen:
                    seen.add(d)
                    queue.extend(children.get(d, set()))
            return seen

        result: list = []
        for node in nodes:
            desc = _descendants(node.id)
            if not desc:
                result.append(node)
                continue
            p = torch.from_numpy(node.apex).float().unsqueeze(0)
            required = node.aperture
            for did in desc:
                if did not in by_id:
                    continue
                d_node = by_id[did]
                c = torch.from_numpy(d_node.apex).float().unsqueeze(0)
                angle = float(self._angle_at(p, c).item())
                if angle > required:
                    required = angle
            if required > node.aperture:
                node = dataclasses.replace(node, aperture=required + _EPS)
            result.append(node)
        return result
