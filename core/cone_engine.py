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

    def fit_and_close(self, tree: ClusterTree) -> "list[ConeNode]":
        """Fit cones then close transitivity in one call; the recommended production API."""
        nodes = list(self.fit(tree))
        return self.close_transitivity(nodes, list(tree.edges))

    def close_transitivity(
        self,
        nodes: "Sequence[ConeNode]",
        edges: "Sequence[tuple[NodeId, NodeId]]",
    ) -> "list[ConeNode]":
        """Expand apertures so cone containment closes over the transitive hull of edges.

        The Ganea 2018 loss trains direct edges only; skip-N containment is not
        guaranteed. This post-processing pass computes every node's full descendant
        set and widens its aperture to cover the maximum geodesic angle to any
        descendant. Apices are unchanged; only apertures grow.
        """
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
