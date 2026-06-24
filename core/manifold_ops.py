"""Manifold geometry utilities for hyperbolic embedding space."""
from __future__ import annotations
import math
import numpy as np

_EPS = 1e-9


def lorentz_project(vec: np.ndarray) -> np.ndarray:
    """Project unit vector in R^d to Lorentz hyperboloid: prepend x0=sqrt(1+||v||^2)."""
    x0 = math.sqrt(1.0 + float(np.dot(vec, vec)))
    return np.concatenate([[x0], vec])


def frechet_mean(vecs: list, n_iter: int = 20) -> np.ndarray:
    """Iterative Frechet mean on hyperboloid; falls back to arithmetic mean."""
    if not vecs:
        raise ValueError("empty vecs")
    arr = np.array(vecs, dtype=float)
    # Normalize each to unit sphere first
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms > _EPS, norms, 1.0)
    # Try torch+geoopt Frechet mean
    try:
        import torch, geoopt  # noqa: F401
        manifold = geoopt.Lorentz()
        pts = torch.tensor(np.apply_along_axis(lorentz_project, 1, arr), dtype=torch.float64)
        mean = manifold.frechet_mean(pts)
        result = mean.detach().numpy()[1:]  # drop x0
        norm = np.linalg.norm(result)
        return result / (norm + _EPS)
    except Exception:
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / (norm + _EPS)


def twonn_intrinsic_dim(vecs: list) -> float:
    """TwoNN intrinsic dimensionality estimator (Facco 2017)."""
    if len(vecs) < 3:
        return 1.0
    arr = np.array(vecs, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms > _EPS, norms, 1.0)
    mus = []
    for i in range(len(arr)):
        dists = 1.0 - np.dot(arr, arr[i])  # cosine distances
        dists[i] = float("inf")            # exclude self
        idx = np.argpartition(dists, 2)[:2]
        d1, d2 = sorted([dists[idx[0]], dists[idx[1]]])
        if d1 > _EPS:
            mus.append(d2 / d1)
    if not mus:
        return 1.0
    mean_log = float(np.mean(np.log(np.array(mus) + _EPS)))
    return max(0.1, 1.0 / (mean_log + _EPS)) if mean_log > 0 else 1.0
