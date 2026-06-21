"""JSON-serializable dict conversion for ConeNode; lossless round-trip via float64 apex list."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interfaces import ConeNode, NodeId, PhraseId, Prefix


def cone_node_to_dict(node: ConeNode) -> dict[str, Any]:
    """Convert ConeNode to a plain dict; apex stored as list[float] (float64)."""
    return {
        "id": str(node.id),
        "apex": node.apex.tolist(),
        "aperture": float(node.aperture),
        "prefix": int(node.prefix),
        "members": [str(m) for m in node.members],
        "label": node.label,
        "digest": node.digest,
        "pinned": bool(node.pinned),
        "centroid": list(node.centroid) if node.centroid is not None else None,
    }


def cone_node_from_dict(d: dict[str, Any]) -> ConeNode:
    """Reconstruct ConeNode from a dict produced by cone_node_to_dict."""
    return ConeNode(
        id=NodeId(str(d["id"])),
        apex=np.array(d["apex"], dtype=np.float64),
        aperture=float(d["aperture"]),
        prefix=Prefix(int(d["prefix"])),
        members=tuple(PhraseId(str(m)) for m in d.get("members", [])),
        label=d.get("label"),
        digest=d.get("digest"),
        pinned=bool(d.get("pinned", False)),
        centroid=tuple(d["centroid"]) if d.get("centroid") is not None else None,
    )
