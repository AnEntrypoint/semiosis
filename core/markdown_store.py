"""Markdown tree persistence: the cone DAG as a human-readable, grep-searchable folder structure."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence

from .interfaces import ConeNode, NodeId, Prefix, phrase_to_text_index
from .serialization import cone_node_from_dict, cone_node_to_dict

_META_DIR = "_meta"


def _slug(text: str, node_id: str, max_len: int = 40) -> str:
    """Filesystem-safe slug from a label/digest, suffixed with a short id hash for uniqueness."""
    base = re.sub(r"[^a-z0-9]+", "-", (text or "node").lower()).strip("-")[:max_len] or "node"
    h = hashlib.blake2b(node_id.encode("utf-8"), digest_size=3).hexdigest()
    return f"{base}-{h}"


def _frontmatter(name: str, description: str) -> str:
    desc = " ".join((description or name).split())[:200]
    return f"---\nname: {name}\ndescription: {desc}\n---\n"


def save_markdown_tree(root_dir: "str | os.PathLike[str]", texts: Sequence[str],
                       nodes: Sequence[ConeNode], extra_meta: dict | None = None) -> None:
    """Write the finest-octave tree as browsable markdown; every node of every octave to _meta."""
    root = Path(root_dir)
    meta_nodes = root / _META_DIR / "nodes"
    meta_nodes.mkdir(parents=True, exist_ok=True)
    kb_meta = {"version": 2, "texts": list(texts)}
    kb_meta.update(extra_meta or {})
    (root / _META_DIR / "kb.json").write_text(json.dumps(kb_meta), encoding="utf-8")
    for n in nodes:
        safe = re.sub(r"[^A-Za-z0-9@._-]+", "_", str(n.id))
        (meta_nodes / f"{safe}.json").write_text(json.dumps(cone_node_to_dict(n)), encoding="utf-8")

    by_octave: dict[int, list[ConeNode]] = {}
    for n in nodes:
        by_octave.setdefault(int(n.prefix), []).append(n)
    if not by_octave:
        (root / "README.md").write_text(
            _frontmatter("knowledge-base", "empty knowledge base") + "\nEmpty.\n", encoding="utf-8")
        return
    finest = max(by_octave)
    finest_nodes = {n.id: n for n in by_octave[finest]}
    children: dict[NodeId, list[NodeId]] = {}
    for n in finest_nodes.values():
        if n.parent is not None and n.parent in finest_nodes:
            children.setdefault(n.parent, []).append(n.id)
    roots = [n for n in finest_nodes.values() if n.parent is None or n.parent not in finest_nodes]

    def node_name(n: ConeNode) -> str:
        return _slug(n.label or n.digest or _first_text(n), str(n.id))

    def _first_text(n: ConeNode) -> str:
        for m in n.members:
            idx = phrase_to_text_index(m, len(texts))
            if idx is not None:
                return texts[idx]
        return str(n.id)

    def write_node(n: ConeNode, parent_dir: Path) -> str:
        """Write one node; returns the relative link target it was written to."""
        name = node_name(n)
        desc = n.digest or n.label or _first_text(n)
        kids = children.get(n.id, [])
        if kids:
            d = parent_dir / name
            d.mkdir(parents=True, exist_ok=True)
            links = []
            for cid in kids:
                target = write_node(finest_nodes[cid], d)
                links.append(f"- [{target}]({target})")
            body = _frontmatter(name, desc) + f"\n# {name}\n\n{desc}\n\n## Contents\n\n" + "\n".join(links) + "\n"
            (d / "README.md").write_text(body, encoding="utf-8")
            return f"{name}/README.md"
        lines = [_frontmatter(name, desc), f"\n# {name}\n\n{desc}\n\n## Texts\n"]
        for m in n.members:
            idx = phrase_to_text_index(m, len(texts))
            if idx is not None:
                lines.append(f"- {texts[idx]}")
        (parent_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"{name}.md"

    top_links = []
    for r in sorted(roots, key=lambda n: str(n.id)):
        target = write_node(r, root)
        top_links.append(f"- [{target}]({target})")
    octave_line = ", ".join(str(o) for o in sorted(by_octave))
    readme = (
        _frontmatter("knowledge-base", f"{len(texts)} texts structured over octaves {octave_line}")
        + f"\n# Knowledge base\n\n{len(texts)} texts; octaves {octave_line}; "
        + f"{len(finest_nodes)} nodes at the finest octave ({finest}).\n"
        + "Machine-layer metadata lives in `_meta/` (one JSON per cone node, every octave).\n\n"
        + "## Sections\n\n" + "\n".join(top_links) + "\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")


def load_markdown_tree(root_dir: "str | os.PathLike[str]") -> tuple[list[str], list[ConeNode], dict]:
    """Restore texts + every cone node from the companion metadata; markdown is the human surface."""
    root = Path(root_dir)
    kb_path = root / _META_DIR / "kb.json"
    if not kb_path.exists():
        raise FileNotFoundError(f"not a markdown KB tree (missing {kb_path})")
    meta = json.loads(kb_path.read_text(encoding="utf-8"))
    texts = list(meta.get("texts", []))
    nodes = []
    nodes_dir = root / _META_DIR / "nodes"
    if nodes_dir.exists():
        for p in sorted(nodes_dir.glob("*.json")):
            nodes.append(cone_node_from_dict(json.loads(p.read_text(encoding="utf-8"))))
    return texts, nodes, meta
