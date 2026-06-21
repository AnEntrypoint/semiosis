"""Tests for KnowledgeBase save/load session continuity."""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
geoopt = pytest.importorskip("geoopt")

from core.agent_api import KnowledgeBase  # noqa: E402
from core.settings import Settings  # noqa: E402

FACTS = ["alpha unique term", "beta distinct phrase", "gamma separate idea", "delta other concept"]


def _settings() -> Settings:
    s = Settings()
    s.cone.epochs = 4
    return s


def _tmp() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def test_save_load_roundtrip_equivalent() -> None:
    kb = KnowledgeBase(_settings())
    kb.ingest(FACTS)
    kb.remember("user prefers ascii", "p1")
    kb.record_outcome("alpha", ["beta distinct phrase"])
    path = _tmp()
    try:
        kb.save(path)
        kb2 = KnowledgeBase.load(path, _settings())
        assert kb.search_texts("alpha unique term", 2) == kb2.search_texts("alpha unique term", 2)
        assert kb2._usage.get("beta distinct phrase") == 1
        assert "ascii" in kb2.recall("x", 200)
    finally:
        os.remove(path)


def test_save_load_empty_kb() -> None:
    kb = KnowledgeBase(_settings())
    path = _tmp()
    try:
        kb.save(path)
        kb2 = KnowledgeBase.load(path, _settings())
        assert kb2.search("anything") == []
    finally:
        os.remove(path)


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.load(os.path.join(tempfile.gettempdir(), "does-not-exist-xyz.json"))
