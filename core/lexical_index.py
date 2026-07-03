"""Lightweight incremental BM25 inverted index -- lexical half of hybrid search (SeekStorm-style dual index)."""
from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Incremental BM25 over doc_id -> text; supports add() without full rebuild."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)   # term -> {doc_id: tf}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0
        self._n_docs = 0

    def add(self, doc_id: str, text: str) -> None:
        """Incrementally index one document; O(len(text)), no rebuild of existing postings."""
        if doc_id in self._doc_len:
            self.remove(doc_id)
        tokens = _tokenize(text)
        if not tokens:
            self._doc_len[doc_id] = 0
            self._n_docs += 1
            return
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, c in tf.items():
            self._postings[t][doc_id] = c
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        self._n_docs += 1

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_len:
            return
        for term_map in self._postings.values():
            term_map.pop(doc_id, None)
        self._total_len -= self._doc_len.pop(doc_id)
        self._n_docs -= 1

    def _avg_len(self) -> float:
        return (self._total_len / self._n_docs) if self._n_docs else 0.0

    def score(self, query: str, doc_ids: "list[str] | None" = None) -> list[tuple[str, float]]:
        """Return (doc_id, bm25_score) pairs, sorted descending; optionally restricted to doc_ids."""
        terms = _tokenize(query)
        if not terms or self._n_docs == 0:
            return []
        avg_len = self._avg_len() or 1.0
        scores: dict[str, float] = defaultdict(float)
        for t in set(terms):
            postings = self._postings.get(t)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                if doc_ids is not None and doc_id not in doc_ids:
                    continue
                dl = self._doc_len.get(doc_id, 0)
                denom = tf + self._k1 * (1 - self._b + self._b * dl / avg_len)
                scores[doc_id] += idf * (tf * (self._k1 + 1)) / (denom or 1.0)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
