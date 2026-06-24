"""Summarizer implementations: StubSummarizer (no LLM) and LLMSummarizer (HTTP)."""
from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class CategorizationSummarizer:
    """LLM-backed summarizer: generates opinionated 1-sentence category label from member texts."""

    def __init__(
        self,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        timeout: int = 10,
    ) -> None:
        import os
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._timeout = timeout

    def summarize(self, node_id: str, member_texts: list[str]) -> str:
        """Call LLM to produce a dense opinionated summary; fallback to joined preview on error."""
        if not member_texts:
            return f"category:{node_id}"
        preview = "; ".join(member_texts[:5])
        if not self._api_key:
            return f"Category covering: {preview[:120]}"
        prompt = (
            f"In one sentence, state the most specific unifying concept that covers all of these: {preview}. "
            f"Be opinionated and precise -- name the concept, not a description of the list."
        )
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.3,
        }).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"Category covering: {preview[:120]}"
