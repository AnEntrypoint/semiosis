"""KnowledgeBase mixin: NLA activation-blended embeddings with an LRU cache."""
from __future__ import annotations

try:
    from .activation_predictor import stub_activations as _stub_activations
except ImportError:
    _stub_activations = None  # type: ignore[assignment]


class ActivationMixin:
    """Blends encoder output with a fitted NLA activation projection; cached per text."""

    def activation_embed(self, text: str) -> list:
        """Blend encoder + activation projection (weights in AgentSettings) with LRU cache."""
        return self.activation_embed_batch([text])[0]

    def activation_embed_batch(self, texts: list) -> list:
        """Batch-encode texts once; blend each with its activation projection; LRU-cached per text."""
        import numpy as _np
        if not hasattr(self, '_act_embed_cache'):
            self._act_embed_cache: dict = {}
        w = self._settings.agent.activation_blend_encoder_weight
        results: list = [None] * len(texts)
        to_encode: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text in self._act_embed_cache:
                results[i] = self._act_embed_cache[text]
            else:
                to_encode.append((i, text))
        if to_encode:
            missing_texts = [t for _, t in to_encode]
            if self._pipeline is None:
                bases = [list(_stub_activations(t, dim=64)) if _stub_activations else [0.0] * 64
                         for t in missing_texts]
            else:
                bases = self._pipeline._encoder.encode(missing_texts).tolist()
            for (i, text), base in zip(to_encode, bases):
                act_vec = None
                if self._act_predictor is not None and getattr(self._act_predictor, '_fitted', False) and _stub_activations is not None:
                    try:
                        act_raw = self._act_predictor.predict_embedding(_stub_activations(text))
                        act_arr = _np.array(act_raw, dtype=float)
                        base_arr = _np.array(base, dtype=float)
                        if act_arr.shape == base_arr.shape:
                            act_vec = (w * base_arr + (1.0 - w) * act_arr).tolist()
                    except Exception:
                        pass
                result = act_vec if act_vec is not None else list(base)
                results[i] = result
                if len(self._act_embed_cache) < 1024:
                    self._act_embed_cache[text] = result
        return results
