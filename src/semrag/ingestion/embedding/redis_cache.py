"""
Redis-backed embedding cache.

Stores text-hash → embedding list under
``semrag:embedding:{model_name}:{text_hash}`` so different models
don't share namespace. Embeddings are stored as a JSON-encoded list of
floats; pickling is the alternative but JSON is portable across Python
versions and inspectable.

The cache is read-and-write batched through ``MGET`` / pipeline to
avoid a per-item round-trip on the hot path. ``flush`` is a no-op
because Redis is already persistent per-call.
"""
from __future__ import annotations

import json
from typing import Iterable


class RedisEmbeddingCache:
    """Redis-backed embedding cache. Implements the same surface as
    ``InMemoryEmbeddingCache`` / ``JSONFileEmbeddingCache``."""

    def __init__(self, client, model_name: str):
        self._client = client
        self._model_name = model_name

    def _key(self, text_hash: str) -> str:
        return f"semrag:embedding:{self._model_name}:{text_hash}"

    def get(self, text_hash: str) -> list[float] | None:
        raw = self._client.get(self._key(text_hash))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set(self, text_hash: str, embedding: list[float]) -> None:
        self._client.set(self._key(text_hash), json.dumps(embedding))

    def get_many(self, text_hashes: Iterable[str]) -> dict[str, list[float]]:
        hashes = list(text_hashes)
        if not hashes:
            return {}
        keys = [self._key(h) for h in hashes]
        raws = self._client.mget(keys)
        out: dict[str, list[float]] = {}
        for h, raw in zip(hashes, raws):
            if raw is None:
                continue
            try:
                out[h] = json.loads(raw)
            except (TypeError, ValueError):
                continue
        return out

    def set_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        pipe = self._client.pipeline()
        for h, embedding in items.items():
            pipe.set(self._key(h), json.dumps(embedding))
        pipe.execute()

    def flush(self) -> None:
        """No-op: Redis writes are already durable per-call."""
        return None
