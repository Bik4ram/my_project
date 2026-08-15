"""
Embedding cache implementations.

Same two-tier pattern as Part 2's `manifest_store.py`
(`InMemoryManifestStore` / `JSONFileManifestStore`): an in-memory version
for tests and short-lived scripts, and a disk-persisted version that's
genuinely usable for local development without standing up a database
just for this. A production deployment would likely want this as a table
(keyed by content hash) alongside Part 5's other tables rather than loose
files — flagged as future work, same as Part 2 flagged for its own
manifest store, not built here to avoid scope creep into a fourth
storage-layer decision this part doesn't need to make.

Cache keys are blake3 content hashes of the chunk text (reusing Part 2's
`hashing.py` rather than reimplementing hashing a second time), not
chunk_id — see `interfaces.py`'s `EmbeddingCache` docstring for why that
distinction is deliberate.
"""
from __future__ import annotations

import json
from pathlib import Path

from semrag.ingestion.scanner.hashing import hash_bytes


def text_hash(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


class InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def get(self, text_hash: str) -> list[float] | None:
        return self._store.get(text_hash)

    def set(self, text_hash: str, embedding: list[float]) -> None:
        self._store[text_hash] = embedding

    def get_many(self, text_hashes: list[str]) -> dict[str, list[float]]:
        return {h: self._store[h] for h in text_hashes if h in self._store}

    def set_many(self, items: dict[str, list[float]]) -> None:
        self._store.update(items)

    def flush(self) -> None:
        """No-op for the in-memory cache."""
        return None


class JSONFileEmbeddingCache:
    """One JSON file per (model_name, dimension) combination — a cache
    entry from one embedding model is never valid for another (different
    model = different vector space), so the cache is namespaced by model
    identity at the file-path level rather than needing a model-name
    field checked on every read."""

    def __init__(self, cache_dir: Path, model_name: str):
        self._path = cache_dir / f"{model_name}.embedding_cache.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, list[float]] = self._load()
        self._dirty = False

    def _load(self) -> dict[str, list[float]]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _save(self) -> None:
        if not self._dirty:
            return
        self._path.write_text(json.dumps(self._store))
        self._dirty = False

    def flush(self) -> None:
        """Persist any pending writes immediately."""
        self._save()

    def get(self, text_hash: str) -> list[float] | None:
        return self._store.get(text_hash)

    def set(self, text_hash: str, embedding: list[float]) -> None:
        self._store[text_hash] = embedding
        self._dirty = True
        self._save()

    def get_many(self, text_hashes: list[str]) -> dict[str, list[float]]:
        return {h: self._store[h] for h in text_hashes if h in self._store}

    def set_many(self, items: dict[str, list[float]]) -> None:
        """Batch write — marks dirty without writing. The caller is
        expected to invoke ``flush()`` once after the batch completes;
        doing a full file rewrite per-batch would defeat the point of
        batching."""
        if not items:
            return
        self._store.update(items)
        self._dirty = True
