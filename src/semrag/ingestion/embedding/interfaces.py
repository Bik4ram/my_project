"""
Embedding provider and cache interfaces.

`EmbeddingProvider` is the model-agnostic boundary Part 1 requires: every
call site in this system (this pipeline, and later re-embed workflows)
depends on this Protocol, never on a concrete provider — swapping the
local TF-IDF baseline for a real neural model behind an OpenAI-compatible
endpoint (Ollama, vLLM, llama.cpp) is a config change, not a code change,
matching exactly how Part 6's `VectorStore` and Part 7's `GraphStore`
Protocols were built.

`dimension` is a property, not a constant, because it's a property of
*which provider is configured* — Part 6's `EMBEDDING_DIMENSION = 768`
constant is a schema-level commitment that must match whatever provider
is actually running; mismatches are a real operational hazard (see this
part's design doc for how that's caught rather than silently corrupting
the vector column).
"""
from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingCache(Protocol):
    """Keyed by content hash, not chunk_id — deliberately: what determines
    whether an embedding needs recomputing is whether the *text* changed,
    not whether the chunk's identity changed. Two different chunk_ids with
    identical text (rare, but real — e.g. a boilerplate license header
    chunked separately per file) should share one cache entry, and a
    chunk whose text is byte-identical after a re-chunk should hit the
    cache even if upstream chunking logic changed its chunk_id."""

    def get(self, text_hash: str) -> list[float] | None: ...

    def set(self, text_hash: str, embedding: list[float]) -> None: ...

    def get_many(self, text_hashes: list[str]) -> dict[str, list[float]]:
        """Batch lookup. Returns only the hits. Default falls back to
        per-key ``get`` for caches that don't override."""
        out: dict[str, list[float]] = {}
        for h in text_hashes:
            v = self.get(h)
            if v is not None:
                out[h] = v
        return out

    def set_many(self, items: dict[str, list[float]]) -> None:
        """Batch write. Default falls back to per-key ``set``."""
        for k, v in items.items():
            self.set(k, v)

    def flush(self) -> None:
        """Persist any pending writes. No-op for in-memory caches."""
