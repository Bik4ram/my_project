"""
Vector store interface.

Every other plane depends on this Protocol, not on `PgvectorStore`
directly — consistent with every prior part's module-boundary rule. This
is what makes the pgvector-vs-Qdrant decision (see this part's design
doc) a swappable implementation detail rather than something wired
throughout the codebase: if a future scale threshold makes Qdrant the
better call after all, a `QdrantStore` implementing this same Protocol
is the only new code required — Part 11 (retrieval) and Part 8/9
(chunking/embedding, once built) never import `PgvectorStore` directly.
"""
from __future__ import annotations

from typing import Protocol

from semrag.storage.vector.models import VectorChunk, VectorSearchFilter, VectorSearchResult


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[VectorChunk]) -> None: ...

    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    def delete_by_project(self, project_id: str) -> None: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]: ...
