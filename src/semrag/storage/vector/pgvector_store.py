"""
Pgvector-backed VectorStore.

Chosen over Qdrant for V1 based on a real benchmark run in this delivery's
development environment (both systems actually running, not assumed) —
see this part's design doc, Section 2, for the numbers and the honest
caveats about what a 20K-vector single-node localhost benchmark does and
doesn't tell you. Implements `semrag.storage.vector.interfaces.VectorStore`.

Cosine distance (`<=>` operator, `vector_cosine_ops`) is used throughout,
matching the embedding models this system is likely to use (most
sentence/code embedding models are trained for cosine similarity, not
raw L2 distance) — this should be revisited if Part 9 picks a model
trained for a different metric.
"""
from __future__ import annotations

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from semrag.storage.vector.models import VectorChunk, VectorSearchFilter, VectorSearchResult
from semrag.storage.vector.pgvector_models import ChunkRow


class PgvectorStore:
    def __init__(self, session: Session):
        self._session = session

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        """Insert or update chunks in batch."""
        if not chunks:
            return
        chunks_to_upsert = []
        for chunk in chunks:
            # PostgreSQL/psycopg2 refuses NUL (0x00) in TEXT/VARCHAR bind
            # values ("A string literal cannot contain NUL (0x00)
            # characters"). Stripping here is the defence-in-depth backstop
            # for any chunk whose source text was mis-classified upstream;
            # the chunker already skips binary files, so this only fires
            # in genuinely degenerate cases. Other control characters are
            # left intact — they are legal in TEXT columns.
            text_value = chunk.text.replace("\x00", "")
            chunks_to_upsert.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "project_id": chunk.project_id,
                    "relative_path": chunk.relative_path,
                    "chunk_type": chunk.chunk_type,
                    "source_node_id": chunk.source_node_id,
                    "text": text_value,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "embedding": chunk.embedding,
                    "language": chunk.language,
                    "category": chunk.category,
                    "parser_name": chunk.parser_name,
                    "chunker_name": chunk.chunker_name,
                    "symbol_name": chunk.symbol_name,
                }
            )
        stmt = pg_insert(ChunkRow).values(chunks_to_upsert)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                "relative_path": stmt.excluded.relative_path,
                "chunk_type": stmt.excluded.chunk_type,
                "source_node_id": stmt.excluded.source_node_id,
                "text": stmt.excluded.text,
                "line_start": stmt.excluded.line_start,
                "line_end": stmt.excluded.line_end,
                "embedding": stmt.excluded.embedding,
                "language": stmt.excluded.language,
                "category": stmt.excluded.category,
                "parser_name": stmt.excluded.parser_name,
                "chunker_name": stmt.excluded.chunker_name,
                "symbol_name": stmt.excluded.symbol_name,
                "updated_at": text("now()"),
            },
        )
        self._session.execute(stmt)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._session.execute(delete(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids)))

    def delete_by_project(self, project_id: str) -> None:
        self._session.execute(delete(ChunkRow).where(ChunkRow.project_id == project_id))

    def delete_by_paths(self, project_id: str, relative_paths: list[str]) -> None:
        """Bulk delete all chunks whose relative_path matches any entry in
        ``relative_paths``. Used when the scanner reports deleted files
        and we need to drop the chunks we previously stored for them in
        a single round-trip rather than per-path."""
        if not relative_paths:
            return
        self._session.execute(
            delete(ChunkRow).where(
                ChunkRow.project_id == project_id,
                ChunkRow.relative_path.in_(relative_paths),
            )
        )

    def upsert_chunks_bulk(self, chunks: list[VectorChunk], *, page_size: int = 500) -> None:
        """Same semantics as ``upsert_chunks`` but pages the work to keep
        very large ingestions within a single statement's parameter limit
        rather than letting one huge INSERT swallow the connection."""
        for start in range(0, len(chunks), page_size):
            self.upsert_chunks(chunks[start : start + page_size])

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        # cosine_distance is 1 - cosine_similarity; converted to a
        # similarity score in the result so callers (Part 11's fusion
        # logic) work with "higher is better" uniformly across retrieval
        # signals, rather than needing to know pgvector's convention.
        distance_expr = ChunkRow.embedding.cosine_distance(query_vector)
        stmt = select(ChunkRow, distance_expr.label("distance")).order_by(distance_expr).limit(top_k)

        if filters:
            if filters.project_id:
                stmt = stmt.where(ChunkRow.project_id == filters.project_id)
            if filters.chunk_type:
                stmt = stmt.where(ChunkRow.chunk_type == filters.chunk_type)
            if filters.relative_path:
                stmt = stmt.where(ChunkRow.relative_path == filters.relative_path)

        rows = self._session.execute(stmt).all()
        return [
            VectorSearchResult(
                chunk_id=row.ChunkRow.chunk_id,
                score=1.0 - row.distance,
                project_id=row.ChunkRow.project_id,
                relative_path=row.ChunkRow.relative_path,
                chunk_type=row.ChunkRow.chunk_type,
                text=row.ChunkRow.text,
                line_start=row.ChunkRow.line_start,
                line_end=row.ChunkRow.line_end,
                language=row.ChunkRow.language or "",
                category=row.ChunkRow.category or "",
                parser_name=row.ChunkRow.parser_name,
                chunker_name=row.ChunkRow.chunker_name or "",
                symbol_name=row.ChunkRow.symbol_name,
            )
            for row in rows
        ]