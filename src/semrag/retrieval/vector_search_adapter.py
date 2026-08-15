"""
Adapter from Part 6's `VectorSearchResult` to this part's `RetrievalResult`.

Part 6's `VectorSearchResult` doesn't carry `source_node_id` — it wasn't
needed for anything Part 6 itself does. Rather than modify Part 6's
already-shipped model (and risk rippling a change through everything that
already depends on it), this adapter does one batch lookup of
`source_node_id` for the returned chunk_ids directly against the `chunks`
table, and attaches it — the same "small adapter at the boundary, not a
change to the thing you're adapting" approach used throughout this
series for cross-part integration.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.retrieval.models import RetrievalResult
from semrag.storage.vector.models import VectorSearchResult
from semrag.storage.vector.pgvector_models import ChunkRow


def to_retrieval_results(
    session: Session, vector_results: list[VectorSearchResult], signal_name: str = "vector"
) -> list[RetrievalResult]:
    if not vector_results:
        return []

    chunk_ids = [r.chunk_id for r in vector_results]
    rows = session.execute(
        select(
            ChunkRow.chunk_id,
            ChunkRow.source_node_id,
            ChunkRow.language,
            ChunkRow.category,
            ChunkRow.parser_name,
            ChunkRow.chunker_name,
            ChunkRow.symbol_name,
        ).where(ChunkRow.chunk_id.in_(chunk_ids))
    ).all()
    row_by_chunk = {row.chunk_id: row for row in rows}

    return [
        RetrievalResult(
            chunk_id=r.chunk_id,
            project_id=r.project_id,
            relative_path=r.relative_path,
            chunk_type=r.chunk_type,
            text=r.text,
            source_node_id=row_by_chunk[r.chunk_id].source_node_id,
            line_start=r.line_start,
            line_end=r.line_end,
            language=row_by_chunk[r.chunk_id].language or r.language,
            category=row_by_chunk[r.chunk_id].category or r.category,
            parser_name=row_by_chunk[r.chunk_id].parser_name or r.parser_name,
            chunker_name=row_by_chunk[r.chunk_id].chunker_name or r.chunker_name,
            symbol_name=row_by_chunk[r.chunk_id].symbol_name or r.symbol_name,
            fused_score=r.score,
            signal_scores={signal_name: r.score},
        )
        for r in vector_results
    ]
