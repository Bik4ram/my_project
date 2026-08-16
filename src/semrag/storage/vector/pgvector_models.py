"""
Chunks table — SQLAlchemy model, pgvector-backed.

Embedding dimension is a real constraint, not a detail: pgvector requires
a fixed dimension per column at table-creation time, and that dimension
is a property of whatever embedding model Part 9 ends up using. `768` is
used here as the working assumption (matches common code-embedding model
output sizes, e.g. many CodeBERT-family and jina-code-embedding models),
stated explicitly rather than left implicit, because it WILL need to
change when Part 9 picks a specific model — see this part's design doc,
Section 5 ("Embedding Dimension Is Not Yet a Real Decision") for why that
part, not this one, owns the final number, and what changing it costs
(a migration + full re-embed, not a config flag).

Why a separate table from Part 5's `ir_nodes`, not a `vector` column added
there: chunks and IR nodes have a many-to-one relationship (one module
might produce several chunks — signature, body, assertions — from Part 8's
future chunker) and different lifecycle (embeddings are versioned/refreshed
independently of IR re-parsing, per Part 1's "embedding versioning" and
"embedding refresh" requirements). Coupling them into one table would
force every IR re-parse to touch embedding rows and vice versa.
"""
from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from semrag.storage.postgres.models import Base

EMBEDDING_DIMENSION = 768


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    chunk_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_node_id: Mapped[str | None] = mapped_column(String(512), ForeignKey("ir_nodes.id"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int | None] = mapped_column()
    line_end: Mapped[int | None] = mapped_column()
    # RI-06 retrieval metadata. Nullable so pre-RI-06 rows (and rows
    # written before enrichment) remain readable; ingestion fills them
    # for all new chunks. ``language``/``category``/``chunker_name`` are
    # always populated for enriched chunks; ``parser_name``/``symbol_name``
    # are None where no parser/IR symbol exists.
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunker_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    symbol_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Portable multi-root metadata: which area a chunk's source file
    # belongs to ('rtl'/'tb'/'docs') and that area's absolute root. NULL
    # for legacy single-root projects (source_type defaults to '').
    source_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source_root: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    # Generated column (Postgres computes and maintains this automatically
    # on every insert/update — never written to directly) backing Part 10's
    # lexical/BM25-style search. Real Postgres full-text search (ts_rank_cd
    # over a tsvector), not the literal BM25 algorithm — see Part 10's
    # design doc for that distinction and why it's an honest one to draw.
    text_search: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    __table_args__ = (
        Index("ix_chunks_project_type", "project_id", "chunk_type"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_text_search_gin", "text_search", postgresql_using="gin"),
    )
