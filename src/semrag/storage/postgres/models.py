"""
PostgreSQL schema — SQLAlchemy 2.0 declarative models.

Design decisions stated once here, since they apply across every table:

**Deterministic IDs as primary keys, not surrogate auto-increment.**
`ir_nodes.id` and `ir_edges.id` are the exact deterministic strings Part 4's
`id_scheme.py` produces. This is what makes every write in this module an
upsert (`INSERT ... ON CONFLICT DO UPDATE`) rather than an insert — running
the ingestion pipeline twice on an unchanged project must not create
duplicate rows, and a surrogate auto-increment PK would make that
guarantee something the *application* has to enforce (via an extra lookup
before every insert) rather than something the *database* enforces for
free via a natural key.

**JSONB for parser-payload-shaped data, real columns for everything
queried directly.** `attributes` (on `ir_nodes`) and `candidate_ids` (on
`ir_edges`) are JSONB because their shape varies by node/edge kind and
isn't something Part 6 (vector DB) or Part 10 (knowledge graph) needs to
filter on directly — they read the whole blob. Everything a query in
Part 11 (retrieval) would filter or sort on — `project_id`, `kind`,
`relative_path`, `resolved`, `severity` — is a real typed column with an
index, not buried in JSON. Mixing these two storage strategies
deliberately, not defaulting to "put everything in JSONB because Postgres
supports it" — that default produces a database that can store data but
not efficiently query it.

**A separate `parse_diagnostics` table, not diagnostics nested in
`ir_nodes.attributes`.** Diagnostics need to be queryable on their own
("show me every file with a parse error across the project") independent
of which node they're attached to, and a file that fails to parse
entirely may have diagnostics with *no* corresponding IR node at all.
Modeling them as their own table with a `relative_path` column (not a
foreign key to `ir_nodes`, since a failed parse may have zero nodes)
handles both cases without a nullable-FK special case.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# JSONB is Postgres-specific; fall back to generic JSON for any future
# non-Postgres test target (e.g. SQLite in a lightweight unit test) by
# using JSON().with_variant(JSONB, "postgresql") wherever a JSON column
# appears below, rather than importing the Postgres dialect type directly.
_JSONType = JSON().with_variant(JSONB(), "postgresql")


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    root_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()"
    )


class ScanManifestRow(Base):
    """One row per completed scan — an append-only audit log, not the
    current-state table (that's `FileRow`, below). Kept lightweight: the
    full per-file listing is NOT duplicated here as a JSONB blob, since
    that would mean two sources of truth for file state that can drift.
    """
    __tablename__ = "scan_manifests"

    scan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), nullable=False, index=True
    )
    root_path: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    git_metadata: Mapped[dict] = mapped_column(_JSONType, nullable=False, default=dict)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_scan_manifests_project_started", "project_id", "started_at"),
    )


class FileRow(Base):
    """Current-state table: one row per file *currently present* in the
    project, upserted on every scan. A file deleted from the repo is
    deleted from this table too (see `PostgresManifestStore.save_manifest`
    in manifest_store.py) — this table always reflects "what exists right
    now", not history. History lives in `scan_manifests` at the
    scan-summary level and in git itself at the content level; this
    module does not attempt to duplicate git's own history.
    """
    __tablename__ = "files"

    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), primary_key=True
    )
    relative_path: Mapped[str] = mapped_column(String(1024), primary_key=True)

    absolute_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mtime: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    hash_algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    extension: Mapped[str] = mapped_column(String(32), nullable=False)
    is_binary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    git_info: Mapped[dict | None] = mapped_column(_JSONType)
    raw_dependency_refs: Mapped[list] = mapped_column(_JSONType, nullable=False, default=list)
    last_scan_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("scan_manifests.scan_id"))
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()"
    )

    __table_args__ = (
        Index("ix_files_project_category", "project_id", "category"),
        Index("ix_files_content_hash", "content_hash"),
    )


class IRNodeRow(Base):
    __tablename__ = "ir_nodes"

    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    parent_file_id: Mapped[str | None] = mapped_column(
        String(512), ForeignKey("ir_nodes.id")
    )
    attributes: Mapped[dict] = mapped_column(_JSONType, nullable=False, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()"
    )

    __table_args__ = (
        Index("ix_ir_nodes_project_kind_name", "project_id", "kind", "name"),
    )


class IREdgeRow(Base):
    __tablename__ = "ir_edges"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("ir_nodes.id"), nullable=False, index=True
    )
    target_name: Mapped[str] = mapped_column(String(512), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(512), ForeignKey("ir_nodes.id"), index=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    ambiguous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    candidate_ids: Mapped[list] = mapped_column(_JSONType, nullable=False, default=list)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()"
    )


class ParseDiagnosticRow(Base):
    """Not a foreign key to ir_nodes.id — a file that fails to parse
    entirely produces diagnostics with no corresponding node. Queried by
    (project_id, relative_path) or (project_id, severity), not by node."""
    __tablename__ = "parse_diagnostics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("projects.project_id"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    parser_name: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    line: Mapped[int | None] = mapped_column(Integer)
    column: Mapped[int | None] = mapped_column(Integer)
    scan_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("scan_manifests.scan_id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    __table_args__ = (
        Index("ix_parse_diagnostics_project_path", "project_id", "relative_path"),
    )
