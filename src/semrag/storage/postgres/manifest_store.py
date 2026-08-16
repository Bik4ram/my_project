"""
Postgres-backed ManifestStore.

Implements Part 2's `ManifestStore` Protocol — this is the production
replacement for Part 2's `JSONFileManifestStore`, exactly as flagged as
future work there. No change was needed to Part 2's `Scanner` to plug this
in; that's the point of depending on a Protocol instead of a concrete
class.

The tricky part, worth explaining directly: `ManifestStore.get_latest_manifest`
must return a `ScanManifest` whose `.files` dict matches Part 2's in-memory
shape exactly (keyed by relative_path, values are `FileRecord`), but this
store's actual persistence is split across two tables (`scan_manifests` for
the summary, `files` for current per-file state) — so `get_latest_manifest`
reconstructs the in-memory shape by joining the two at read time. This
split-storage/joined-read approach was chosen over storing the full file
list as one JSONB blob on `scan_manifests` specifically so `files` stays
independently queryable (Part 6/Part 11 need "all files in category X",
not just "the file list for scan Y").
"""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from semrag.ingestion.scanner.models import (
    FileRecord,
    GitRepositoryMetadata,
    ScanManifest,
)
from semrag.storage.postgres.models import FileRow, Project, ScanManifestRow


class PostgresManifestStore:
    """Implements `semrag.ingestion.scanner.interfaces.ManifestStore`.
    Takes a live `Session` per call rather than owning its own — matches
    the transaction-boundary philosophy in `session.py`."""

    def __init__(self, session: Session):
        self._session = session

    def get_latest_manifest(self, project_id: str) -> ScanManifest | None:
        latest_scan = self._session.execute(
            select(ScanManifestRow)
            .where(ScanManifestRow.project_id == project_id)
            .order_by(ScanManifestRow.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest_scan is None:
            return None

        project = self._session.execute(
            select(Project).where(Project.project_id == project_id)
        ).scalar_one_or_none()

        file_rows = self._session.execute(
            select(FileRow).where(FileRow.project_id == project_id)
        ).scalars().all()

        files = {
            row.relative_path: FileRecord(
                project_id=row.project_id,
                absolute_path=row.absolute_path,
                relative_path=row.relative_path,
                size_bytes=row.size_bytes,
                mtime=row.mtime,
                content_hash=row.content_hash,
                hash_algorithm=row.hash_algorithm,
                category=row.category,
                extension=row.extension,
                is_binary=row.is_binary,
                git_info=row.git_info,
                raw_dependency_refs=row.raw_dependency_refs or [],
            )
            for row in file_rows
        }

        return ScanManifest(
            scan_id=latest_scan.scan_id,
            project_id=latest_scan.project_id,
            root_path=latest_scan.root_path,
            rtl_root=project.rtl_root if project else None,
            tb_root=project.tb_root if project else None,
            docs_root=project.docs_root if project else None,
            started_at=latest_scan.started_at,
            completed_at=latest_scan.completed_at,
            git_metadata=GitRepositoryMetadata.model_validate(latest_scan.git_metadata),
            file_count=latest_scan.file_count,
            total_bytes=latest_scan.total_bytes,
            files=files,
        )

    def save_manifest(self, manifest: ScanManifest) -> None:
        self._ensure_project(
            manifest.project_id,
            manifest.root_path,
            rtl_root=manifest.rtl_root,
            tb_root=manifest.tb_root,
            docs_root=manifest.docs_root,
        )

        self._session.execute(
            pg_insert(ScanManifestRow)
            .values(
                scan_id=manifest.scan_id,
                project_id=manifest.project_id,
                root_path=manifest.root_path,
                started_at=manifest.started_at,
                completed_at=manifest.completed_at,
                git_metadata=manifest.git_metadata.model_dump(mode="json"),
                file_count=manifest.file_count,
                total_bytes=manifest.total_bytes,
            )
            .on_conflict_do_nothing(index_elements=["scan_id"])
        )

        self._sync_files(manifest)

    def _ensure_project(
        self,
        project_id: str,
        root_path: str,
        rtl_root: str | None = None,
        tb_root: str | None = None,
        docs_root: str | None = None,
    ) -> None:
        """Upsert the project row. Area roots are additive: when provided
        they overwrite stale values; when ``None`` the existing values are
        left untouched (a legacy single-root save must not clobber a
        multi-root project's area roots)."""
        self._session.execute(
            pg_insert(Project)
            .values(
                project_id=project_id,
                root_path=root_path,
                rtl_root=rtl_root,
                tb_root=tb_root,
                docs_root=docs_root,
            )
            .on_conflict_do_update(
                index_elements=["project_id"],
                set_={
                    "root_path": root_path,
                    **(
                        {"rtl_root": rtl_root}
                        if rtl_root is not None
                        else {}
                    ),
                    **({"tb_root": tb_root} if tb_root is not None else {}),
                    **({"docs_root": docs_root} if docs_root is not None else {}),
                },
            )
        )

    def _sync_files(self, manifest: ScanManifest) -> None:
        """Upserts every file in the manifest, then deletes any row for
        this project not present in the manifest — this table always
        reflects "what exists right now" (see models.py's FileRow
        docstring), so a file the scanner no longer sees must be removed,
        not left stale."""
        current_paths = set(manifest.files.keys())

        for relative_path, record in manifest.files.items():
            self._session.execute(
                pg_insert(FileRow)
                .values(
                    project_id=record.project_id,
                    relative_path=record.relative_path,
                    absolute_path=record.absolute_path,
                    size_bytes=record.size_bytes,
                    mtime=record.mtime,
                    content_hash=record.content_hash,
                    hash_algorithm=record.hash_algorithm,
                    category=record.category.value,
                    extension=record.extension,
                    is_binary=record.is_binary,
                    git_info=record.git_info.model_dump(mode="json") if record.git_info else None,
                    raw_dependency_refs=record.raw_dependency_refs,
                    last_scan_id=manifest.scan_id,
                )
                .on_conflict_do_update(
                    index_elements=["project_id", "relative_path"],
                    set_={
                        "absolute_path": record.absolute_path,
                        "size_bytes": record.size_bytes,
                        "mtime": record.mtime,
                        "content_hash": record.content_hash,
                        "hash_algorithm": record.hash_algorithm,
                        "category": record.category.value,
                        "extension": record.extension,
                        "is_binary": record.is_binary,
                        "git_info": record.git_info.model_dump(mode="json") if record.git_info else None,
                        "raw_dependency_refs": record.raw_dependency_refs,
                        "last_scan_id": manifest.scan_id,
                        "updated_at": manifest.completed_at or manifest.started_at,
                    },
                )
            )

        delete_stmt = delete(FileRow).where(FileRow.project_id == manifest.project_id)
        if current_paths:
            delete_stmt = delete_stmt.where(FileRow.relative_path.notin_(current_paths))
        self._session.execute(delete_stmt)
