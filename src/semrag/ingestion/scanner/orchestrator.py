"""
Scanner orchestrator.

This is the subsystem's entry point (`Scanner.scan`) and the only place
that knows how to assemble a full ScanManifest from the lower-level pieces
(walker, hashing, git_metadata, dependency_tracker, change_detector).
Everything else in this package is a pure, independently-testable function
or class; this module wires them together and owns the one genuinely
stateful, side-effecting operation: reading a project's manifest, running
a scan, and deciding what needs re-hashing.

Concurrency: file hashing is I/O-bound (reading potentially large files
from disk/NFS), so a ThreadPoolExecutor is used rather than
multiprocessing — Python's GIL is not the bottleneck here, disk/network
I/O wait is, and threads avoid the pickling overhead multiprocessing would
add for no benefit. The worker count defaults to a modest value because
many semiconductor project filesystems are network-mounted (NFS home
directories are common in EDA environments) where too much hashing
concurrency saturates the network link rather than speeding anything up;
this is exposed as a constructor parameter so it's tunable per-deployment
rather than hardcoded.
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from semrag.core.project_paths import AREA_ORDER
from semrag.ingestion.scanner import git_metadata, hashing
from semrag.ingestion.scanner.change_detector import detect_changes
from semrag.ingestion.scanner.dependency_tracker import extract_raw_dependency_refs
from semrag.ingestion.scanner.interfaces import ManifestStore, ScannerInterface
from semrag.ingestion.scanner.models import (
    ChangeSet,
    FileRecord,
    ScanManifest,
)
from semrag.ingestion.scanner.walker import WalkedFile, walk_repository


class Scanner(ScannerInterface):
    def __init__(self, manifest_store: ManifestStore, max_hash_workers: int = 8):
        self._manifest_store = manifest_store
        self._max_hash_workers = max_hash_workers

    def scan(self, project_id: str, root_path: str) -> tuple[ScanManifest, ChangeSet]:
        root = Path(root_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Scan root does not exist or is not a directory: {root}")

        previous_manifest = self._manifest_store.get_latest_manifest(project_id)
        previous_head_sha = (
            previous_manifest.git_metadata.head_commit_sha
            if previous_manifest and previous_manifest.git_metadata.is_git_repo
            else None
        )

        git_meta = git_metadata.get_repository_metadata(root)
        blob_info_map = {}
        if git_meta.is_git_repo:
            # Bulk git calls once per scan (see git_metadata.py docstring
            # for why this must never become per-file).
            blob_info_map = git_metadata.build_blob_info_map(root, previous_head_sha)

        walked_files = list(walk_repository(root))

        # Split into "can skip re-hash" (git blob SHA unchanged from the
        # previous manifest) vs "must hash" up front, so the thread pool
        # below only does I/O for files that actually need it.
        previous_files_by_path = previous_manifest.files if previous_manifest else {}
        to_hash: list[WalkedFile] = []
        reused_records: dict[str, FileRecord] = {}

        for wf in walked_files:
            git_info = blob_info_map.get(wf.relative_path)
            previous_record = previous_files_by_path.get(wf.relative_path)

            if (
                previous_record is not None
                and git_info is not None
                and previous_record.git_info is not None
                and previous_record.git_info.blob_sha == git_info.blob_sha
            ):
                # Unchanged per git; reuse the previous content hash and
                # dependency refs rather than re-reading the file.
                reused_records[wf.relative_path] = previous_record.model_copy(
                    update={
                        "size_bytes": wf.size_bytes,
                        "mtime": wf.mtime,
                        "git_info": git_info,
                    }
                )
            else:
                to_hash.append(wf)

        hashed_records = self._hash_and_build_records(project_id, to_hash, blob_info_map)

        all_records: dict[str, FileRecord] = {**reused_records, **hashed_records}

        scan_id = str(uuid.uuid4())
        current_manifest = ScanManifest(
            scan_id=scan_id,
            project_id=project_id,
            root_path=str(root),
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            git_metadata=git_meta,
            file_count=len(all_records),
            total_bytes=sum(r.size_bytes for r in all_records.values()),
            files=all_records,
        )

        change_set = detect_changes(previous_manifest, current_manifest)
        self._manifest_store.save_manifest(current_manifest)

        return current_manifest, change_set

    def scan_multi_root(
        self,
        project_id: str,
        area_roots: dict[str, str],
        display_root: str | None = None,
    ) -> tuple[ScanManifest, ChangeSet]:
        """Scan a multi-root project: each area root is walked
        independently and every file's ``relative_path`` is prefixed with
        its area (``rtl/...``, ``tb/...``, ``docs/...``) so the whole
        project shares one manifest with disjoint, portable paths.

        ``area_roots`` maps area name -> absolute root (order-insensitive;
        iteration follows ``AREA_ORDER``). ``display_root`` is the optional
        legacy ``root_path`` used as the manifest's root and as the git
        metadata source when present.

        The same git-blob reuse optimization as the single-root path is
        applied per area root; a changed root must NOT silently re-read
        every file on every scan, but must correctly pick up new/moved
        files. Deleted files are caught by ``detect_changes`` exactly like
        the single-root path."""
        previous_manifest = self._manifest_store.get_latest_manifest(project_id)
        previous_head_sha = (
            previous_manifest.git_metadata.head_commit_sha
            if previous_manifest and previous_manifest.git_metadata.is_git_repo
            else None
        )
        previous_files_by_path = previous_manifest.files if previous_manifest else {}

        display = Path(display_root).resolve() if display_root else None
        git_source = display if display is not None and display.is_dir() else None
        git_meta = git_metadata.get_repository_metadata(git_source or Path(area_roots[AREA_ORDER[0]]))

        all_records: dict[str, FileRecord] = {}
        for area in AREA_ORDER:
            raw_root = area_roots.get(area)
            if not raw_root:
                continue
            area_root = Path(raw_root).resolve()
            if not area_root.is_dir():
                raise NotADirectoryError(
                    f"{area} scan root does not exist or is not a directory: {area_root}"
                )

            blob_info_map = {}
            if git_meta.is_git_repo:
                # Keys are prefixed so git provenance matches the
                # area-prefixed relative paths stored in the manifest.
                blob_info_map = {
                    f"{area}/{rel}": info
                    for rel, info in git_metadata.build_blob_info_map(
                        area_root, previous_head_sha
                    ).items()
                }

            walked_files = list(walk_repository(area_root))
            to_hash: list[WalkedFile] = []
            reused_records: dict[str, FileRecord] = {}
            for wf in walked_files:
                rel = f"{area}/{wf.relative_path}"
                git_info = blob_info_map.get(rel)
                previous_record = previous_files_by_path.get(rel)
                if (
                    previous_record is not None
                    and git_info is not None
                    and previous_record.git_info is not None
                    and previous_record.git_info.blob_sha == git_info.blob_sha
                ):
                    reused_records[rel] = previous_record.model_copy(
                        update={
                            "size_bytes": wf.size_bytes,
                            "mtime": wf.mtime,
                            "git_info": git_info,
                        }
                    )
                else:
                    to_hash.append(wf)

            hashed_records = self._hash_and_build_records(
                project_id, to_hash, blob_info_map, path_prefix=f"{area}/"
            )
            all_records.update(reused_records)
            all_records.update(hashed_records)

        scan_id = str(uuid.uuid4())
        current_manifest = ScanManifest(
            scan_id=scan_id,
            project_id=project_id,
            root_path=str(display) if display is not None else str(Path(area_roots[AREA_ORDER[0]]).resolve()),
            rtl_root=area_roots.get("rtl"),
            tb_root=area_roots.get("tb"),
            docs_root=area_roots.get("docs"),
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            git_metadata=git_meta,
            file_count=len(all_records),
            total_bytes=sum(r.size_bytes for r in all_records.values()),
            files=all_records,
        )

        change_set = detect_changes(previous_manifest, current_manifest)
        self._manifest_store.save_manifest(current_manifest)

        return current_manifest, change_set

    def _hash_and_build_records(
        self,
        project_id: str,
        files: list[WalkedFile],
        blob_info_map: dict,
        path_prefix: str = "",
    ) -> dict[str, FileRecord]:
        records: dict[str, FileRecord] = {}

        def build_one(wf: WalkedFile) -> FileRecord | None:
            try:
                content_hash = hashing.hash_file(wf.absolute_path)
            except (FileNotFoundError, PermissionError):
                return None

            refs: list[str] = []
            if not wf.is_binary and wf.sample_bytes:
                try:
                    text_sample = wf.sample_bytes.decode("utf-8", errors="replace")
                    refs = extract_raw_dependency_refs(wf.category, text_sample)
                except UnicodeDecodeError:
                    refs = []

            rel = f"{path_prefix}{wf.relative_path}"
            return FileRecord(
                project_id=project_id,
                absolute_path=str(wf.absolute_path),
                relative_path=rel,
                size_bytes=wf.size_bytes,
                mtime=wf.mtime,
                content_hash=content_hash,
                hash_algorithm=hashing.algorithm_name(),
                category=wf.category,
                extension=wf.extension,
                is_binary=wf.is_binary,
                git_info=blob_info_map.get(rel),
                raw_dependency_refs=refs,
            )

        if not files:
            return records

        with ThreadPoolExecutor(max_workers=self._max_hash_workers) as pool:
            futures = {pool.submit(build_one, wf): wf for wf in files}
            for future in as_completed(futures):
                record = future.result()
                if record is not None:
                    records[record.relative_path] = record

        return records
