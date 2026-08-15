"""
Change detection between consecutive scans.

The core operation is a three-way set comparison between the previous
manifest's file set and the current scan's file set, keyed by
relative_path — the same shape as a standard dict-diff, specialized with
one important optimization: when git metadata is available, a file whose
git *blob SHA* is unchanged is treated as unchanged without inspecting our
own content_hash at all. Git already did the content-comparison work when
it computed that blob SHA; re-deriving the same answer via blake3 would be
redundant I/O. blake3 content_hash is still the source of truth for
untracked files (git has no blob SHA for those) and as a defense-in-depth
cross-check available for the rare case blob-SHA bookkeeping is ever
suspect (e.g. a manifest imported from a different clone of the repo with
different git internals).
"""
from __future__ import annotations

from semrag.ingestion.scanner.models import ChangeSet, FileRecord, ScanManifest


def detect_changes(
    previous: ScanManifest | None,
    current: ScanManifest,
) -> ChangeSet:
    if previous is None:
        # First-ever scan of this project: everything is "added".
        return ChangeSet(
            project_id=current.project_id,
            previous_scan_id=None,
            current_scan_id=current.scan_id,
            added=list(current.files.values()),
            modified=[],
            deleted_paths=[],
            unchanged_count=0,
        )

    added: list[FileRecord] = []
    modified: list[FileRecord] = []
    unchanged_count = 0

    for rel_path, current_file in current.files.items():
        previous_file = previous.files.get(rel_path)
        if previous_file is None:
            added.append(current_file)
            continue

        if _is_unchanged(previous_file, current_file):
            unchanged_count += 1
        else:
            modified.append(current_file)

    deleted_paths = [
        rel_path for rel_path in previous.files if rel_path not in current.files
    ]

    return ChangeSet(
        project_id=current.project_id,
        previous_scan_id=previous.scan_id,
        current_scan_id=current.scan_id,
        added=added,
        modified=modified,
        deleted_paths=deleted_paths,
        unchanged_count=unchanged_count,
    )


def _is_unchanged(previous: FileRecord, current: FileRecord) -> bool:
    # Fast path: both sides have git blob info and it matches.
    if previous.git_info and current.git_info:
        if previous.git_info.blob_sha == current.git_info.blob_sha:
            return True
        return False

    # Fallback: compare our own content hash directly (untracked files,
    # or a project scanned outside a git context entirely).
    return previous.content_hash == current.content_hash
