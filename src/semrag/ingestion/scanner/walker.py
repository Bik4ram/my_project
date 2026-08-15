"""
Repository filesystem walker.

Two perf-critical decisions here, both aimed at the same problem: a
semiconductor project directory contains huge subtrees (`Waveforms/`,
`Build/`, `.git/`, simulation scratch dirs) that must never be descended
into during a routine scan.

1. Directory pruning happens *during* the walk (`os.walk` with in-place
   `dirnames` mutation), not by walking everything and filtering
   afterward. Filtering after the fact still pays the filesystem-stat cost
   of every file under a multi-hundred-GB waveform directory; pruning
   during the walk skips descending into it at all.

2. `.gitignore`-style pattern matching (via `pathspec`, the same library
   git itself effectively re-implements semantics for) is combined with an
   explicit always-exclude list, because relying on `.gitignore` alone is
   not safe here — some semiconductor project waveform/build directories
   are *intentionally* git-tracked (checked-in golden waveforms, for
   example) but still must not be walked/hashed/embedded as if they were
   source text.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pathspec

from semrag.ingestion.scanner.classification import (
    classify,
    is_binary_extension,
    looks_like_binary_content,
)
from semrag.ingestion.scanner.models import FileCategory

# Directories pruned unconditionally, regardless of .gitignore contents.
# These are large-binary / non-source-of-truth directories by convention
# in the project layout described in Part 1 Section 2 ("Data").
ALWAYS_EXCLUDED_DIRS = {
    ".git",
    "Waveforms",
    "waveforms",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
}

# File size above which content is treated as opaque for the purposes of
# dependency-ref scanning (still hashed and recorded, just not read for
# regex scanning) — protects the walker from reading multi-GB report files
# line-by-line looking for `import` statements that won't be there anyway.
MAX_CONTENT_SCAN_BYTES = 5 * 1024 * 1024  # 5MB


@dataclass
class WalkedFile:
    absolute_path: Path
    relative_path: str  # posix-style, relative to repo root
    size_bytes: int
    mtime: datetime
    category: FileCategory
    extension: str
    is_binary: bool
    sample_bytes: bytes  # first 8KB, used for dependency-ref scanning downstream


def _load_gitignore_spec(repo_root: Path) -> pathspec.PathSpec:
    gitignore_path = repo_root / ".gitignore"
    lines: list[str] = []
    if gitignore_path.exists():
        lines = gitignore_path.read_text(errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def walk_repository(repo_root: Path) -> Iterator[WalkedFile]:
    """Lazily yields every file the scanner should consider, applying
    directory pruning and .gitignore exclusion as it goes. A generator, not
    a list — callers (orchestrator.py) must not materialize this into a
    list for large repos; it's consumed incrementally so peak memory stays
    bounded regardless of repo size."""
    spec = _load_gitignore_spec(repo_root)

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune in place: os.walk respects mutations to dirnames and will
        # not descend into removed entries. This is what makes pruning
        # cheaper than post-hoc filtering.
        dirnames[:] = [d for d in dirnames if d not in ALWAYS_EXCLUDED_DIRS]

        dir_rel = os.path.relpath(dirpath, repo_root)
        dirnames[:] = [
            d for d in dirnames
            if not spec.match_file(
                os.path.join(dir_rel, d) if dir_rel != "." else d
            )
        ]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = os.path.relpath(abs_path, repo_root).replace(os.sep, "/")

            if spec.match_file(rel_path):
                continue

            try:
                stat = abs_path.stat()
            except (FileNotFoundError, PermissionError):
                # File vanished mid-walk (common in active build dirs) or is
                # unreadable — skip rather than fail the whole scan.
                continue

            category = classify(rel_path)
            ext = abs_path.suffix.lower()
            is_binary = is_binary_extension(rel_path)
            sample = b""

            if not is_binary and stat.st_size <= MAX_CONTENT_SCAN_BYTES:
                try:
                    with abs_path.open("rb") as f:
                        sample = f.read(8192)
                    if looks_like_binary_content(sample):
                        is_binary = True
                except (FileNotFoundError, PermissionError):
                    continue

            yield WalkedFile(
                absolute_path=abs_path,
                relative_path=rel_path,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                category=category,
                extension=ext,
                is_binary=is_binary,
                sample_bytes=sample,
            )
