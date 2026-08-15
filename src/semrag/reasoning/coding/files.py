"""Safe project-relative file operations for the coding agent.

Reuses the exact invariants enforced by the Phase-1 write surface
(``projects.py``): project-relative paths only, no absolute paths, no
``..`` traversal, no symlink escapes, UTF-8 text only, atomic writes,
and patches that match exactly once. Instead of HTTP errors these raise
``ProjectFileError`` so the orchestrator can turn them into a safe stop.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

MAX_WRITE_BYTES = 2 * 1024 * 1024  # 2 MiB cap, mirrors projects.py


class ProjectFileError(Exception):
    """A file operation was rejected or failed. ``kind`` groups the
    failure (path, not_found, encoding, size, mismatch, io)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def safe_resolve_target(root: Path, file_path: str) -> Path:
    """Resolve a project-relative path inside ``root`` or raise."""
    if not file_path or not file_path.strip():
        raise ProjectFileError("path", "file_path must not be empty.")
    if "\x00" in file_path:
        raise ProjectFileError("path", "file_path must not contain NUL bytes.")

    raw = Path(file_path)
    if raw.is_absolute():
        raise ProjectFileError(
            "path",
            f"file_path must be relative to the project root, got absolute path: {file_path}",
        )
    if any(part == ".." for part in raw.parts):
        raise ProjectFileError(
            "path",
            f"file_path must not contain '..' traversal: {file_path}",
        )

    target = (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ProjectFileError(
            "path",
            f"Resolved file path escapes project root: {file_path}",
        ) from None
    return target


def _atomic_write(target: Path, content: str, file_path: str) -> None:
    """Stage a sibling temp file then ``os.replace`` so a crash cannot
    leave a half-written file at the target path."""
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_WRITE_BYTES:
        raise ProjectFileError(
            "size",
            f"Content too large to write in one operation "
            f"({len(content_bytes)} bytes > {MAX_WRITE_BYTES})",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        raise ProjectFileError("io", f"Failed to stage write for {file_path}: {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content_bytes)
        os.replace(tmp_path, target)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ProjectFileError("io", f"Failed to write {file_path}: {exc}") from exc


def write_file(root: Path, file_path: str, content: str) -> dict:
    """Create or replace a project file. Returns a small summary dict."""
    target = safe_resolve_target(root, file_path)
    existed = target.exists()
    _atomic_write(target, content, file_path)
    lines_written = content.count("\n") + (
        0 if content.endswith("\n") or content == "" else 1
    )
    return {
        "relative_path": file_path,
        "created": not existed,
        "existed": existed,
        "bytes_written": len(content.encode("utf-8")),
        "lines_written": lines_written,
        "requires_reindex": True,
    }


def read_file(root: Path, file_path: str, start: int | None = None, end: int | None = None) -> str:
    """Read a project file as UTF-8 text, optionally a line range
    (1-indexed inclusive)."""
    target = safe_resolve_target(root, file_path)
    if not target.is_file():
        raise ProjectFileError("not_found", f"File does not exist: {file_path}")
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectFileError("encoding", f"File is not valid UTF-8 text: {file_path}") from exc
    except OSError as exc:
        raise ProjectFileError("io", f"Failed to read {file_path}: {exc}") from exc
    if start is not None or end is not None:
        lines = text.splitlines(keepends=True)
        start = start or 1
        end = end or len(lines)
        return "".join(lines[max(start - 1, 0):end])
    return text


def file_exists(root: Path, file_path: str) -> bool:
    try:
        return safe_resolve_target(root, file_path).is_file()
    except ProjectFileError:
        return False


def patch_file(root: Path, file_path: str, old_text: str, new_text: str) -> dict:
    """Apply a single exact-text replacement that must match exactly once."""
    target = safe_resolve_target(root, file_path)
    if not target.is_file():
        raise ProjectFileError("not_found", f"File does not exist: {file_path}")
    try:
        current = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectFileError("encoding", f"File is not valid UTF-8 text: {file_path}") from exc
    except OSError as exc:
        raise ProjectFileError("io", f"Failed to read {file_path}: {exc}") from exc

    match_count = current.count(old_text)
    if match_count == 0:
        raise ProjectFileError(
            "mismatch",
            f"old_text not found in {file_path} (0 matches); no change applied",
        )
    if match_count > 1:
        raise ProjectFileError(
            "mismatch",
            f"old_text found {match_count} times in {file_path}; refusing ambiguous edit",
        )

    updated = current.replace(old_text, new_text)
    _atomic_write(target, updated, file_path)
    return {
        "relative_path": file_path,
        "replaced_count": 1,
        "bytes_written": len(updated.encode("utf-8")),
        "requires_reindex": True,
    }


def snapshot_paths(root: Path) -> set[str]:
    """Relative paths of every file currently under the project root
    (used to prove unrelated files were not touched)."""
    paths: set[str] = set()
    if not root.is_dir():
        return paths
    for dirpath, _dirnames, filenames in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        for name in filenames:
            full = Path(dirpath) / name
            try:
                rel = full.relative_to(root)
            except ValueError:
                continue
            paths.add(rel.as_posix())
    return paths