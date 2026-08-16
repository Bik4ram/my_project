"""
Data models for the Project Scanner subsystem.

These are the objects the scanner produces. They are intentionally *not* the
same as the IR objects built in Part 4 — the scanner knows nothing about
SystemVerilog modules or UVM sequences. It only knows about files, their
identity (hash), their git provenance, and coarse structural hints (raw
include/import strings) cheap enough to extract without a real parser.

Everything here is a pydantic model so it serializes cleanly to/from
PostgreSQL JSONB columns (Part 6) and across the Celery task boundary
(Part 2 workers) without a separate DTO layer.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FileCategory(str, enum.Enum):
    """Coarse classification used for downstream routing (which parser
    subpackage in Part 3 should claim this file). Kept coarse deliberately —
    fine-grained dialect detection (SV vs VHDL vs mixed) happens in the
    parser, not here, because that requires actually opening the file.
    """
    RTL_VERILOG = "rtl_verilog"
    RTL_VHDL = "rtl_vhdl"
    TESTBENCH_UVM = "testbench_uvm"
    ASSERTION = "assertion"
    PYTHON = "python"
    TCL = "tcl"
    SHELL = "shell"
    MARKDOWN = "markdown"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    LOG = "log"
    COVERAGE_REPORT = "coverage_report"
    TIMING_REPORT = "timing_report"
    LINT_REPORT = "lint_report"
    WAVEFORM = "waveform"
    BUILD_CONFIG = "build_config"
    GIT_INTERNAL = "git_internal"
    BINARY_OTHER = "binary_other"
    UNKNOWN = "unknown"


class GitBlobInfo(BaseModel):
    """Per-file git provenance, extracted in bulk via `git ls-tree` rather
    than one subprocess call per file (see git_metadata.py for why that
    distinction matters at million-file scale)."""
    blob_sha: str
    last_commit_sha: Optional[str] = None
    last_commit_author: Optional[str] = None
    last_commit_date: Optional[datetime] = None
    is_tracked: bool = True


class FileRecord(BaseModel):
    """One row per file, as of one scan. This is the atomic unit the
    ChangeDetector diffs between scans."""
    project_id: str
    absolute_path: str
    relative_path: str
    size_bytes: int
    mtime: datetime
    content_hash: str = Field(..., description="blake3 hex digest of file bytes")
    hash_algorithm: str = "blake3"
    category: FileCategory
    extension: str
    is_binary: bool
    git_info: Optional[GitBlobInfo] = None
    raw_dependency_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Cheap, unresolved dependency strings pulled via regex during "
            "the walk (e.g. `include \"foo.svh\"`, `import bar`). Resolution "
            "into real IR edges happens in Part 4, not here."
        ),
    )


class GitRepositoryMetadata(BaseModel):
    """Repository-level (not file-level) git state, captured once per scan."""
    is_git_repo: bool
    current_branch: Optional[str] = None
    head_commit_sha: Optional[str] = None
    remote_url: Optional[str] = None
    is_dirty: bool = False
    untracked_file_count: int = 0


class ScanManifest(BaseModel):
    """Summary record of one completed scan. Persisted so the *next* scan
    can diff against it without re-hashing unchanged files."""
    scan_id: str
    project_id: str
    root_path: str
    # Optional authoritative area roots (multi-root / portable projects).
    # For a multi-root scan these are the roots each area-prefixed
    # ``relative_path`` maps back to; for a legacy single-root scan all
    # three stay ``None`` and ``root_path`` alone carries the layout.
    rtl_root: Optional[str] = None
    tb_root: Optional[str] = None
    docs_root: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    git_metadata: GitRepositoryMetadata
    file_count: int = 0
    total_bytes: int = 0
    files: dict[str, FileRecord] = Field(
        default_factory=dict,
        description="Keyed by relative_path for O(1) diffing.",
    )


class ChangeSet(BaseModel):
    """Output of ChangeDetector: what changed between two manifests."""
    project_id: str
    previous_scan_id: Optional[str]
    current_scan_id: str
    added: list[FileRecord] = Field(default_factory=list)
    modified: list[FileRecord] = Field(default_factory=list)
    deleted_paths: list[str] = Field(default_factory=list)
    unchanged_count: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted_paths)

    @property
    def total_changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted_paths)
