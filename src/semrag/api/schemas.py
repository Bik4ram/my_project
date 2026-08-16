"""API request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from semrag.reasoning.prompting.models import ChatMessage
from semrag.retrieval.models import RetrievalResult
from typing import List, Dict, Any


class IndexProjectRequest(BaseModel):
    """Registration request for a project to index.

    Two shapes, both backward compatible:

    * Legacy single-root: only ``root_path`` (absolute path to the project
      root). Multi-root fields all ``None``.
    * Multi-root / portable: at least one of ``rtl_root`` / ``tb_root`` /
      ``docs_root`` is set. ``root_path`` is optional here and used only as
      a display/fallback root (e.g. as the validation working directory);
      file resolution always goes through the area roots.
    """

    root_path: str | None = Field(
        default=None,
        description="Absolute path to the legacy project root (required for single-root registration).",
    )
    rtl_root: str | None = Field(
        default=None,
        description="Absolute path to the RTL source area root.",
    )
    tb_root: str | None = Field(
        default=None,
        description="Absolute path to the testbench/UVM area root.",
    )
    docs_root: str | None = Field(
        default=None,
        description="Absolute path to the documentation area root (read-only by default).",
    )


class IndexProjectResponse(BaseModel):
    project_id: str
    job_id: str
    file_count: int
    chunk_count: int
    scan_id: str


class FileSummary(BaseModel):
    relative_path: str
    category: str
    size_bytes: int


class ListFilesResponse(BaseModel):
    project_id: str
    files: list[FileSummary]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    enable_graph_expansion: bool = True
    enable_reranking: bool = True
    enable_intent_boost: bool = True


class QueryResponse(BaseModel):
    query: str
    messages: list[ChatMessage]
    included_chunks: list[RetrievalResult]
    excluded_chunk_count: int
    estimated_total_tokens: int
    from_cache: bool = False


class WriteFileRequest(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Project-relative path. Must be inside the project's root. No absolute paths, no '..' traversal.",
    )
    content: str = Field(
        ...,
        description="UTF-8 text content to write. Subject to max_write_file_bytes on the server.",
    )


class WriteFileResponse(BaseModel):
    project_id: str
    relative_path: str
    bytes_written: int
    lines_written: int
    created: bool
    requires_reindex: bool


class PatchFileRequest(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Project-relative path. Must be inside the project's root. No absolute paths, no '..' traversal.",
    )
    old_text: str = Field(
        ...,
        min_length=1,
        description="Exact text to find in the current file content. Must match exactly once.",
    )
    new_text: str = Field(
        ...,
        description="UTF-8 replacement text for the single matched old_text.",
    )


class PatchFileResponse(BaseModel):
    project_id: str
    relative_path: str
    replacements: int
    bytes_written: int
    requires_reindex: bool


class DeleteFileRequest(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Project-relative path. Must be inside the project's writable root. No absolute paths, no '..' traversal.",
    )


class DeleteFileResponse(BaseModel):
    project_id: str
    relative_path: str
    deleted: bool
    requires_reindex: bool


class CitationSummary(BaseModel):
    valid_count: int
    fabricated_count: int
    fabrication_rate: float
    cited_paths: List[str]


class GroundingSummary(BaseModel):
    """A1 grounding-gate result: whether the answer's source-specific
    claims were verified against the retrieved evidence."""

    status: str  # grounded | revised | insufficient_evidence | unsupported
    unsupported_claims: list[str] = []
    revision_attempted: bool = False
    revised: bool = False


class AnswerResponse(BaseModel):
    answer: str
    citations: CitationSummary
    sources: List[RetrievalResult]
    iterations: int
    tools_used: List[str]
    retrieval: Dict[str, Any]
    timings: Dict[str, Any]
    metadata: Dict[str, Any]
    grounding: GroundingSummary | None = None


class CodeRequest(BaseModel):
    """A2 — request for the grounded coding agent workflow."""

    request: str = Field(..., min_length=1, description="Natural-language coding request.")
    max_repair_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Bounded diagnose → repair → validate iterations (spec: max 3).",
    )
    validation_target: str | None = Field(
        default=None,
        description="Validation target to run after changes (defaults to the project default).",
    )


class CodeChangeSummary(BaseModel):
    op: str
    file_path: str
    applied: bool
    verified: bool
    error: str | None = None


class CodeValidationSummary(BaseModel):
    target: str
    status: str
    exit_code: int | None = None
    command: str = ""
    blocked_reason: str | None = None
    duration_seconds: float = 0.0


class CodeResponse(BaseModel):
    """A2 — grounded coding agent result."""

    project_id: str
    intent: str | None = None
    status: str  # completed | blocked | failed | insufficient_evidence | needs_clarification
    final_answer: str
    grounding: GroundingSummary | None = None
    changes: list[CodeChangeSummary] = []
    validation: list[CodeValidationSummary] = []
    repair_attempts: int = 0
    files_inspected: list[str] = []
    files_modified: list[str] = []
    validation_target: str = "rtl_lint"
    needs_clarification: str | None = None
    error: str | None = None
    request_id: str = ""