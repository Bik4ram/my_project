"""Data model for the A2 grounded coding agent workflow.

Everything the orchestrator produces is captured in plain dataclasses so
tests can assert on the exact state transitions and the API / MCP layer
can serialize it without touching the workflow logic.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CodingIntent(str, enum.Enum):
    """User-intent classification for a coding request."""

    INFORMATION = "information"
    CODE_CHANGE = "code_change"
    DEBUG = "debug"
    REVIEW = "review"
    AMBIGUOUS = "ambiguous"


class CodingPhase(str, enum.Enum):
    """The state machine phases a coding request moves through."""

    INTAKE = "intake"
    RETRIEVE = "retrieve"
    INSPECT = "inspect"
    PLAN = "plan"
    MODIFY = "modify"
    VALIDATE = "validate"
    VERIFY = "verify"
    DIAGNOSE = "diagnose"
    REPAIR = "repair"
    FINAL = "final"
    STOPPED = "stopped"


class ChangeOp(str, enum.Enum):
    """The two allowed file-change operations."""

    PATCH = "patch"
    WRITE = "write"


@dataclass(frozen=True)
class PlannedChange:
    """A change the planner declared. ``old_text`` is required for a
    patch and must be absent for a write of a new file."""

    op: ChangeOp
    file_path: str
    new_text: str
    old_text: str | None = None
    reason: str = ""


@dataclass
class FileChange:
    """The deterministic application record of one planned change."""

    op: ChangeOp
    file_path: str
    applied: bool = False
    verified: bool = False
    requires_reindex: bool = True
    error: str | None = None
    expected_text: str | None = None


@dataclass
class ValidationAttempt:
    """One validation run, recorded exactly as the runner reported it."""

    target: str
    status: str  # passed | failed | blocked
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    blocked_reason: str | None = None
    duration_seconds: float = 0.0


@dataclass
class CodingState:
    """Mutable state threaded through the workflow (test-observable)."""

    request_text: str
    intent: CodingIntent | None = None
    phase: CodingPhase = CodingPhase.INTAKE
    files_inspected: list[str] = field(default_factory=list)
    files_to_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    change_surface: list[str] = field(default_factory=list)
    new_file_path: str = ""
    changes: list[FileChange] = field(default_factory=list)
    validation_attempts: list[ValidationAttempt] = field(default_factory=list)
    repair_attempts: int = 0
    validation_target: str = "rtl_lint"
    note: str = ""
    snapshot_before: set[str] = field(default_factory=set)


@dataclass
class CodingRequest:
    project_id: str
    request: str
    max_repair_attempts: int = 3
    validation_target: str | None = None


@dataclass
class CodingResult:
    """Outcome of a coding workflow run.

    ``final_status`` values: ``completed``, ``blocked``, ``failed``,
    ``insufficient_evidence``, ``needs_clarification``.
    """

    request: CodingRequest
    intent: CodingIntent | None = None
    final_status: str = "failed"
    final_answer: str = ""
    grounding: dict | None = None
    changes: list[FileChange] = field(default_factory=list)
    validation_attempts: list[ValidationAttempt] = field(default_factory=list)
    repair_attempts: int = 0
    files_inspected: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    change_surface: list[str] = field(default_factory=list)
    validation_target: str = "rtl_lint"
    needs_clarification: str | None = None
    error: str | None = None