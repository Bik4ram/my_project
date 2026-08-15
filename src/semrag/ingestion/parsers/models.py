"""
Shared models for the parsing pipeline (Part 3).

These are deliberately *not* the full cross-file IR (that's Part 4's job).
A `ParseResult` is the structured output of parsing exactly one file in
isolation — it has no knowledge of other files, no resolved cross-file
relationships, and no notion of "the project". Part 4's IR builder consumes
a stream of `ParseResult`s (one per file, from whichever language-specific
parser handled it) plus the scanner's `raw_dependency_refs` hints and
produces the linked graph (module instantiates module, assertion covers
register, etc.).

Keeping this boundary strict matters for incremental re-indexing: when one
file changes, only that file needs to be re-parsed into a new ParseResult;
Part 4 only needs to re-link the edges that touch that file, not re-parse
the world.
"""
from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class DiagnosticSeverity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ParseDiagnostic(BaseModel):
    severity: DiagnosticSeverity
    message: str
    line: int | None = None
    column: int | None = None


class ParseResult(BaseModel):
    """Generic envelope every language parser returns. `payload` holds the
    language-specific structured data (e.g. a `SystemVerilogFileAST` for
    the SV parser, a `PythonModuleAST` for the Python parser) — kept as
    `dict[str, Any]` here rather than a union type so ParseResult itself
    never needs to change when a new language parser is added; only the
    payload's shape (defined in that parser's own `models.py`) does."""
    project_id: str
    relative_path: str
    parser_name: str
    success: bool
    diagnostics: list[ParseDiagnostic] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)
