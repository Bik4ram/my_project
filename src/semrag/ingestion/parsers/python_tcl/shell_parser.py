"""
Shell script parser.

Same regex-structural category as TCL: shell function definitions and
`source`/`.`-sourced scripts are the two facts worth capturing cheaply for
dependency-graph seeding. Full shell parsing (quoting rules, variable
expansion, subshells) is not attempted — EDA regression wrapper scripts
are almost always simple sequential command lists, not shell programs
exercising the language's more complex features, so this heuristic level
matches the actual usage pattern.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_FUNCTION_PATTERN = re.compile(
    r"^\s*(?:function\s+)?([\w-]+)\s*\(\)\s*\{", re.MULTILINE
)
_SOURCE_PATTERN = re.compile(r"^\s*(?:source|\.)\s+(\S+)", re.MULTILINE)


class ParsedShellFunction(BaseModel):
    name: str
    line: int


class ParsedShellSourceRef(BaseModel):
    path: str
    line: int


class ShellFileStructure(BaseModel):
    functions: list[ParsedShellFunction] = Field(default_factory=list)
    source_refs: list[ParsedShellSourceRef] = Field(default_factory=list)


class ShellParser:
    name = "shell_regex_structural"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        try:
            text = absolute_path.read_text(errors="replace")
        except (FileNotFoundError, PermissionError) as exc:
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=f"Could not read file: {exc}")
                ],
            )

        functions = [
            ParsedShellFunction(name=m.group(1), line=text.count("\n", 0, m.start()) + 1)
            for m in _FUNCTION_PATTERN.finditer(text)
        ]
        source_refs = [
            ParsedShellSourceRef(path=m.group(1), line=text.count("\n", 0, m.start()) + 1)
            for m in _SOURCE_PATTERN.finditer(text)
        ]

        structure = ShellFileStructure(functions=functions, source_refs=source_refs)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )
