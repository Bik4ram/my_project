"""
TCL parser.

Deliberately NOT a full TCL grammar parser. TCL's actual grammar is
notoriously context-dependent (famously, "everything is a string" and
command behavior can be redefined at runtime), and no mature, maintained
Python TCL-grammar library exists that would justify the investment for
this system's purposes. EDA build/regression scripts almost always use TCL
in a narrow, disciplined subset (`proc` definitions, `source` calls,
variable sets), so a line-oriented regex extractor captures the signal
that actually matters for retrieval and dependency tracking — "what
procedures does this script define, what other scripts does it pull in" —
at a fraction of the engineering cost of a real parser.

This is the same category of trade-off as the scanner's
`dependency_tracker.py` (Part 2): explicitly a heuristic hint layer, not a
system of record. It will over-match inside multi-line strings/comments
and under-match dynamically constructed `source` paths (e.g. `source
[file join $dir "setup.tcl"]`). Flagged here rather than silently accepted
so a future revision can decide if TCL usage in your specific regression
scripts is disciplined enough for this to be sufficient long-term, or if
it warrants a real parser investment later.
"""
from __future__ import annotations

import re
from pathlib import Path

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.parsers.python_tcl.tcl_models import (
    ParsedProc,
    ParsedSourceRef,
    TclFileStructure,
)
from semrag.ingestion.scanner.models import FileRecord

_PROC_PATTERN = re.compile(r'^\s*proc\s+([\w:]+)\s*\{([^}]*)\}', re.MULTILINE)
_SOURCE_PATTERN = re.compile(r'^\s*source\s+(\S+)', re.MULTILINE)


class TclParser:
    name = "tcl_regex_structural"

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

        procs: list[ParsedProc] = []
        for match in _PROC_PATTERN.finditer(text):
            name = match.group(1)
            raw_args = match.group(2).strip()
            args = [a.strip().split()[0] for a in raw_args.split() if a.strip()] if raw_args else []
            line = text.count("\n", 0, match.start()) + 1
            procs.append(ParsedProc(name=name, line=line, args=args))

        source_refs: list[ParsedSourceRef] = []
        for match in _SOURCE_PATTERN.finditer(text):
            path = match.group(1).strip()
            line = text.count("\n", 0, match.start()) + 1
            source_refs.append(ParsedSourceRef(path=path, line=line))

        structure = TclFileStructure(procs=procs, source_refs=source_refs)

        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )
