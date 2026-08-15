"""
Regression/simulation log parser.

Deliberately generic, and deliberately scoped to only `FileCategory.LOG`
(plain regression run logs), not `COVERAGE_REPORT`/`TIMING_REPORT`/
`LINT_REPORT` — those are real formats (VCS URG, Verdi, PrimeTime STA
reports, Spyglass/lint tool output) that differ meaningfully by vendor and
tool version, and parsing them well requires representative sample files
from an actual toolchain to validate against, which this delivery doesn't
have. Building a parser against assumed formats without real samples to
check it against would produce exactly the kind of shallow, unverified
code Part 1 explicitly ruled out — so those three categories are left
unregistered here rather than given a parser that looks complete but was
never validated against real tool output. Flagged in this part's design
doc, not silently skipped.

What IS generic across nearly every DV regression flow (UVM or otherwise)
is the PASS/FAIL-per-test-run + UVM severity-count pattern this parser
extracts — common enough across VCS/Xcelium/Questa regression wrapper
output to be worth capturing heuristically.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_RESULT_PATTERN = re.compile(r"\b(PASS(?:ED)?|FAIL(?:ED)?)\b", re.IGNORECASE)
_TEST_NAME_PATTERN = re.compile(r"(?:TEST(?:NAME)?|Test)\s*[:=]\s*(\S+)", re.IGNORECASE)
_SEED_PATTERN = re.compile(r"SEED\s*[:=]\s*(\d+)", re.IGNORECASE)
_UVM_SEVERITY_PATTERN = re.compile(r"\bUVM_(INFO|WARNING|ERROR|FATAL)\b")


class RegressionLogSummary(BaseModel):
    result: str | None = Field(default=None, description="'PASS' | 'FAIL' | None if undetermined")
    test_name: str | None = None
    seed: int | None = None
    uvm_info_count: int = 0
    uvm_warning_count: int = 0
    uvm_error_count: int = 0
    uvm_fatal_count: int = 0
    line_count: int = 0


class RegressionLogParser:
    name = "regression_log_heuristic"

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

        severity_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "FATAL": 0}
        for m in _UVM_SEVERITY_PATTERN.finditer(text):
            severity_counts[m.group(1)] += 1

        # Result: last PASS/FAIL token in the file wins — regression logs
        # commonly print intermediate status before a final summary line.
        result = None
        for m in _RESULT_PATTERN.finditer(text):
            token = m.group(1).upper()
            result = "PASS" if token.startswith("PASS") else "FAIL"

        test_name_match = _TEST_NAME_PATTERN.search(text)
        seed_match = _SEED_PATTERN.search(text)

        summary = RegressionLogSummary(
            result=result,
            test_name=test_name_match.group(1) if test_name_match else None,
            seed=int(seed_match.group(1)) if seed_match else None,
            uvm_info_count=severity_counts["INFO"],
            uvm_warning_count=severity_counts["WARNING"],
            uvm_error_count=severity_counts["ERROR"],
            uvm_fatal_count=severity_counts["FATAL"],
            line_count=text.count("\n") + 1,
        )

        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=summary.model_dump(),
        )
