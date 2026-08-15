"""
Excel parser.

Uses openpyxl (read-only mode for memory efficiency on large workbooks —
register maps and test plan spreadsheets in real projects run to
thousands of rows). Extracts per-sheet shape (header row, dimensions) and
a bounded sample of data rows, on the same "shape + sample, not full
duplication" principle as the CSV parser — full row content becomes
retrievable chunks in Part 8, not re-embedded whole into the IR payload.

Header row detection is a heuristic: the first non-empty row of each
sheet is assumed to be headers. This is correct for the overwhelming
majority of register-map/test-list spreadsheets (which are built as
single flat tables) and wrong for spreadsheets with merged title rows,
multiple stacked tables, or dashboards with charts — those are flagged as
a known gap rather than silently mishandled (see this parser's diagnostics
for a `no_data_rows_found` warning when a sheet's shape doesn't fit the
assumption).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_MAX_SAMPLE_ROWS = 10


class ParsedSheet(BaseModel):
    name: str
    row_count: int
    column_count: int
    headers: list[str] = Field(default_factory=list)
    sample_rows: list[list[str]] = Field(default_factory=list)


class ExcelFileStructure(BaseModel):
    sheets: list[ParsedSheet] = Field(default_factory=list)


class ExcelParser:
    name = "openpyxl"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        diagnostics: list[ParseDiagnostic] = []
        try:
            workbook = openpyxl.load_workbook(absolute_path, read_only=True, data_only=True)
        except Exception as exc:  # noqa: BLE001 - openpyxl raises assorted exceptions on corrupt/unsupported files
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=f"Failed to open workbook: {exc}")
                ],
            )

        sheets: list[ParsedSheet] = []
        try:
            for sheet_name in workbook.sheetnames:
                ws = workbook[sheet_name]
                rows_iter = ws.iter_rows(values_only=True)

                headers: list[str] = []
                sample_rows: list[list[str]] = []
                row_count = 0

                for row in rows_iter:
                    if row_count == 0:
                        if all(cell is None for cell in row):
                            row_count += 1
                            continue
                        headers = [str(c) if c is not None else "" for c in row]
                    elif len(sample_rows) < _MAX_SAMPLE_ROWS:
                        sample_rows.append([str(c) if c is not None else "" for c in row])
                    row_count += 1

                if not headers:
                    diagnostics.append(
                        ParseDiagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            message=f"no_data_rows_found in sheet '{sheet_name}'",
                        )
                    )

                sheets.append(
                    ParsedSheet(
                        name=sheet_name,
                        row_count=max(row_count - 1, 0),
                        column_count=len(headers),
                        headers=headers,
                        sample_rows=sample_rows,
                    )
                )
        finally:
            workbook.close()

        structure = ExcelFileStructure(sheets=sheets)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            diagnostics=diagnostics,
            payload=structure.model_dump(),
        )
