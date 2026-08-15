"""
PDF parser.

Uses pdfplumber for both text and table extraction — the two things
semiconductor spec/register-map PDFs actually contain that matter here:
prose (requirements, protocol descriptions) and tables (register maps,
pin lists, timing tables). Both are extracted per-page, so Part 8's
chunker can chunk along page boundaries and Part 6 can preserve a
page-number citation for grounding (a RAG answer citing "register map,
page 14" is meaningfully more useful to an engineer than an unattributed
excerpt).

Explicitly NOT implemented: OCR for scanned/image-only PDFs. pdfplumber
extracts embedded text layers only — a scanned datasheet with no text
layer will yield empty page text here. OCR (e.g. via pytesseract +
poppler) is a real, heavier dependency (external binary, not a pure
Python wheel) and is deferred rather than silently attempted badly;
`page_text_empty` diagnostics are emitted so a future OCR pass has a
worklist of exactly which pages/files need it, instead of needing to
re-scan everything to find out.
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber
from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_MAX_ROWS_PER_TABLE = 200


class ParsedPdfTable(BaseModel):
    page_number: int
    row_count: int
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[list[str | None]] = Field(default_factory=list)


class ParsedPdfPage(BaseModel):
    page_number: int
    text: str
    char_count: int


class PdfFileStructure(BaseModel):
    page_count: int
    pages: list[ParsedPdfPage] = Field(default_factory=list)
    tables: list[ParsedPdfTable] = Field(default_factory=list)


class PdfParser:
    name = "pdfplumber"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        diagnostics: list[ParseDiagnostic] = []
        try:
            with pdfplumber.open(absolute_path) as pdf:
                pages: list[ParsedPdfPage] = []
                tables: list[ParsedPdfTable] = []

                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append(ParsedPdfPage(page_number=i, text=text, char_count=len(text)))
                    if not text.strip():
                        diagnostics.append(
                            ParseDiagnostic(
                                severity=DiagnosticSeverity.WARNING,
                                message="page_text_empty (likely scanned/image-only page; OCR not implemented)",
                                line=i,
                            )
                        )

                    for raw_table in page.extract_tables():
                        if not raw_table:
                            continue
                        header, *rows = raw_table
                        tables.append(
                            ParsedPdfTable(
                                page_number=i,
                                row_count=len(rows),
                                columns=[c or "" for c in header],
                                sample_rows=rows[:_MAX_ROWS_PER_TABLE],
                            )
                        )

                structure = PdfFileStructure(page_count=len(pdf.pages), pages=pages, tables=tables)
        except Exception as exc:  # noqa: BLE001 - pdfplumber/pdfminer raise assorted exceptions on malformed PDFs
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=f"Failed to open/parse PDF: {exc}")
                ],
            )

        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            diagnostics=diagnostics,
            payload=structure.model_dump(),
        )
