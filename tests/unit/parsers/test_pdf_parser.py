from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from semrag.ingestion.parsers.documents.pdf_parser import PdfParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=path.name,
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=FileCategory.PDF,
        extension=".pdf",
        is_binary=True,
    )


def _build_fixture_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    table = Table(
        [
            ["Address", "Register Name", "Width"],
            ["0x00", "CTRL", "8"],
            ["0x04", "STATUS", "8"],
        ]
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    story = [
        Paragraph("I2C Master Register Map Specification", styles["Title"]),
        Paragraph("This document describes the register map for the I2C master IP block.", styles["Normal"]),
        table,
    ]
    doc.build(story)


def test_pdf_extracts_text_and_tables(tmp_path: Path):
    pdf_path = tmp_path / "i2c_spec.pdf"
    _build_fixture_pdf(pdf_path)

    result = PdfParser().parse(pdf_path, _record(pdf_path))
    assert result.success is True
    assert result.payload["page_count"] == 1

    page_text = result.payload["pages"][0]["text"]
    assert "I2C Master Register Map Specification" in page_text

    tables = result.payload["tables"]
    assert len(tables) == 1
    assert tables[0]["columns"] == ["Address", "Register Name", "Width"]
    assert ["0x00", "CTRL", "8"] in tables[0]["sample_rows"]


def test_pdf_invalid_file_fails_gracefully(tmp_path: Path):
    fake_pdf = tmp_path / "not_really_a_pdf.pdf"
    fake_pdf.write_bytes(b"this is not a pdf file at all")

    result = PdfParser().parse(fake_pdf, _record(fake_pdf))
    assert result.success is False
    assert result.has_errors is True
