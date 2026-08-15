from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.chunking.markdown_chunker import MarkdownChunker
from semrag.ingestion.parsers.documents.markdown_parser import MarkdownParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1", absolute_path=str(path), relative_path=path.name,
        size_bytes=path.stat().st_size, mtime=datetime.now(timezone.utc), content_hash="x",
        category=FileCategory.MARKDOWN, extension=".md", is_binary=False,
    )


def test_sections_split_at_heading_boundaries(tmp_path: Path):
    f = tmp_path / "spec.md"
    f.write_text(
        "# I2C Master Spec\n"
        "\n"
        "## Overview\n"
        "Overview text.\n"
        "\n"
        "### Timing Requirements\n"
        "Timing text.\n"
        "\n"
        "## Register Map\n"
        "Register text.\n"
    )
    record = _record(f)
    parse_result = MarkdownParser().parse(f, record)
    chunks = MarkdownChunker().chunk(record, parse_result, f.read_text(), [])

    overview_chunk = next(c for c in chunks if "Overview text" in c.text)
    # Leaf content only — the subsection's content must NOT be duplicated
    # into the parent section's chunk (this was a real bug: the first
    # implementation let same-or-shallower-level sections swallow all
    # their children's content too).
    assert "Timing text" not in overview_chunk.text

    timing_chunk = next(c for c in chunks if "Timing text" in c.text)
    # Hierarchy is preserved as a breadcrumb, not content duplication.
    assert "I2C Master Spec > Overview > Timing Requirements" in timing_chunk.text

    register_chunk = next(c for c in chunks if "Register text" in c.text)
    assert "Register Map" in register_chunk.text
    assert "Timing text" not in register_chunk.text


def test_preamble_before_first_heading_is_captured(tmp_path: Path):
    f = tmp_path / "readme.md"
    f.write_text(
        "This is a preamble line before any heading.\n"
        "\n"
        "# First Heading\n"
        "Section content.\n"
    )
    record = _record(f)
    parse_result = MarkdownParser().parse(f, record)
    chunks = MarkdownChunker().chunk(record, parse_result, f.read_text(), [])

    preamble_chunks = [c for c in chunks if c.chunk_type == "markdown_preamble"]
    assert len(preamble_chunks) == 1
    assert "preamble line" in preamble_chunks[0].text


def test_no_headings_produces_single_whole_file_chunk(tmp_path: Path):
    f = tmp_path / "plain.md"
    f.write_text("Just plain prose, no headings at all.\n")
    record = _record(f)
    parse_result = MarkdownParser().parse(f, record)
    chunks = MarkdownChunker().chunk(record, parse_result, f.read_text(), [])

    assert len(chunks) == 1
    assert "plain prose" in chunks[0].text


def test_sibling_sections_do_not_bleed_into_each_other(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text(
        "## Section A\n"
        "Content A.\n"
        "## Section B\n"
        "Content B.\n"
    )
    record = _record(f)
    parse_result = MarkdownParser().parse(f, record)
    chunks = MarkdownChunker().chunk(record, parse_result, f.read_text(), [])

    a_chunk = next(c for c in chunks if "Content A" in c.text)
    b_chunk = next(c for c in chunks if "Content B" in c.text)
    assert "Content B" not in a_chunk.text
    assert "Content A" not in b_chunk.text
