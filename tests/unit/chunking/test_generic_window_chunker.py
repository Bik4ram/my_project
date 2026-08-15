from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.chunking.generic_window_chunker import GenericWindowChunker
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1", absolute_path=str(path), relative_path=path.name,
        size_bytes=path.stat().st_size, mtime=datetime.now(timezone.utc), content_hash="x",
        category=FileCategory.TCL, extension=".tcl", is_binary=False,
    )


def test_small_file_produces_one_window(tmp_path: Path):
    f = tmp_path / "small.tcl"
    f.write_text("\n".join(f"line {i}" for i in range(10)) + "\n")
    record = _record(f)
    chunker = GenericWindowChunker(window_lines=60, overlap_lines=10)
    chunks = chunker.chunk(record, None, f.read_text(), [])

    assert len(chunks) == 1
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 10


def test_large_file_produces_overlapping_windows_covering_everything(tmp_path: Path):
    total_lines = 200
    f = tmp_path / "large.tcl"
    f.write_text("\n".join(f"line {i}" for i in range(total_lines)) + "\n")
    record = _record(f)
    chunker = GenericWindowChunker(window_lines=60, overlap_lines=10)
    chunks = chunker.chunk(record, None, f.read_text(), [])

    assert len(chunks) > 1
    assert chunks[0].line_start == 1
    assert chunks[-1].line_end == total_lines

    # Every line must be covered by at least one window (no gaps).
    covered = set()
    for c in chunks:
        covered.update(range(c.line_start, c.line_end + 1))
    assert covered == set(range(1, total_lines + 1))

    # Consecutive windows overlap by the configured amount.
    for a, b in zip(chunks, chunks[1:]):
        assert b.line_start <= a.line_end  # overlap or contiguous, never a gap


def test_empty_file_produces_no_chunks(tmp_path: Path):
    f = tmp_path / "empty.tcl"
    f.write_text("")
    record = _record(f)
    chunker = GenericWindowChunker()
    chunks = chunker.chunk(record, None, f.read_text(), [])
    assert chunks == []


def test_invalid_overlap_configuration_rejected():
    import pytest
    with pytest.raises(ValueError):
        GenericWindowChunker(window_lines=50, overlap_lines=50)
