from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.python_tcl.tcl_parser import TclParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _make_record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=f"Scripts/{path.name}",
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=FileCategory.TCL,
        extension=".tcl",
        is_binary=False,
    )


def test_extracts_procs_and_source_refs(tmp_path: Path):
    f = tmp_path / "build.tcl"
    f.write_text(
        """
        source common_setup.tcl
        source ./scripts/lint_setup.tcl

        proc run_lint {design_name top_module} {
            puts "linting $design_name"
        }

        proc setup_env {} {
            puts "env ready"
        }
        """
    )
    parser = TclParser()
    result = parser.parse(f, _make_record(f))

    assert result.success is True
    procs = {p["name"]: p for p in result.payload["procs"]}
    assert set(procs.keys()) == {"run_lint", "setup_env"}
    assert procs["run_lint"]["args"] == ["design_name", "top_module"]
    assert procs["setup_env"]["args"] == []

    source_paths = {s["path"] for s in result.payload["source_refs"]}
    assert source_paths == {"common_setup.tcl", "./scripts/lint_setup.tcl"}


def test_empty_file_produces_empty_structure(tmp_path: Path):
    f = tmp_path / "empty.tcl"
    f.write_text("")
    parser = TclParser()
    result = parser.parse(f, _make_record(f))
    assert result.success is True
    assert result.payload["procs"] == []
    assert result.payload["source_refs"] == []
