from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path, category: FileCategory) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=path.name,
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=category,
        extension=path.suffix,
        is_binary=False,
    )


def test_registry_dispatches_by_category(tmp_path: Path):
    registry = build_default_registry()

    sv_file = tmp_path / "top.sv"
    sv_file.write_text("module top; endmodule\n")
    result = registry.parse_file(sv_file, _record(sv_file, FileCategory.RTL_VERILOG))
    assert result is not None
    assert result.parser_name == "pyslang_syntax"

    py_file = tmp_path / "script.py"
    py_file.write_text("import os\n")
    result = registry.parse_file(py_file, _record(py_file, FileCategory.PYTHON))
    assert result is not None
    assert result.parser_name == "stdlib_ast"


def test_registry_returns_none_for_unregistered_category(tmp_path: Path):
    registry = build_default_registry()
    waveform_file = tmp_path / "dump.vcd"
    waveform_file.write_bytes(b"$date fake $end")
    result = registry.parse_file(waveform_file, _record(waveform_file, FileCategory.WAVEFORM))
    assert result is None


def test_uvm_and_assertion_categories_use_same_sv_parser(tmp_path: Path):
    registry = build_default_registry()

    tb_file = tmp_path / "driver.sv"
    tb_file.write_text("class i2c_driver extends uvm_driver; endclass\n")
    tb_result = registry.parse_file(tb_file, _record(tb_file, FileCategory.TESTBENCH_UVM))

    sva_file = tmp_path / "checks.sv"
    sva_file.write_text("module checks; endmodule\n")
    sva_result = registry.parse_file(sva_file, _record(sva_file, FileCategory.ASSERTION))

    assert tb_result.parser_name == "pyslang_syntax"
    assert sva_result.parser_name == "pyslang_syntax"
