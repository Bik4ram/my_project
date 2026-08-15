from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.logs.regression_log_parser import RegressionLogParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=path.name,
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=FileCategory.LOG,
        extension=".log",
        is_binary=False,
    )


def test_extracts_pass_result_and_uvm_severity_counts(tmp_path: Path):
    f = tmp_path / "run_001.log"
    f.write_text(
        "TESTNAME: i2c_master_smoke_test\n"
        "SEED: 12345\n"
        "UVM_INFO: starting test\n"
        "UVM_WARNING: minor timing skew observed\n"
        "UVM_INFO: test complete\n"
        "TEST PASSED\n"
    )
    result = RegressionLogParser().parse(f, _record(f))
    assert result.success is True
    assert result.payload["result"] == "PASS"
    assert result.payload["test_name"] == "i2c_master_smoke_test"
    assert result.payload["seed"] == 12345
    assert result.payload["uvm_info_count"] == 2
    assert result.payload["uvm_warning_count"] == 1


def test_extracts_fail_result_with_errors(tmp_path: Path):
    f = tmp_path / "run_002.log"
    f.write_text(
        "TESTNAME: i2c_master_stress_test\n"
        "UVM_ERROR: assertion failed at time 500ns\n"
        "UVM_FATAL: simulation aborted\n"
        "TEST FAILED\n"
    )
    result = RegressionLogParser().parse(f, _record(f))
    assert result.payload["result"] == "FAIL"
    assert result.payload["uvm_error_count"] == 1
    assert result.payload["uvm_fatal_count"] == 1


def test_log_with_no_recognizable_result_is_still_success(tmp_path: Path):
    f = tmp_path / "generic.log"
    f.write_text("some unrelated tool output\nnothing matches\n")
    result = RegressionLogParser().parse(f, _record(f))
    assert result.success is True
    assert result.payload["result"] is None
