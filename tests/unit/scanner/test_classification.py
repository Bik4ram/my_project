from semrag.ingestion.scanner.classification import classify, looks_like_binary_content
from semrag.ingestion.scanner.models import FileCategory


def test_classify_plain_rtl():
    assert classify("RTL/i2c_master.sv") == FileCategory.RTL_VERILOG


def test_classify_uvm_by_path_hint():
    assert classify("TB/uvm/i2c_driver.sv") == FileCategory.TESTBENCH_UVM


def test_classify_assertion_by_path_hint():
    assert classify("Assertions/i2c_sva.sv") == FileCategory.ASSERTION


def test_classify_coverage_report_vs_generic_log():
    assert classify("Coverage/i2c_master.log") == FileCategory.COVERAGE_REPORT
    assert classify("Regression/run_001.log") == FileCategory.LOG


def test_classify_lint_report():
    assert classify("Lint/i2c_master.rpt") == FileCategory.LINT_REPORT


def test_classify_git_internal():
    assert classify(".git/HEAD") == FileCategory.GIT_INTERNAL


def test_classify_unknown_extension():
    assert classify("Scripts/notes.xyz") == FileCategory.UNKNOWN


def test_binary_content_sniff():
    assert looks_like_binary_content(b"\x00\x01\x02")
    assert not looks_like_binary_content(b"module top; endmodule")
