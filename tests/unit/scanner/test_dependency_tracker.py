from semrag.ingestion.scanner.dependency_tracker import extract_raw_dependency_refs
from semrag.ingestion.scanner.models import FileCategory


def test_extract_sv_includes():
    text = '`include "defines.svh"\n`include "i2c_pkg.svh"\n'
    refs = extract_raw_dependency_refs(FileCategory.RTL_VERILOG, text)
    assert refs == ["defines.svh", "i2c_pkg.svh"]


def test_extract_uvm_package_import():
    text = "import i2c_pkg::*;\nimport uvm_pkg::*;\n"
    refs = extract_raw_dependency_refs(FileCategory.TESTBENCH_UVM, text)
    assert "i2c_pkg" in refs
    assert "uvm_pkg" in refs


def test_extract_python_imports():
    text = "import os\nfrom pathlib import Path\nimport semrag.ingestion.scanner\n"
    refs = extract_raw_dependency_refs(FileCategory.PYTHON, text)
    assert "os" in refs
    assert "pathlib" in refs
    assert "semrag.ingestion.scanner" in refs


def test_extract_tcl_source():
    text = "source common.tcl\nsource ./build/setup.tcl\n"
    refs = extract_raw_dependency_refs(FileCategory.TCL, text)
    assert "common.tcl" in refs
    assert "./build/setup.tcl" in refs


def test_no_patterns_for_unhandled_category_returns_empty():
    refs = extract_raw_dependency_refs(FileCategory.PDF, "import os")
    assert refs == []


def test_dedup_preserves_order():
    text = '`include "a.svh"\n`include "b.svh"\n`include "a.svh"\n'
    refs = extract_raw_dependency_refs(FileCategory.RTL_VERILOG, text)
    assert refs == ["a.svh", "b.svh"]
