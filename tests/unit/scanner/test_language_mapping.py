from semrag.ingestion.scanner.classification import UNKNOWN_LANGUAGE, language_for_extension


def test_systemverilog_language_detection():
    assert language_for_extension("RTL/i2c_master.sv") == "systemverilog"
    assert language_for_extension("uvm/i2c_driver.svh") == "systemverilog"


def test_python_language_detection():
    assert language_for_extension("src/runner.py") == "python"
    assert language_for_extension("src/stubs.pyi") == "python"


def test_tcl_language_detection():
    assert language_for_extension("Scripts/build.tcl") == "tcl"


def test_markdown_language_detection():
    assert language_for_extension("docs/spec.md") == "markdown"
    assert language_for_extension("docs/spec.markdown") == "markdown"


def test_log_language_detection():
    assert language_for_extension("sim/log/regression.log") == "log"


def test_unknown_extension_behavior():
    assert language_for_extension("Scripts/notes.xyz") == UNKNOWN_LANGUAGE
    assert language_for_extension("Makefile") == UNKNOWN_LANGUAGE  # no extension at all


def test_extension_lookup_is_case_insensitive():
    assert language_for_extension("RTL/i2c_master.SV") == "systemverilog"
    assert language_for_extension("src/Runner.Py") == "python"
