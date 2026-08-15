"""
Default parser pipeline assembly.

This is the single place that wires concrete parsers to the
`FileCategory`s they handle. Downstream code (the future ingestion worker
in Part 2's `workers/`, and Part 4's IR builder) imports
`build_default_registry()`, not individual parser classes — consistent
with Part 1's rule that a plane's internals stay behind its interface.

Coverage as of this delivery: SystemVerilog/UVM/assertions (pyslang),
VHDL (regex-structural), Python (stdlib ast), TCL and Shell
(regex-structural), Markdown (regex-structural headings), JSON/YAML/CSV/XML
(shape extraction), PDF (pdfplumber, text + tables), Excel (openpyxl,
sheet shape + samples), and plain regression logs (heuristic PASS/FAIL/
severity-count extraction).

NOT yet wired, and deliberately so — see each parser's module docstring
for the specific reasoning: COVERAGE_REPORT, TIMING_REPORT, LINT_REPORT
(tool-specific formats needing real sample files to validate against,
which this delivery doesn't have), WAVEFORM (binary dump formats — VCD is
technically parseable but waveform *content* isn't retrieval-relevant
text; likely a metadata-only extraction — signal list, time range — is
the right scope, not attempted yet), BUILD_CONFIG (`.f` filelists and
Makefiles — filelists are a strong Part 6 candidate since they directly
define compile-order dependency edges the IR would want).
"""
from __future__ import annotations

from semrag.ingestion.parsers.documents.excel_parser import ExcelParser
from semrag.ingestion.parsers.documents.markdown_parser import MarkdownParser
from semrag.ingestion.parsers.documents.pdf_parser import PdfParser
from semrag.ingestion.parsers.interfaces import ParserRegistry
from semrag.ingestion.parsers.logs.regression_log_parser import RegressionLogParser
from semrag.ingestion.parsers.python_tcl.python_parser import PythonParser
from semrag.ingestion.parsers.python_tcl.shell_parser import ShellParser
from semrag.ingestion.parsers.python_tcl.tcl_parser import TclParser
from semrag.ingestion.parsers.structured_data.parsers import (
    CsvParser,
    JsonParser,
    XmlParser,
    YamlParser,
)
from semrag.ingestion.parsers.systemverilog.parser import SystemVerilogParser
from semrag.ingestion.parsers.vhdl.parser import VhdlParser
from semrag.ingestion.scanner.models import FileCategory


def build_default_registry() -> ParserRegistry:
    registry = ParserRegistry()

    sv_parser = SystemVerilogParser()
    for category in (
        FileCategory.RTL_VERILOG,
        FileCategory.TESTBENCH_UVM,
        FileCategory.ASSERTION,
    ):
        registry.register(category, sv_parser)
        # A tree-sitter fallback parser would be registered here, second,
        # for the same categories — see systemverilog/README.md.

    registry.register(FileCategory.RTL_VHDL, VhdlParser())
    registry.register(FileCategory.PYTHON, PythonParser())
    registry.register(FileCategory.TCL, TclParser())
    registry.register(FileCategory.SHELL, ShellParser())
    registry.register(FileCategory.MARKDOWN, MarkdownParser())
    registry.register(FileCategory.JSON, JsonParser())
    registry.register(FileCategory.YAML, YamlParser())
    registry.register(FileCategory.CSV, CsvParser())
    registry.register(FileCategory.XML, XmlParser())
    registry.register(FileCategory.PDF, PdfParser())
    registry.register(FileCategory.EXCEL, ExcelParser())
    registry.register(FileCategory.LOG, RegressionLogParser())

    return registry
