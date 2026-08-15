"""
Extension/path-based file classification.

This is deliberately a fast, cheap heuristic layer — a lookup table plus a
handful of path-pattern rules — not a parser. Its only job is to tell the
orchestrator which downstream parser subpackage (Part 3) should eventually
claim the file, and whether the walker should even read the file's content
for hashing+dependency scanning or treat it as opaque binary.

Design note: report/log classification uses *path* patterns (e.g. a file
living under a `Coverage/` directory) in addition to extension, because
compile logs, coverage reports, and lint reports frequently share the same
`.log` or `.txt` extension and are only distinguishable by where the tool
that produced them wrote them. This is intentionally a soft heuristic —
Part 3's log parsers re-confirm classification by sniffing file content
(e.g. a VCS coverage summary header) before committing to a parser.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from semrag.ingestion.scanner.models import FileCategory

_EXTENSION_MAP: dict[str, FileCategory] = {
    ".v": FileCategory.RTL_VERILOG,
    ".vh": FileCategory.RTL_VERILOG,
    ".sv": FileCategory.RTL_VERILOG,
    ".svh": FileCategory.RTL_VERILOG,
    ".vhd": FileCategory.RTL_VHDL,
    ".vhdl": FileCategory.RTL_VHDL,
    ".py": FileCategory.PYTHON,
    ".pyi": FileCategory.PYTHON,
    ".tcl": FileCategory.TCL,
    ".sh": FileCategory.SHELL,
    ".bash": FileCategory.SHELL,
    ".md": FileCategory.MARKDOWN,
    ".markdown": FileCategory.MARKDOWN,
    ".pdf": FileCategory.PDF,
    ".xlsx": FileCategory.EXCEL,
    ".xls": FileCategory.EXCEL,
    ".xlsm": FileCategory.EXCEL,
    ".csv": FileCategory.CSV,
    ".json": FileCategory.JSON,
    ".yaml": FileCategory.YAML,
    ".yml": FileCategory.YAML,
    ".xml": FileCategory.XML,
    ".log": FileCategory.LOG,
    ".rpt": FileCategory.TIMING_REPORT,
    ".vcd": FileCategory.WAVEFORM,
    ".fsdb": FileCategory.WAVEFORM,
    ".wlf": FileCategory.WAVEFORM,
    ".shm": FileCategory.WAVEFORM,
    ".mk": FileCategory.BUILD_CONFIG,
    ".cmake": FileCategory.BUILD_CONFIG,
    ".f": FileCategory.BUILD_CONFIG,  # filelists (-f files), common in DV flows
}

# Canonical extension → human-readable language mapping (RI-06). This is
# the single source of truth for chunk ``language`` metadata; it exists so
# retrieval can distinguish e.g. SystemVerilog from Python/TCL without
# guessing from text at query time. Coarser than ``FileCategory`` on
# purpose — ``language`` is what the file is written in, ``category`` is
# what kind of document it is (RTL vs testbench vs spec), and the two are
# deliberately separate concepts.
_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".v": "verilog",
    ".vh": "verilog",
    ".sv": "systemverilog",
    ".svh": "systemverilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
    ".py": "python",
    ".pyi": "python",
    ".tcl": "tcl",
    ".sh": "shell",
    ".bash": "shell",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".csv": "csv",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".log": "log",
    ".rpt": "text",
    ".vcd": "waveform",
    ".fsdb": "waveform",
    ".wlf": "waveform",
    ".shm": "waveform",
    ".mk": "build",
    ".cmake": "build",
    ".f": "build",
}

# Explicit unknown sentinel so a missing extension is distinguishable from
# an empty default (a chunk constructed before enrichment was filled in).
UNKNOWN_LANGUAGE = "unknown"


def language_for_extension(relative_path: str) -> str:
    """Return the canonical language for a file's extension.

    ``relative_path`` (posix-style) rather than the bare suffix so the
    call site reads naturally and the extension lookup is centralized.
    Unknown/extensionless files return ``UNKNOWN_LANGUAGE``.
    """
    ext = PurePosixPath(relative_path).suffix.lower()
    return _LANGUAGE_BY_EXTENSION.get(ext, UNKNOWN_LANGUAGE)


# Known-binary extensions the walker should never attempt to read as text
# for dependency-ref scanning, even if bytes happen to decode.
_BINARY_EXTENSIONS = {
    ".vcd", ".fsdb", ".wlf", ".shm", ".db", ".so", ".o", ".a",
    ".xlsx", ".xls", ".xlsm", ".pdf", ".png", ".jpg", ".jpeg", ".gif",
}

_UVM_PATH_HINT = re.compile(r"(^|/)(uvm|tb|testbench)(/|$)", re.IGNORECASE)
_ASSERTION_PATH_HINT = re.compile(r"(^|/)(assertions?|sva)(/|$)", re.IGNORECASE)
_COVERAGE_PATH_HINT = re.compile(r"(^|/)coverage(/|$)", re.IGNORECASE)
_LINT_PATH_HINT = re.compile(r"(^|/)lint(/|$)", re.IGNORECASE)
_GIT_INTERNAL_HINT = re.compile(r"(^|/)\.git(/|$)")


def classify(relative_path: str) -> FileCategory:
    """Classify a file given its repo-relative path (posix-style separators
    expected — callers normalize this in the walker)."""
    p = PurePosixPath(relative_path)
    ext = p.suffix.lower()

    if _GIT_INTERNAL_HINT.search(relative_path):
        return FileCategory.GIT_INTERNAL

    # Path-pattern overrides take priority over raw extension for the
    # ambiguous report/log/testbench cases described in the module docstring.
    if ext in (".log", ".rpt", ".txt"):
        if _COVERAGE_PATH_HINT.search(relative_path):
            return FileCategory.COVERAGE_REPORT
        if _LINT_PATH_HINT.search(relative_path):
            return FileCategory.LINT_REPORT
        if ext in (".log",):
            return FileCategory.LOG
        if ext == ".rpt":
            return FileCategory.TIMING_REPORT

    if ext in (".sv", ".svh") and (
        _UVM_PATH_HINT.search(relative_path) or _ASSERTION_PATH_HINT.search(relative_path)
    ):
        if _ASSERTION_PATH_HINT.search(relative_path):
            return FileCategory.ASSERTION
        return FileCategory.TESTBENCH_UVM

    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]

    return FileCategory.UNKNOWN


def is_binary_extension(relative_path: str) -> bool:
    ext = PurePosixPath(relative_path).suffix.lower()
    return ext in _BINARY_EXTENSIONS


def looks_like_binary_content(sample: bytes) -> bool:
    """Cheap binary sniff on the first chunk of a file, used as a second
    line of defense for files whose extension doesn't tell us (e.g. `.f`
    filelists are always text; some `.dat` files are not). A NUL byte in
    the first 8KB is the standard heuristic (same one git itself uses)."""
    return b"\x00" in sample
