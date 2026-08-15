"""
Lightweight, regex-based dependency reference extraction.

Explicitly NOT a parser. Part 3's PySlang/tree-sitter parsers will resolve
real dependency edges (module instantiation, package imports, `include
chains) with full semantic accuracy. This module exists because the
scanner walks every file's content anyway (for the binary sniff and
hashing) and it is cheap to also regex-scan the same in-memory sample for
raw reference strings — capturing this signal now means Part 3 has a
worklist of candidate edges to resolve, rather than having to rediscover
"what files does this file mention" from scratch.

The trade-off is explicit: regexes here will both over- and under-match
(a commented-out `` `include `` line still counts; a dynamically
constructed TCL `source` path won't be caught at all). That's acceptable
for a hint signal that Part 4's IR builder verifies before treating as a
real edge, and would NOT be acceptable if this were the system of record
for dependency graph edges — it isn't.
"""
from __future__ import annotations

import re

from semrag.ingestion.scanner.models import FileCategory

_PATTERNS: dict[FileCategory, list[re.Pattern]] = {
    FileCategory.RTL_VERILOG: [
        re.compile(r'`include\s+"([^"]+)"'),
    ],
    FileCategory.TESTBENCH_UVM: [
        re.compile(r'`include\s+"([^"]+)"'),
        re.compile(r'\bimport\s+([a-zA-Z_][\w:]*)::\*?;'),  # SV package import
    ],
    FileCategory.ASSERTION: [
        re.compile(r'`include\s+"([^"]+)"'),
    ],
    FileCategory.PYTHON: [
        re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE),
        re.compile(r'^\s*from\s+([\w.]+)\s+import\b', re.MULTILINE),
    ],
    FileCategory.TCL: [
        re.compile(r'\bsource\s+([^\s;]+)'),
    ],
    FileCategory.BUILD_CONFIG: [
        # filelist (-f file) entries referencing other files/filelists
        re.compile(r'^\s*-f\s+(\S+)', re.MULTILINE),
        re.compile(r'^\s*([^\s#-][\S]*\.(?:sv|svh|v|vh))\s*$', re.MULTILINE),
    ],
}


def extract_raw_dependency_refs(category: FileCategory, text_sample: str) -> list[str]:
    """Returns a de-duplicated, order-preserved list of raw reference
    strings found in the given text sample. Caller (walker/orchestrator)
    decodes bytes to text before calling this; kept as a pure string
    function so it's trivially unit-testable without touching the
    filesystem."""
    patterns = _PATTERNS.get(category)
    if not patterns:
        return []

    seen: set[str] = set()
    refs: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text_sample):
            ref = match.group(1).strip()
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs
