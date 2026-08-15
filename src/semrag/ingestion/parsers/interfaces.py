"""
Parser plugin interface and registry.

Design decision: parsers are dispatched by the scanner's `FileCategory`
(Part 2), not by file extension directly — the scanner already did the
(cheap, heuristic) work of disambiguating e.g. a UVM testbench `.sv` file
from a plain RTL `.sv` file via path hints, and the registry reuses that
rather than re-deriving it. A parser is free to internally re-classify
(e.g. the SystemVerilog parser treats RTL_VERILOG, TESTBENCH_UVM, and
ASSERTION identically at the syntax level, since the actual grammar is the
same file format regardless of directory convention) — the registry's
category mapping is about routing, not a claim that the categories are
semantically distinct to every parser.

Fallback chain: each `FileCategory` maps to an ordered list of parsers,
not a single one. `ParserRegistry.parse_file` tries them in order and
moves to the next only if the previous one's `ParseResult.success` is
False. This is what makes room for a tree-sitter fallback parser for
malformed/partial SystemVerilog files (see systemverilog/README.md for
why this fallback is designed but not yet implemented in this delivery)
without the orchestrator needing any special-case logic — it's just a
second registered parser for the same categories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileCategory, FileRecord


class Parser(Protocol):
    """Contract every language-specific parser implements."""

    name: str

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult: ...


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers_by_category: dict[FileCategory, list[Parser]] = {}

    def register(self, category: FileCategory, parser: Parser) -> None:
        self._parsers_by_category.setdefault(category, []).append(parser)

    def parsers_for(self, category: FileCategory) -> list[Parser]:
        return list(self._parsers_by_category.get(category, []))

    def parse_file(self, absolute_path: Path, file_record: FileRecord) -> ParseResult | None:
        """Tries each registered parser for this file's category in
        order, returning the first successful result. Returns None (not
        an exception) if no parser is registered for the category at
        all — an unparsed file is an expected, common case (e.g. a
        waveform dump), not an error."""
        parsers = self._parsers_by_category.get(file_record.category, [])
        if not parsers:
            return None

        last_result: ParseResult | None = None
        for parser in parsers:
            result = parser.parse(absolute_path, file_record)
            last_result = result
            if result.success:
                return result
        # All registered parsers failed; return the last failure so the
        # caller has diagnostics to log, rather than silently returning None.
        return last_result
