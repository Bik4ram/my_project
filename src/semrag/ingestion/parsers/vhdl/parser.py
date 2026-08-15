"""
VHDL parser.

Same trade-off class as `python_tcl/tcl_parser.py`: regex-structural, not a
full grammar parser. This isn't just a cost-saving shortcut taken without
checking — `hdlConvertor` (the most credible Python VHDL/Verilog AST
library) was evaluated for this delivery and failed to build in this
environment (native CMake/Ninja compilation step, not a pure-Python
wheel), which is itself informative: a real VHDL grammar parser is a much
heavier operational dependency (native toolchain required at install time)
than pyslang's prebuilt wheel was for SystemVerilog. Given VHDL is a
secondary language in most of the DV flows this system targets (SystemVerilog/UVM
being primary, per Part 1's stated project background), a regex-structural
extractor — entity name, ports, generics, architecture-to-entity binding —
captures what retrieval and dependency-graph seeding need, without taking
on a native-build dependency for secondary-language coverage. Revisit if
VHDL turns out to be a larger fraction of a given project than assumed.

Handles the common case (one entity/architecture per declaration, generic
and port clauses using standard `name : type` and `name : type := default`
forms) and will under-match unusual formatting (multiple entities using
shared generic/port block macros, heavily vendor-specific preprocessor
usage) — same category of limitation as the TCL parser, same reasoning
for accepting it.
"""
from __future__ import annotations

import re
from pathlib import Path

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.parsers.vhdl.models import (
    ParsedVhdlArchitecture,
    ParsedVhdlEntity,
    ParsedVhdlGeneric,
    ParsedVhdlPort,
    VhdlFileStructure,
)
from semrag.ingestion.scanner.models import FileRecord

_ENTITY_PATTERN = re.compile(
    r"entity\s+(\w+)\s+is(.*?)end\s+(?:entity\s+)?\1?\s*;",
    re.IGNORECASE | re.DOTALL,
)
_ARCHITECTURE_PATTERN = re.compile(
    r"architecture\s+(\w+)\s+of\s+(\w+)\s+is", re.IGNORECASE
)
_GENERIC_KEYWORD_PATTERN = re.compile(r"generic\s*\(", re.IGNORECASE)
_PORT_KEYWORD_PATTERN = re.compile(r"port\s*\(", re.IGNORECASE)
_DIRECTION_KEYWORDS = {"in", "out", "inout", "buffer"}


def _extract_balanced_block(text: str, keyword_pattern: re.Pattern) -> str | None:
    """Finds `keyword (` and returns the content between that `(` and its
    matching `)`, tracking paren depth rather than relying on a regex
    lookahead for what follows — necessary because a port's own type can
    contain parens (`std_logic_vector(7 downto 0)`), which a naive
    non-greedy `.*?\\)` regex can be fooled by."""
    match = keyword_pattern.search(text)
    if match is None:
        return None
    start = match.end()  # position right after the opening '('
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unbalanced parens — malformed input, bail out cleanly
    return text[start:i - 1]


def _split_top_level(text: str) -> list[str]:
    """Splits a declaration block on ';' at paren-depth 0, so a default
    value expression like `to_unsigned(5, 8)` doesn't get split mid-call."""
    parts: list[str] = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


class VhdlParser:
    name = "vhdl_regex_structural"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        try:
            text = absolute_path.read_text(errors="replace")
        except (FileNotFoundError, PermissionError) as exc:
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=f"Could not read file: {exc}")
                ],
            )

        entities = [self._extract_entity(m, text) for m in _ENTITY_PATTERN.finditer(text)]
        architectures = [
            ParsedVhdlArchitecture(
                name=m.group(1),
                entity_name=m.group(2),
                line=text.count("\n", 0, m.start()) + 1,
            )
            for m in _ARCHITECTURE_PATTERN.finditer(text)
        ]

        structure = VhdlFileStructure(entities=entities, architectures=architectures)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )

    def _extract_entity(self, match: re.Match, full_text: str) -> ParsedVhdlEntity:
        name = match.group(1)
        body = match.group(2)
        line = full_text.count("\n", 0, match.start()) + 1

        generics: list[ParsedVhdlGeneric] = []
        generic_block = _extract_balanced_block(body, _GENERIC_KEYWORD_PATTERN)
        if generic_block:
            for decl in _split_top_level(generic_block):
                generics.extend(self._parse_generic_decl(decl))

        ports: list[ParsedVhdlPort] = []
        port_block = _extract_balanced_block(body, _PORT_KEYWORD_PATTERN)
        if port_block:
            for decl in _split_top_level(port_block):
                ports.extend(self._parse_port_decl(decl))

        return ParsedVhdlEntity(name=name, line=line, ports=ports, generics=generics)

    def _parse_port_decl(self, decl: str) -> list[ParsedVhdlPort]:
        if ":" not in decl:
            return []
        names_part, rest = decl.split(":", 1)
        names = [n.strip() for n in names_part.split(",") if n.strip()]
        rest = rest.strip()
        tokens = rest.split(None, 1)
        if not tokens:
            return []
        direction = tokens[0].lower()
        if direction not in _DIRECTION_KEYWORDS:
            # No explicit direction keyword found — not a port-like decl.
            return []
        data_type = tokens[1].strip() if len(tokens) > 1 else ""
        return [ParsedVhdlPort(name=n, direction=direction, data_type=data_type) for n in names]

    def _parse_generic_decl(self, decl: str) -> list[ParsedVhdlGeneric]:
        if ":" not in decl:
            return []
        names_part, rest = decl.split(":", 1)
        names = [n.strip() for n in names_part.split(",") if n.strip()]
        rest = rest.strip()
        default_value = None
        if ":=" in rest:
            data_type, default_value = rest.split(":=", 1)
            data_type = data_type.strip()
            default_value = default_value.strip()
        else:
            data_type = rest
        return [
            ParsedVhdlGeneric(name=n, data_type=data_type or None, default_value=default_value)
            for n in names
        ]
