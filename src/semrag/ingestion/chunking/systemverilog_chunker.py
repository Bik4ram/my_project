"""
SystemVerilog/UVM chunker.

Chunk boundaries come from Part 3/4's already-extracted structure (module
spans, assertion spans, procedural-block spans) — not a fixed-token
window — per Part 1's explicit chunking decision ("a fixed 512-token
window will cut a module definition in half"). This is the part of the
series where that architectural decision actually gets implemented, not
just stated.

Strategy, stated plainly because it has a real size-dependent branch:

- **Small modules (≤150 lines, the common case for well-factored DV
  code):** one chunk, the whole module. Splitting an 80-line module into
  fragments would hurt more than help — a query about the module benefits
  from seeing ports, always-block, and assertion together.
- **Large modules (>150 lines):** three chunk types instead of one whole
  chunk. A **signature chunk** (module declaration through the last port,
  before the first procedural/assertion construct) — this is what answers
  "what are this module's ports/parameters" without pulling in
  implementation. One **assertion chunk per assertion**, each with a
  couple of lines of surrounding context — assertions are, per Part 1's
  own example query ("why is coverage low on this assertion"), a
  high-value independently-retrievable unit, so they get their own chunk
  even inside a large module rather than only being reachable via a
  giant body chunk. Then **sliding-window chunks over the full module
  body** for complete coverage — assertions are important but a large
  module has plenty of other content (FSM logic, combinational blocks)
  that needs to be retrievable too, and enumerating "one chunk per
  procedural block" would miss non-block code and under-chunk large
  blocks; a window pass guarantees no gaps at the cost of some overlap
  with the assertion chunks, which is the same "duplication beats gaps"
  trade-off the generic fallback chunker makes explicitly.

Packages get one whole-package chunk each — SV packages in practice are
almost always small (type/parameter definitions), so the large-module
branch's complexity isn't needed there.

Classes get one whole-class chunk each, linked to the ``sv_class`` IR
node (``source_node_id`` + ``symbol_name``). The chunk deliberately spans
from the class declaration through the next top-level construct (module /
interface / package / class) or end-of-file, not just the class node's
span: the node span covers only ``class ... endclass``, while the bulk of
a UVM class's content lives in the out-of-class ``extern`` method bodies
that follow it (``function cls::new``, ``task cls::body``). Restricting
the chunk to the node span would silently drop that content — a
regression against the generic-window fallback that previously covered
the whole file.
"""
from __future__ import annotations

from semrag.ingestion.chunking.id_scheme import chunk_id, extract_lines
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode, IRNodeKind
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileRecord

SMALL_MODULE_LINE_THRESHOLD = 150
ASSERTION_CONTEXT_LINES = 2
BODY_WINDOW_LINES = 60
BODY_WINDOW_OVERLAP = 10


class SystemVerilogChunker:
    chunker_name = "systemverilog"

    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in ir_nodes:
            if node.kind in (IRNodeKind.SV_MODULE, IRNodeKind.SV_INTERFACE):
                chunks.extend(self._chunk_module(node, file_text))
            elif node.kind == IRNodeKind.SV_PACKAGE:
                chunks.extend(self._chunk_package(node, file_text))
            elif node.kind == IRNodeKind.SV_CLASS:
                chunks.extend(self._chunk_class(node, file_text, ir_nodes))
        return chunks

    def _chunk_class(self, node: IRNode, file_text: str, ir_nodes: list[IRNode]) -> list[Chunk]:
        span = node.attributes.get("span", {})
        start_line = span.get("start_line")
        if start_line is None:
            return []
        end_line = self._class_region_end(node, file_text, ir_nodes)
        if end_line < start_line:
            return []
        return [
            Chunk(
                chunk_id=chunk_id(node.project_id, node.relative_path, "sv_class", node.name),
                project_id=node.project_id,
                relative_path=node.relative_path,
                chunk_type="sv_class",
                source_node_id=node.id,
                text=extract_lines(file_text, start_line, end_line),
                line_start=start_line,
                line_end=end_line,
                symbol_name=node.name,
            )
        ]

    def _class_region_end(
        self, node: IRNode, file_text: str, ir_nodes: list[IRNode]
    ) -> int:
        """End of the class-owned region: the line before the next
        top-level construct in the file, or EOF. Deliberately not bounded
        by the class node span so out-of-class extern method bodies stay
        part of the chunk."""
        top_level_kinds = (
            IRNodeKind.SV_MODULE,
            IRNodeKind.SV_INTERFACE,
            IRNodeKind.SV_PACKAGE,
            IRNodeKind.SV_CLASS,
        )
        start_line = node.attributes.get("span", {}).get("start_line")
        next_starts = [
            n.attributes.get("span", {}).get("start_line")
            for n in ir_nodes
            if n.kind in top_level_kinds
            and n.id != node.id
            and n.attributes.get("span", {}).get("start_line") is not None
            and n.attributes["span"]["start_line"] > start_line
        ]
        if next_starts:
            return min(next_starts) - 1
        return len(file_text.splitlines())

    def _chunk_module(self, node: IRNode, file_text: str) -> list[Chunk]:
        span = node.attributes.get("span", {})
        start_line, end_line = span.get("start_line"), span.get("end_line")
        if start_line is None or end_line is None:
            return []

        total_lines = end_line - start_line + 1
        if total_lines <= SMALL_MODULE_LINE_THRESHOLD:
            return [self._whole_module_chunk(node, file_text, start_line, end_line)]

        chunks = [self._signature_chunk(node, file_text, start_line, end_line)]
        chunks.extend(self._assertion_chunks(node, file_text, start_line, end_line))
        chunks.extend(self._body_window_chunks(node, file_text, start_line, end_line))
        return chunks

    def _whole_module_chunk(self, node: IRNode, file_text: str, start_line: int, end_line: int) -> Chunk:
        return Chunk(
            chunk_id=chunk_id(node.project_id, node.relative_path, "sv_module", node.name),
            project_id=node.project_id,
            relative_path=node.relative_path,
            chunk_type="sv_module",
            source_node_id=node.id,
            text=extract_lines(file_text, start_line, end_line),
            line_start=start_line,
            line_end=end_line,
            symbol_name=node.name,
        )

    def _signature_end_line(self, node: IRNode, module_end_line: int) -> int:
        body_starts = [
            b["span"]["start_line"] for b in node.attributes.get("procedural_blocks", [])
        ] + [
            a["span"]["start_line"] for a in node.attributes.get("assertions", [])
        ]
        if not body_starts:
            return module_end_line
        return min(body_starts) - 1

    def _signature_chunk(self, node: IRNode, file_text: str, start_line: int, end_line: int) -> Chunk:
        sig_end = max(self._signature_end_line(node, end_line), start_line)
        return Chunk(
            chunk_id=chunk_id(node.project_id, node.relative_path, "sv_module_signature", node.name),
            project_id=node.project_id,
            relative_path=node.relative_path,
            chunk_type="sv_module_signature",
            source_node_id=node.id,
            text=extract_lines(file_text, start_line, sig_end),
            line_start=start_line,
            line_end=sig_end,
            symbol_name=node.name,
        )

    def _assertion_chunks(self, node: IRNode, file_text: str, module_start: int, module_end: int) -> list[Chunk]:
        chunks = []
        for i, assertion in enumerate(node.attributes.get("assertions", [])):
            a_start = max(assertion["span"]["start_line"] - ASSERTION_CONTEXT_LINES, module_start)
            a_end = min(assertion["span"]["end_line"] + ASSERTION_CONTEXT_LINES, module_end)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(
                        node.project_id, node.relative_path, "sv_assertion", f"{node.name}::{i}"
                    ),
                    project_id=node.project_id,
                    relative_path=node.relative_path,
                    chunk_type="sv_assertion",
                    source_node_id=node.id,
                    text=extract_lines(file_text, a_start, a_end),
                    line_start=a_start,
                    line_end=a_end,
                    symbol_name=node.name,
                )
            )
        return chunks

    def _body_window_chunks(self, node: IRNode, file_text: str, start_line: int, end_line: int) -> list[Chunk]:
        chunks = []
        step = BODY_WINDOW_LINES - BODY_WINDOW_OVERLAP
        window_start = start_line
        index = 0
        while window_start <= end_line:
            window_end = min(window_start + BODY_WINDOW_LINES - 1, end_line)
            text = extract_lines(file_text, window_start, window_end)
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(
                            node.project_id, node.relative_path, "sv_body_window", f"{node.name}::{index}"
                        ),
                        project_id=node.project_id,
                        relative_path=node.relative_path,
                        chunk_type="sv_body_window",
                        source_node_id=node.id,
                        text=text,
                        line_start=window_start,
                        line_end=window_end,
                        symbol_name=node.name,
                    )
                )
                index += 1
            if window_end == end_line:
                break
            window_start += step
        return chunks

    def _chunk_package(self, node: IRNode, file_text: str) -> list[Chunk]:
        span = node.attributes.get("span", {})
        start_line, end_line = span.get("start_line"), span.get("end_line")
        if start_line is None or end_line is None:
            return []
        return [
            Chunk(
                chunk_id=chunk_id(node.project_id, node.relative_path, "sv_package", node.name),
                project_id=node.project_id,
                relative_path=node.relative_path,
                chunk_type="sv_package",
                source_node_id=node.id,
                text=extract_lines(file_text, start_line, end_line),
                line_start=start_line,
                line_end=end_line,
                symbol_name=node.name,
            )
        ]
