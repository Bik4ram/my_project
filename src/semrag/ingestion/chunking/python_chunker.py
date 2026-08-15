"""
Python chunker.

Same structure-aware philosophy as the SystemVerilog chunker: chunk
boundaries are function/class definitions from Part 3's `ast`-derived
structure, not fixed windows. A module-overview chunk (imports +
docstring, everything before the first function/class) is emitted when
that region is non-trivial, so a module's imports/purpose is
independently retrievable from any one function's implementation.

Classes get the same size-dependent branch as SV modules: a small class
(≤100 lines — most well-factored Python classes) is one chunk; a large
class gets a signature chunk (class line through the line before its
first method) plus one chunk per method, so an individual method is
independently retrievable inside a large class rather than only reachable
via one giant class-sized chunk.
"""
from __future__ import annotations

from semrag.ingestion.chunking.id_scheme import chunk_id, extract_lines
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode, IRNodeKind
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileRecord

SMALL_CLASS_LINE_THRESHOLD = 100


class PythonChunker:
    chunker_name = "python"

    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in ir_nodes:
            if node.kind != IRNodeKind.PY_MODULE:
                continue
            chunks.extend(self._module_overview_chunk(node, file_text))
            for func in node.attributes.get("functions", []):
                chunks.append(self._function_chunk(node, file_text, func, chunk_type="py_function"))
            for cls in node.attributes.get("classes", []):
                chunks.extend(self._chunk_class(node, file_text, cls))
        return chunks

    def _first_construct_line(self, node: IRNode) -> int | None:
        lines = [f["line"] for f in node.attributes.get("functions", [])]
        lines += [c["line"] for c in node.attributes.get("classes", [])]
        return min(lines) if lines else None

    def _module_overview_chunk(self, node: IRNode, file_text: str) -> list[Chunk]:
        first_line = self._first_construct_line(node)
        overview_end = (first_line - 1) if first_line else len(file_text.splitlines())
        if overview_end < 1:
            return []
        text = extract_lines(file_text, 1, overview_end)
        if not text.strip():
            return []
        return [
            Chunk(
                chunk_id=chunk_id(node.project_id, node.relative_path, "py_module_overview", node.name),
                project_id=node.project_id,
                relative_path=node.relative_path,
                chunk_type="py_module_overview",
                source_node_id=node.id,
                text=text,
                line_start=1,
                line_end=overview_end,
                symbol_name=node.name,
            )
        ]

    def _function_chunk(self, node: IRNode, file_text: str, func: dict, chunk_type: str, prefix: str = "") -> Chunk:
        name = f"{prefix}{func['name']}"
        return Chunk(
            chunk_id=chunk_id(node.project_id, node.relative_path, chunk_type, f"{node.name}::{name}"),
            project_id=node.project_id,
            relative_path=node.relative_path,
            chunk_type=chunk_type,
            source_node_id=node.id,
            text=extract_lines(file_text, func["line"], func["end_line"]),
            line_start=func["line"],
            line_end=func["end_line"],
            symbol_name=name,
        )

    def _chunk_class(self, node: IRNode, file_text: str, cls: dict) -> list[Chunk]:
        total_lines = cls["end_line"] - cls["line"] + 1
        if total_lines <= SMALL_CLASS_LINE_THRESHOLD:
            return [
                Chunk(
                    chunk_id=chunk_id(node.project_id, node.relative_path, "py_class", f"{node.name}::{cls['name']}"),
                    project_id=node.project_id,
                    relative_path=node.relative_path,
                    chunk_type="py_class",
                    source_node_id=node.id,
                    text=extract_lines(file_text, cls["line"], cls["end_line"]),
                    line_start=cls["line"],
                    line_end=cls["end_line"],
                    symbol_name=cls["name"],
                )
            ]

        methods = cls.get("methods", [])
        sig_end = (min(m["line"] for m in methods) - 1) if methods else cls["end_line"]
        sig_end = max(sig_end, cls["line"])
        chunks = [
            Chunk(
                chunk_id=chunk_id(
                    node.project_id, node.relative_path, "py_class_signature", f"{node.name}::{cls['name']}"
                ),
                project_id=node.project_id,
                relative_path=node.relative_path,
                chunk_type="py_class_signature",
                source_node_id=node.id,
                text=extract_lines(file_text, cls["line"], sig_end),
                line_start=cls["line"],
                line_end=sig_end,
                symbol_name=cls["name"],
            )
        ]
        for method in methods:
            chunks.append(
                self._function_chunk(node, file_text, method, chunk_type="py_method", prefix=f"{cls['name']}.")
            )
        return chunks
