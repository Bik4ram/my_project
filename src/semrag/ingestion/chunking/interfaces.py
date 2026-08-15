"""
Chunker interface and registry.

Same category-based dispatch pattern as Part 3's `ParserRegistry` —
deliberately, for the same reason: consistency across the ingestion
pipeline means a new engineer who understands Part 3's registry already
understands this one. Dispatch is keyed on `FileCategory` (Part 2), not
`IRNodeKind` (Part 4), because chunking's input is broader than IR nodes —
several categories (Markdown, VHDL, structured data, PDF, Excel, logs)
have no IR node at all (Part 4's builder only creates nodes for the SV/
Python/TCL categories), so a chunker must be runnable from
`FileRecord` + `ParseResult` alone, using an IR node when one happens to
exist rather than requiring one.

**Fallback to the generic chunker when the category-specific chunker
produces nothing:** the SystemVerilog chunker only emits chunks from
parsed module/interface/package nodes — a file containing only UVM
classes (e.g. a `uvm_scoreboard` subclass) parses cleanly but has zero
IR nodes, so the SV chunker returns an empty list. Without the
fallback, those files silently vanish from retrieval. The registry
runs the fallback whenever the primary chunker produced no chunks,
preserving the "sliding-window beats nothing" guarantee the generic
chunker was always meant to enforce.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileCategory, FileRecord


class Chunker(Protocol):
    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]: ...


class ChunkerRegistry:
    def __init__(self) -> None:
        self._chunkers: dict[FileCategory, Chunker] = {}
        self._fallback: Chunker | None = None

    def register(self, category: FileCategory, chunker: Chunker) -> None:
        self._chunkers[category] = chunker

    def register_fallback(self, chunker: Chunker) -> None:
        self._fallback = chunker

    def chunk_file(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        chunker = self._chunkers.get(file_record.category, self._fallback)
        if chunker is None:
            return []
        chunks = chunker.chunk(file_record, parse_result, file_text, ir_nodes)
        # Empty result + available fallback = the file has IR nodes that
        # the primary chunker didn't know how to slice. Run the generic
        # chunker so the file is at least discoverable in retrieval.
        if not chunks and self._fallback is not None and chunker is not self._fallback:
            chunker = self._fallback
            chunks = chunker.chunk(file_record, parse_result, file_text, ir_nodes)
        return self._stamp_chunker_name(chunks, chunker)

    def _stamp_chunker_name(self, chunks: list[Chunk], chunker: Chunker) -> list[Chunk]:
        """Stamp the actual producing chunker's name onto every chunk. Done
        here, not inside the chunkers, because the registry is the one place
        that knows which chunker really produced a given chunk — including
        the fallback path above — so the metadata can't lie. ``chunker_name``
        is a class attribute on each concrete chunker; the ``Chunker``
        protocol doesn't require it, so ``getattr`` keeps duck-typed chunkers
        compatible."""
        name = getattr(chunker, "chunker_name", "")
        if not name or not chunks:
            return chunks
        return [c.model_copy(update={"chunker_name": name}) for c in chunks]
