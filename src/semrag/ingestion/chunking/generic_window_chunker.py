"""
Generic sliding-window chunker.

Used wherever no bespoke, structure-aware chunker is registered (see
`pipeline.py` for exactly which categories that is as of this delivery —
matches the same "broadened later, flagged explicitly" pattern Part 3
used for its own parser coverage). Splits raw file text into overlapping
fixed-line windows, ignoring parsed structure entirely — deliberately
simple, because a generic chunker cannot know what structural boundaries
matter for an arbitrary file type; guessing badly (e.g. splitting a CSV
mid-row) would be worse than not guessing at all.

Overlap between consecutive windows exists so a concept spanning a window
boundary isn't invisible to retrieval — Part 1's "chunk overlap"
requirement — at the cost of some retrievable duplication, which is a
better failure mode than a gap.
"""
from __future__ import annotations

from semrag.ingestion.chunking.id_scheme import chunk_id
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileRecord

DEFAULT_WINDOW_LINES = 60
DEFAULT_OVERLAP_LINES = 10


class GenericWindowChunker:
    chunker_name = "generic_window"

    def __init__(self, window_lines: int = DEFAULT_WINDOW_LINES, overlap_lines: int = DEFAULT_OVERLAP_LINES):
        if overlap_lines >= window_lines:
            raise ValueError("overlap_lines must be smaller than window_lines")
        self._window = window_lines
        self._overlap = overlap_lines

    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        lines = file_text.splitlines()
        if not lines:
            return []

        chunks: list[Chunk] = []
        step = self._window - self._overlap
        start = 0
        window_index = 0
        while start < len(lines):
            end = min(start + self._window, len(lines))
            text = "\n".join(lines[start:end])
            if text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(
                            file_record.project_id, file_record.relative_path,
                            "generic_window", str(window_index),
                        ),
                        project_id=file_record.project_id,
                        relative_path=file_record.relative_path,
                        chunk_type="generic_window",
                        source_node_id=None,
                        text=text,
                        line_start=start + 1,
                        line_end=end,
                    )
                )
                window_index += 1
            if end == len(lines):
                break
            start += step

        return chunks
