"""
Default chunker registry assembly and end-to-end pipeline function.

Coverage as of this delivery: SystemVerilog/UVM/assertions and Python get
bespoke, structure-aware chunkers (the two highest-value languages, same
prioritization Part 3 used). Markdown gets a bespoke section-aware
chunker despite having no IR node, because heading-based sectioning is
cheap and high-value for spec documents. Logs get an event-aware chunker
(see `log_chunker.py`) that keeps multi-line events and their
continuation together and groups repetitive single-line events into
bounded windows. Everything else (VHDL, TCL, Shell, JSON/YAML/CSV/XML,
PDF, Excel) falls back to `GenericWindowChunker` — flagged explicitly,
same pattern as Part 3's "broadened later" note, not silently accepted
as sufficient forever.
"""
from __future__ import annotations

from pathlib import Path

from semrag.ingestion.chunking.generic_window_chunker import GenericWindowChunker
from semrag.ingestion.chunking.interfaces import ChunkerRegistry
from semrag.ingestion.chunking.log_chunker import LogChunker
from semrag.ingestion.chunking.markdown_chunker import MarkdownChunker
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.chunking.python_chunker import PythonChunker
from semrag.ingestion.chunking.systemverilog_chunker import SystemVerilogChunker
from semrag.ingestion.ir.models import IRGraph
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.classification import language_for_extension
from semrag.ingestion.scanner.models import FileCategory, FileRecord
from semrag.retrieval.source_authority import classify_document


def build_default_registry() -> ChunkerRegistry:
    registry = ChunkerRegistry()

    sv_chunker = SystemVerilogChunker()
    for category in (FileCategory.RTL_VERILOG, FileCategory.TESTBENCH_UVM, FileCategory.ASSERTION):
        registry.register(category, sv_chunker)

    registry.register(FileCategory.PYTHON, PythonChunker())
    registry.register(FileCategory.MARKDOWN, MarkdownChunker())
    registry.register(FileCategory.LOG, LogChunker())
    registry.register_fallback(GenericWindowChunker())

    return registry


def _enrich_chunk(
    chunk: Chunk,
    record: FileRecord,
    parse_result: ParseResult | None,
    source_type: str = "",
    source_root: str | None = None,
) -> Chunk:
    """Fill in RI-06 metadata centrally, once, using the canonical single
    source for each field — never recalculated downstream: language from
    the scanner's extension mapping, category from the existing Source
    Authority classifier (reused, not duplicated), parser from the
    ParseResult that actually ran. ``chunker_name`` is already stamped by
    the registry; ``symbol_name`` by the chunkers. ``source_type`` /
    ``source_root`` come from the caller's ``ProjectPaths`` when a
    multi-root project is being indexed ('' / None for legacy single-root)."""
    return chunk.model_copy(
        update={
            "language": language_for_extension(record.relative_path),
            "category": classify_document(chunk.chunk_type, chunk.relative_path),
            "parser_name": parse_result.parser_name if parse_result else None,
            "source_type": source_type,
            "source_root": source_root,
        }
    )


def chunk_project(
    root_path: str,
    file_records: dict[str, FileRecord],
    parse_results: dict[str, ParseResult],
    ir_graph: IRGraph,
    registry: ChunkerRegistry | None = None,
    max_workers: int = 1,
    project_paths=None,
) -> list[Chunk]:
    """Chunk every file in the project. ``max_workers=1`` keeps behavior
    identical to the original sequential pass; values >1 run reads and
    chunking across a thread pool — disk I/O bound, threads (not
    processes) avoid pickling overhead. ``project_paths`` (a
    ``core.project_paths.ProjectPaths``) is optional; when provided the
    chunks are stamped with the source area type/root for multi-root
    projects."""
    registry = registry or build_default_registry()

    nodes_by_path: dict[str, list] = {}
    for node in ir_graph.nodes.values():
        nodes_by_path.setdefault(node.relative_path, []).append(node)

    def chunk_one(relative_path: str, record: FileRecord) -> list[Chunk]:
        # The scanner's absolute_path is authoritative for every project
        # shape — a legacy single-root project resolves under ``root``,
        # a multi-root project resolves under the appropriate area root.
        absolute_path = Path(record.absolute_path)
        parse_result = parse_results.get(relative_path)
        ir_nodes = nodes_by_path.get(relative_path, [])

        # Binary files (PDFs, Excels, waveforms, etc.) must NOT be read as
        # raw UTF-8 text: the bytes contain NULs and other control bytes
        # that crash psycopg2 binding (NUL in particular — "A string
        # literal cannot contain NUL (0x00) characters") and the resulting
        # text is never semantically useful anyway. No current chunker
        # consumes the parser's structured payload for these categories,
        # so the right move is to skip them here — a future
        # structure-aware PDF/Excel chunker that reads `parse_result.payload`
        # would lift this guard for the categories it handles.
        if record.is_binary:
            return []

        try:
            file_text = absolute_path.read_text(errors="replace")
        except (FileNotFoundError, PermissionError):
            return []

        source_type = ""
        source_root = None
        if project_paths is not None:
            area = project_paths.area_for(absolute_path)
            if area is not None:
                source_type = area
                source_root = str(project_paths.roots()[area])

        chunks = registry.chunk_file(record, parse_result, file_text, ir_nodes)
        return [
            _enrich_chunk(c, record, parse_result, source_type, source_root)
            for c in chunks
        ]

    items = list(file_records.items())
    if max_workers <= 1 or len(items) <= 1:
        all_chunks: list[Chunk] = []
        for relative_path, record in items:
            all_chunks.extend(chunk_one(relative_path, record))
        return all_chunks

    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(chunk_one, rel, rec): rel for rel, rec in items
        }
        for fut in as_completed(futures):
            all_chunks.extend(fut.result())
    return all_chunks
