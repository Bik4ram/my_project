"""
Deterministic chunk IDs and the shared line-range extraction helper every
chunker in this package uses.

Same content-derived-ID philosophy as `id_scheme.py` (Part 4): a chunk ID
is a pure function of its source (project, file, chunk kind, and a
disambiguator — usually a name or line range), never random, so re-running
the chunker on an unchanged file produces byte-identical chunk IDs and
Part 6's `PgvectorStore.upsert_chunks` naturally upserts instead of
duplicating.
"""
from __future__ import annotations


def chunk_id(project_id: str, relative_path: str, chunk_kind: str, disambiguator: str) -> str:
    return "::".join([project_id, relative_path, chunk_kind, disambiguator])


def extract_lines(file_text: str, start_line: int, end_line: int) -> str:
    """1-indexed, inclusive on both ends — matching the line-number
    convention Part 3's parsers already use (pyslang's `getLineNumber`,
    Python's `ast` line numbers, and every regex-structural parser's
    `text.count("\\n", 0, match.start()) + 1` all produce 1-indexed
    line numbers)."""
    lines = file_text.splitlines()
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    return "\n".join(lines[start:end])
