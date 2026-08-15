"""
Citation formatting.

Every chunk included in a prompt is prefixed with a citation header
(`relative_path`, and `line_start`-`line_end` when known) — this is what
lets a downstream LLM response point back to an exact source location
("see RTL/i2c_master.sv, lines 12-18") instead of an unattributed
paraphrase, satisfying Part 1's "citation generation" and "grounding"
requirements concretely rather than leaving them as an aspiration for
Part 13's LLM layer to somehow retrofit. The citation format is
deliberately plain and consistent (not the chunk_id, which is an internal
identifier meaningless to an engineer reading the response) — a file path
and line range is what someone would actually open their editor to.
"""
from __future__ import annotations

from semrag.retrieval.models import RetrievalResult


def format_citation_header(result: RetrievalResult) -> str:
    if result.line_start is not None and result.line_end is not None:
        return f"[Source: {result.relative_path}, lines {result.line_start}-{result.line_end}]"
    return f"[Source: {result.relative_path}]"


def format_chunk_with_citation(result: RetrievalResult) -> str:
    return f"{format_citation_header(result)}\n{result.text}"
