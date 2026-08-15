"""
Chunk model.

Deliberately not Part 6's `VectorChunk` — this is `VectorChunk` minus
`embedding`, because chunking (splitting source into retrieval-sized,
citable units) and embedding (turning a chunk's text into a vector) are
separate concerns with separate costs: chunking is cheap and
deterministic; embedding requires calling a model. Keeping them as
different types means Part 9 can re-embed every `Chunk` unchanged (e.g.
after switching embedding models — Part 1's "embedding versioning"
requirement) without re-running chunking, and re-chunking (e.g. after a
chunking-strategy change) doesn't require having an embedding model
available at all. `Chunk.chunk_id` uses the same deterministic-ID
philosophy as every other part of this system — content-derived, not
random — for the same upsert-not-duplicate reason.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    project_id: str
    relative_path: str
    chunk_type: str = Field(..., description="Vocabulary defined per-language by each chunker in this package.")
    source_node_id: str | None = Field(
        default=None,
        description="Part 4 IRNode.id this chunk was derived from, when one exists. None for file "
        "categories Part 4's IR builder doesn't create nodes for (see this part's design doc, Section 2).",
    )
    text: str
    line_start: int | None = None
    line_end: int | None = None
    # ------------------------------------------------------------------
    # RI-06 retrieval metadata — additive, factual, single-source-per-field.
    # Defaults keep pre-RI-06 construction sites and persisted rows
    # compatible; the ingestion pipeline (`chunk_project`) fills these in.
    # ------------------------------------------------------------------
    language: str = Field(
        default="",
        description="Canonical language from scanner/classification.language_for_extension(), "
        "e.g. 'systemverilog', 'python', 'tcl', 'markdown', 'log'.",
    )
    category: str = Field(
        default="",
        description="Source Authority's classify_document(chunk_type, relative_path) — the "
        "existing authority-category system reused as the semantic document category, e.g. "
        "'rtl', 'uvm', 'specification', 'simulation_log', 'generated'. Never classified twice.",
    )
    parser_name: str | None = Field(
        default=None,
        description="ParseResult.parser_name of the parser that produced the structured payload "
        "this chunk was derived from, when one ran. None for categories without a parser.",
    )
    chunker_name: str = Field(
        default="",
        description="Name of the chunker that produced this chunk, stamped by ChunkerRegistry "
        "(the one place that knows which chunker actually ran, including the fallback).",
    )
    symbol_name: str | None = Field(
        default=None,
        description="Module/class/function name when the parser/chunker determined it reliably "
        "(e.g. SystemVerilog module, Python function/class). None for generic text/log/markdown "
        "chunks — never heuristically invented.",
    )

    @property
    def is_generated(self) -> bool:
        """Derived — NOT a second stored classifier. A chunk is generated
        when its canonical Source Authority category says so; the category
        is the single source of truth and this property just reads it."""
        return self.category == "generated"
