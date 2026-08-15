"""
Vector store data models.

`VectorChunk` is deliberately not the same object as Part 4's `IRNode` or
Part 8's future chunk object — it's the minimal shape this storage layer
needs: an id, an embedding, and a filter-relevant payload. Part 8
(chunking, not yet built) is what decides *what* text becomes a chunk and
produces these objects; this module only knows how to store and query
them once produced. `chunk_id` is expected to be deterministic (derived
from the source IR node/file the same way Part 4's node IDs are) so
re-embedding an unchanged chunk is an upsert here too, not a duplicate —
consistent with every other storage layer in this series.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class VectorChunk(BaseModel):
    chunk_id: str
    project_id: str
    relative_path: str
    chunk_type: str = Field(
        ..., description="e.g. 'sv_module', 'py_function', 'markdown_section' — Part 8 defines the vocabulary."
    )
    source_node_id: str | None = Field(
        default=None, description="Part 4 IRNode.id this chunk was derived from, if any."
    )
    text: str = Field(..., description="The chunk's source text — kept for citation/grounding and BM25 (Part 10).")
    embedding: list[float]
    line_start: int | None = None
    line_end: int | None = None
    # RI-06 retrieval metadata — mirror of the Chunk fields, carried
    # through storage so retrieval paths can expose them without a second
    # source of truth. Additive; defaults keep old construction sites and
    # pre-migration DB rows compatible.
    language: str = ""
    category: str = ""
    parser_name: str | None = None
    chunker_name: str = ""
    symbol_name: str | None = None


class VectorSearchFilter(BaseModel):
    project_id: str | None = None
    chunk_type: str | None = None
    relative_path: str | None = None


class VectorSearchResult(BaseModel):
    chunk_id: str
    score: float
    project_id: str
    relative_path: str
    chunk_type: str
    text: str
    line_start: int | None = None
    line_end: int | None = None
    # RI-06 metadata, additive.
    language: str = ""
    category: str = ""
    parser_name: str | None = None
    chunker_name: str = ""
    symbol_name: str | None = None
