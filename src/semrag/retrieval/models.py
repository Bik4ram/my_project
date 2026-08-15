"""
Shared models for the retrieval plane.

`RetrievalResult` is deliberately similar to Part 6's `VectorSearchResult`
but distinct — it carries a `signal_scores` breakdown (which retrieval
method(s) contributed to this hit, and with what score) because Part 11's
future prompt-construction layer, and anyone debugging retrieval quality,
needs to see *why* a chunk was retrieved, not just that it was. A pure
vector or lexical result loses that provenance the moment it's fused with
another signal; keeping it is cheap and valuable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    chunk_id: str
    project_id: str
    relative_path: str
    chunk_type: str
    text: str
    source_node_id: str | None = Field(
        default=None,
        description="Part 4 IRNode.id this chunk derives from, when known — used by graph_expansion.py "
        "to pull in structurally related chunks. None for chunk categories without an IR node.",
    )
    line_start: int | None = None
    line_end: int | None = None
    # RI-06 retrieval metadata — factual, additive, single-source-per-field
    # (set in the ingestion pipeline and carried through storage). Exposed
    # here so prompt construction / evaluation can distinguish an RTL module
    # from a UVM class from a sim log without re-deriving it from text.
    language: str = ""
    category: str = ""
    parser_name: str | None = None
    chunker_name: str = ""
    symbol_name: str | None = None
    fused_score: float
    signal_scores: dict[str, float | str] = Field(
        default_factory=dict,
        description="e.g. {'vector': 0.82, 'lexical': 0.41, 'graph_expansion': 1.0, "
        "'graph_direction': 'dependencies'}. Numeric keys are retrieval weights; "
        "non-numeric keys (e.g. graph_direction) are additive metadata only.",
    )

    @property
    def is_generated(self) -> bool:
        """Derived — reads the canonical Source Authority category, never
        a second stored classifier (see ingestion Chunk.is_generated)."""
        return self.category == "generated"
