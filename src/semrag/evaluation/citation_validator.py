"""
Citation validator.

Real faithfulness/hallucination measurement (Part 1's "faithfulness",
"groundedness", "hallucination rate" metrics) needs an LLM response to
judge and, ideally, a human or a second LLM to judge it against - neither
of which is available without a live model in this sandbox (same
limitation as Parts 9/10/13). What IS checkable without a live model,
and genuinely useful on its own: whether an LLM response's citations are
even real - i.e., whether every [Source: path, lines X-Y] the response
claims actually corresponds to a chunk that was in the context it was
given, not a fabricated file/line reference. A response citing a file
that was never in its context is unambiguous evidence of a problem,
regardless of whether the surrounding prose is otherwise accurate - this
is a necessary-but-not-sufficient groundedness check, not a full
faithfulness evaluator, and is presented as exactly that, not more.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from semrag.retrieval.models import RetrievalResult

_CITATION_PATTERN = re.compile(r"\[Source:\s*([^,\]]+?)(?:,\s*lines\s+(\d+)-(\d+))?\]")


class CitationValidationResult(BaseModel):
    cited_paths: set[str]
    valid_citations: set[str]
    fabricated_citations: set[str]

    @property
    def has_fabricated_citations(self) -> bool:
        return len(self.fabricated_citations) > 0

    @property
    def fabrication_rate(self) -> float:
        if not self.cited_paths:
            return 0.0
        return len(self.fabricated_citations) / len(self.cited_paths)


def validate_citations(response_text: str, included_chunks: list[RetrievalResult]) -> CitationValidationResult:
    available_paths = {chunk.relative_path for chunk in included_chunks}
    cited_paths = {match.group(1).strip() for match in _CITATION_PATTERN.finditer(response_text)}

    valid = {path for path in cited_paths if path in available_paths}
    fabricated = cited_paths - valid

    return CitationValidationResult(cited_paths=cited_paths, valid_citations=valid, fabricated_citations=fabricated)
