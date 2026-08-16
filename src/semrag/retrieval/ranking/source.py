"""
Source-based ranking adjustments.

Downweights and reorders chunks based on where they come from (their
source file / document), independent of the query's lexical and vector
signal. Two source-based adjustments are orchestrated here:

- **Noisy-source penalty**: verbose transcript / log files (simulation
  logs, waveform dumps) over-match natural-language queries on both
  retrieval signals, so chunks from them are downweighted before graph
  expansion / intent / rerank so they can't crowd real source files out
  of the final top-k. The matching terms and penalty weight are caller-
  configured (the pipeline exposes ``noisy_source_path_terms`` and
  ``noisy_source_penalty``).
- **Source authority**: the document-aware ranking adjustment (RI-02).
  A chunk's source document is classified (path + ``chunk_type``) into
  a semantic category whose authority weight nudges the fused score.
  The classification rules and weights live in
  ``source_authority.py``, configurable via ``SourceAuthorityConfig``;
  this module only orchestrates applying them during ranking.

Both adjustments share the pipeline's ordering invariant: the result
list is re-sorted by ``fused_score`` descending.
"""
from __future__ import annotations

from semrag.retrieval.doc_type_prior import DEFAULT_MAX_BOOST
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.source_authority import (
    DEFAULT_SOURCE_AUTHORITY,
    SourceAuthorityConfig,
    apply_source_authority,
)

# Default noisy-source matching terms, mirroring the pipeline's
# ``noisy_source_path_terms`` constructor default.
DEFAULT_NOISY_SOURCE_PATH_TERMS: tuple[str, ...] = ("/log/",)


def _is_noisy_source(relative_path: str, terms: tuple[str, ...]) -> bool:
    """True when a chunk's source file is a verbose transcript / log whose
    lexical content is dominated by repeated query keywords (simulation
    logs, waveform dumps). These crowd out real source files in RRF."""
    return any(term in relative_path for term in terms)


def apply_source_ranking(
    results: list[RetrievalResult],
    *,
    noisy_source_path_terms: tuple[str, ...] = DEFAULT_NOISY_SOURCE_PATH_TERMS,
    noisy_source_penalty: float = 0.0,
    enable_doc_type_prior: bool = True,
    doc_type_max_boost: float = DEFAULT_MAX_BOOST,
    source_authority_config: SourceAuthorityConfig | None = None,
) -> list[RetrievalResult]:
    """Apply source-based ranking adjustments to ``results``.

    Two adjustments compose here, matching the pipeline's stage order:

    1. Noisy-source penalty (only when ``noisy_source_penalty > 0``):
       chunks whose ``relative_path`` contains any of
       ``noisy_source_path_terms`` are downweighted by
       ``(1 - noisy_source_penalty)`` and the list is re-sorted. Mutates
       ``results`` in place, mirroring the original pipeline stage.
    2. Source authority (when ``enable_doc_type_prior`` is true):
       delegated to ``source_authority.apply_source_authority``, which
       returns a new re-sorted list. When ``source_authority_config`` is
       given it is used verbatim; otherwise the default authority config
       is used with ``doc_type_max_boost`` as its magnitude cap. The
       classification rules and weights stay in ``source_authority.py``.

    Pass ``enable_doc_type_prior=False`` to use this purely as the
    noisy-source penalty stage; leave the defaults to use it as the
    source-authority stage.
    """
    if noisy_source_penalty > 0.0:
        for r in results:
            if _is_noisy_source(r.relative_path, noisy_source_path_terms):
                r.fused_score = r.fused_score * (1.0 - noisy_source_penalty)
        results.sort(key=lambda r: r.fused_score, reverse=True)

    if enable_doc_type_prior:
        config = source_authority_config
        if config is None:
            config = DEFAULT_SOURCE_AUTHORITY.with_max_boost(doc_type_max_boost)
        return apply_source_authority(results, config=config)
    return results