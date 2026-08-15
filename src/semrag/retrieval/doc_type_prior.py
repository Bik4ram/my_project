"""
Document-type-aware ranking prior (RI-01) — now a compatibility shim.

RI-02 replaced the flat prior tables that lived here with the Source
Authority model in ``source_authority.py``: a single configurable
``SourceAuthorityConfig`` that separates document classification (path +
``chunk_type`` rules) from authority weights, records the adjustment
under a canonical ``signal_scores["source_authority"]`` key, and keeps
writing the legacy ``signal_scores["doc_type_prior"]`` key so the
pipeline trace and the RB-02 / RB-03 analytics are unchanged.

This module exists so existing imports keep working:

- ``apply_doc_type_prior`` — now delegates to
  ``apply_source_authority`` with the default authority config (whose
  weights are the verbatim translation of the RI-01 tables).
- ``DEFAULT_MAX_BOOST`` — the default authority ``max_boost`` (0.5).

Behavior is unchanged: path-based categories win over generic
``chunk_type``; weights are small multiplicative nudges bounded by
``max_boost``; results are re-sorted by ``fused_score`` desc; chunks
with a zero prior are left untouched (no signal key).
"""
from __future__ import annotations

from semrag.retrieval.source_authority import (
    DEFAULT_SOURCE_AUTHORITY,
    apply_source_authority,
)

# Hard cap on the multiplicative boost. Matches intent's ``max_boost=0.5``
# so a stacked prior + intent boost cannot exceed ~1.5x and dominate
# the underlying RRF signal. See retrieval/intent.py for the rationale.
DEFAULT_MAX_BOOST = DEFAULT_SOURCE_AUTHORITY.max_boost


def apply_doc_type_prior(
    results: list,
    *,
    max_boost: float = DEFAULT_MAX_BOOST,
) -> list:
    """Apply the document-type prior to each result (RI-01 API).

    Delegates to ``source_authority.apply_source_authority`` with the
    default authority config, overriding only ``max_boost``. The prior
    is recorded under ``signal_scores["doc_type_prior"]`` (and the
    canonical ``signal_scores["source_authority"]`` key) so the trace
    can show its contribution alongside ``intent_boost``.
    """
    config = DEFAULT_SOURCE_AUTHORITY.with_max_boost(max_boost)
    return apply_source_authority(results, config=config)