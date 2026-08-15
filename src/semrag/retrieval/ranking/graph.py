"""
Graph-expansion ranking adjustments.

Assigns scored graph-expansion chunks (see ``graph_expansion.py``) their
final rank positions, then merges them into the fused ranking and re-sorts.

NOTE: we intentionally do NOT run a second RRF fusion after graph
expansion. The first RRF already combined vector + lexical signals into
meaningful scores. A second RRF would recalculate ALL scores from scratch
using only rank position, which:
  (a) loses the carefully-combined vector+lexical signal,
  (b) compresses the score range (~0.006-0.033 → ~0.010-0.016),
  (c) lets graph expansion chunks (typically 5-15) compete at the same
      level as top base chunks,
  (d) makes intent boost and reranker ineffective on the compressed range.

Instead, each expansion inherits the RRF score of the base seed that
produced it (recorded as ``graph_seed_score``), decayed only by graph
depth. A graph expansion is therefore anchored to *its own* seed's
relevance, not to the single strongest base chunk: a relevant structural
context pulled in by a strong seed can compete fairly with (and even
outrank) a weak base chunk, while context from an irrelevant seed stays
low. Depth attenuation (``1/(depth+1)``: depth-1 -> 0.5, depth-2 -> 0.33)
keeps shallower, directly-related expansions above deeper ones.
"""
from __future__ import annotations

from semrag.retrieval.models import RetrievalResult


def apply_graph_expansion_ranking(
    fused: list[RetrievalResult],
    expansions: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Merge graph-expansion chunks into the fused ranking.

    Each expansion's ``fused_score`` is overwritten with
    ``graph_seed_score * depth_weight`` — the RRF score of the base seed
    that produced it (see module docstring) times the depth-decay weight
    (``1/(depth+1)``, read from ``signal_scores["graph_expansion"]``).
    Expansions are then appended to ``fused`` and the combined list is
    re-sorted descending. Base chunk scores are left untouched. Mutates
    ``fused`` in place and returns it, mirroring the original pipeline
    stage.
    """
    if not expansions:
        return fused
    for exp in expansions:
        depth_weight = exp.signal_scores.get("graph_expansion", 0.5)
        seed_score = exp.signal_scores.get("graph_seed_score", 0.0)
        exp.fused_score = seed_score * depth_weight
    fused.extend(expansions)
    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused
