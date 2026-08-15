"""
Reciprocal Rank Fusion (RRF).

RRF combines ranked lists using each item's *rank position* within each
list, not the raw scores — deliberately, because vector cosine similarity
and Postgres's `ts_rank_cd` are not on comparable scales (one is bounded
[-1, 1], the other is an unbounded relevance weight), so averaging or
summing raw scores directly would let whichever signal happens to produce
larger numbers dominate the fusion regardless of actual relevance. RRF
sidesteps that entirely: `score = sum(1 / (k + rank))` across every list a
chunk appears in, where `rank` is 1-indexed position. A chunk ranked #1 by
vector search and #1 by lexical search scores highest; a chunk appearing
in only one list still contributes, just less.

`k=60` is the standard constant from the original RRF paper (Cormack et
al., 2009) and the value most hybrid-search systems default to — kept as
the default here rather than tuned, since tuning it meaningfully requires
real query/relevance data this system doesn't have yet (Part 15's job).
"""
from __future__ import annotations

from semrag.retrieval.models import RetrievalResult

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievalResult]], k: int = DEFAULT_RRF_K
) -> list[RetrievalResult]:
    """`result_lists` maps a signal name (e.g. "vector", "lexical") to its
    ranked results — order within each list matters, contents are used
    only for rank position and to merge each chunk's `signal_scores`."""
    fused: dict[str, RetrievalResult] = {}
    rrf_scores: dict[str, float] = {}

    for signal_name, results in result_lists.items():
        for rank, result in enumerate(results, start=1):
            contribution = 1.0 / (k + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + contribution

            if result.chunk_id not in fused:
                fused[result.chunk_id] = result.model_copy(deep=True)
            else:
                fused[result.chunk_id].signal_scores.update(result.signal_scores)

    for chunk_id, result in fused.items():
        result.fused_score = rrf_scores[chunk_id]

    return sorted(fused.values(), key=lambda r: r.fused_score, reverse=True)
