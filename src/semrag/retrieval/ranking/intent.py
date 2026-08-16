"""
Intent-based ranking adjustments.

Applies a small multiplicative boost to chunks whose ``chunk_type``
matches the intent detected for the query. The detection side
(``QueryIntent``, ``classify_intent``, the keyword tables, structural
query hints) lives in ``retrieval/intent.py`` and is untouched; this
module owns ONLY the ranking decision — the per-intent boost weights
and how they are applied during ranking.

This is **not** a hard filter — every chunk still gets through if it's
the best match. It's a multiplicative score boost keyed on (intent,
chunk_type): architecture intent adds to ``sv_module``/``sv_package``/
etc., debug intent adds to logs and waveforms. The boost is small
(default 0.2–0.3 per chunk-type bucket) so that a strong genuine match
in an unboosted category still wins; the intent layer only matters near
the boundary of relevance, where it nudges ordering toward the question's
true topic. Tuning the magnitudes is the evaluation harness's job —
intentionally not hard-coded here as "no boost > 0.5" so that a future
evaluation finding that boosts should be larger has a place to land.
"""
from __future__ import annotations

from collections.abc import Mapping

from semrag.retrieval.intent import QueryIntent
from semrag.retrieval.models import RetrievalResult

# Lookup of (chunk_type prefix -> boost). Multiple boosts can stack for
# chunks that fall into several categories (e.g. a log file mentioning
# a module name is unlikely; for an RTL chunk that has assertion windows,
# both the architecture boost and a small verification boost are reasonable).
_DEFAULT_BOOST_TABLE: dict[QueryIntent, Mapping[str, float]] = {
    QueryIntent.ARCHITECTURE: {
        "sv_module": 0.30,
        "sv_module_signature": 0.25,
        "sv_body_window": 0.15,
        "sv_package": 0.20,
        "sv_assertion": 0.10,
        "v_module": 0.30,
        "v_body_window": 0.15,
        "markdown_section": 0.05,
    },
    QueryIntent.VERIFICATION: {
        "sv_assertion": 0.25,
        "sv_module_signature": 0.15,
        "sv_module": 0.10,
        "sv_package": 0.15,
        "sv_body_window": 0.05,
        "py_function": 0.10,
        "py_class": 0.15,
        "py_method": 0.10,
    },
    QueryIntent.DEBUG: {
        "log": 0.30,
        "coverage_report": 0.20,
        "lint_report": 0.20,
        "timing_report": 0.15,
        "waveform": 0.20,
        "sv_assertion": 0.05,  # failures point at assertions
        "sv_body_window": 0.0,
    },
    QueryIntent.GENERAL: {},
}


def boost_for_intent(intent: QueryIntent) -> Mapping[str, float]:
    return _DEFAULT_BOOST_TABLE.get(intent, {})


def apply_intent_ranking(
    results: list[RetrievalResult],
    intent: QueryIntent,
    boost_table: Mapping[str, float] | None = None,
    *,
    max_boost: float = 0.5,
) -> list[RetrievalResult]:
    """Apply the intent-keyed boost to each result's ``fused_score``.

    A multiplier in (1.0, 1.0 + max_boost) is applied; results outside
    the boosted categories are untouched. The cap (``max_boost``) keeps
    even a maximally-boosted chunk from doubling its score relative to a
    neutral competitor — that magnitude of swing belongs in a learned
    re-ranker, not a keyword heuristic.
    """
    table = dict(boost_table) if boost_table is not None else dict(boost_table or boost_for_intent(intent))
    if not table:
        return results

    boosted: list[RetrievalResult] = []
    for result in results:
        # Longest matching chunk-type prefix wins (so ``sv_body_window``
        # doesn't get the small ``sv_module`` boost when both prefixes
        # could match - ``sv_module`` is a strict substring of
        # ``sv_module_signature``).
        boost = 0.0
        matched_key: str | None = None
        for key, value in table.items():
            if result.chunk_type.startswith(key) and (matched_key is None or len(key) > len(matched_key)):
                boost = value
                matched_key = key

        if boost <= 0.0:
            boosted.append(result)
            continue

        boost = min(boost, max_boost)
        updated = result.model_copy(deep=True)
        updated.signal_scores["intent_boost"] = boost
        updated.fused_score = result.fused_score * (1.0 + boost)
        boosted.append(updated)

    boosted.sort(key=lambda r: r.fused_score, reverse=True)
    return boosted
