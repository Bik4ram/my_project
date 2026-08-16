"""
Tests for the extracted intent-ranking module.

The intent boost stage was pure-refactored out of
``retrieval/intent.py`` into ``retrieval/ranking/intent.py`` (renamed
``apply_intent_boost`` -> ``apply_intent_ranking``). Detection
(``QueryIntent``, ``classify_intent``) stays in ``retrieval/intent.py``;
these tests pin the extracted function to the exact pre-refactor inline
logic and cover the ranking rules.
"""
from __future__ import annotations

import pytest

from semrag.retrieval.intent import QueryIntent
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.intent import apply_intent_ranking, boost_for_intent


def _result(
    chunk_id: str,
    chunk_type: str,
    fused_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        chunk_type=chunk_type,
        relative_path="rtl/x.sv",
        text="module x; endmodule",
        fused_score=fused_score,
        signal_scores={"rrf": fused_score},
    )


# ── Regression: identical to the pre-refactor inline logic ───────────────


class TestMatchesPreRefactorInlineLogic:
    """The intent boost used to live in ``retrieval/intent.py`` as
    ``apply_intent_boost``. ``apply_intent_ranking`` must reproduce it
    exactly: same scores, same order, same signal_scores."""

    def _pre_refactor_inline(
        self,
        results,
        intent,
        boost_table=None,
        *,
        max_boost=0.5,
    ):
        # The exact code that lived in retrieval/intent.py before RF-03.
        table = dict(boost_table) if boost_table is not None else dict(boost_table or boost_for_intent(intent))
        if not table:
            return results

        boosted = []
        for result in results:
            boost = 0.0
            matched_key = None
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

    def test_identical_for_architecture_intent(self):
        results = [
            _result("a", "sv_module", fused_score=0.5),
            _result("b", "log", fused_score=0.5),
            _result("c", "sv_assertion", fused_score=0.5),
        ]
        expected = self._pre_refactor_inline(
            [r.model_copy(deep=True) for r in results], QueryIntent.ARCHITECTURE
        )
        got = apply_intent_ranking(
            [r.model_copy(deep=True) for r in results], QueryIntent.ARCHITECTURE
        )

        assert [r.chunk_id for r in got] == [r.chunk_id for r in expected]
        assert [r.fused_score for r in got] == pytest.approx(
            [r.fused_score for r in expected]
        )
        for got_r, exp_r in zip(got, expected, strict=True):
            assert got_r.signal_scores == exp_r.signal_scores

    def test_identical_with_custom_boost_table(self):
        results = [
            _result("a", "sv_module", fused_score=0.5),
            _result("b", "log", fused_score=0.5),
        ]
        table = {"sv_module": 0.4}
        expected = self._pre_refactor_inline(
            [r.model_copy(deep=True) for r in results],
            QueryIntent.ARCHITECTURE,
            boost_table=table,
        )
        got = apply_intent_ranking(
            [r.model_copy(deep=True) for r in results],
            QueryIntent.ARCHITECTURE,
            boost_table=table,
        )

        assert [r.chunk_id for r in got] == [r.chunk_id for r in expected]
        assert [r.fused_score for r in got] == pytest.approx(
            [r.fused_score for r in expected]
        )

    def test_identical_with_max_boost_cap(self):
        results = [_result("a", "sv_module", fused_score=0.5)]
        expected = self._pre_refactor_inline(
            [r.model_copy(deep=True) for r in results],
            QueryIntent.ARCHITECTURE,
            max_boost=0.2,
        )
        got = apply_intent_ranking(
            [r.model_copy(deep=True) for r in results],
            QueryIntent.ARCHITECTURE,
            max_boost=0.2,
        )
        assert [r.fused_score for r in got] == pytest.approx(
            [r.fused_score for r in expected]
        )


# ── Ranking rules ────────────────────────────────────────────────────────


class TestIntentRankingRules:
    def test_boosts_matching_chunks_only(self):
        results = [
            _result("a", "sv_module", fused_score=0.5),
            _result("b", "log", fused_score=0.5),
        ]
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        by_id = {r.chunk_id: r for r in boosted}
        assert by_id["a"].fused_score > 0.5
        assert by_id["a"].signal_scores["intent_boost"] == pytest.approx(0.30)
        assert by_id["b"].fused_score == pytest.approx(0.5)
        assert "intent_boost" not in by_id["b"].signal_scores

    def test_general_is_noop(self):
        results = [
            _result("a", "sv_module", fused_score=0.9),
            _result("b", "log", fused_score=0.5),
        ]
        boosted = apply_intent_ranking(results, QueryIntent.GENERAL)
        assert [r.fused_score for r in boosted] == [0.9, 0.5]

    def test_max_boost_cap_applied(self):
        results = [_result("a", "sv_module", fused_score=0.5)]
        boosted = apply_intent_ranking(
            results, QueryIntent.ARCHITECTURE, max_boost=0.2
        )
        assert boosted[0].fused_score == pytest.approx(0.5 * 1.2)

    def test_longest_prefix_match_wins(self):
        results = [
            _result("a", "sv_module_signature", fused_score=0.5),
            _result("b", "sv_module", fused_score=0.5),
        ]
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        sig_boost = next(
            r.signal_scores.get("intent_boost", 0) for r in boosted if r.chunk_id == "a"
        )
        plain_boost = next(
            r.signal_scores.get("intent_boost", 0) for r in boosted if r.chunk_id == "b"
        )
        assert sig_boost == pytest.approx(0.25)
        assert plain_boost == pytest.approx(0.30)

    def test_re_sorts_descending(self):
        results = [
            _result("a", "sv_module", fused_score=0.020),
            _result("b", "log", fused_score=0.050),
        ]
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        # log untouched at 0.050; sv_module -> 0.020 * 1.3 = 0.026.
        assert [r.chunk_id for r in boosted] == ["b", "a"]


# ── Boost table accessor ─────────────────────────────────────────────────


class TestBoostForIntent:
    def test_each_intent_returns_table(self):
        for intent in QueryIntent:
            assert isinstance(boost_for_intent(intent), dict)

    def test_general_returns_empty(self):
        assert boost_for_intent(QueryIntent.GENERAL) == {}

    def test_arch_maps_expected_types(self):
        table = boost_for_intent(QueryIntent.ARCHITECTURE)
        assert table["sv_module"] == pytest.approx(0.30)
        assert "log" not in table

    def test_debug_maps_logs_and_zero_boost_entries(self):
        table = boost_for_intent(QueryIntent.DEBUG)
        assert table["log"] == pytest.approx(0.30)
        assert table["sv_body_window"] == 0.0

