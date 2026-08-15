"""
Tests for the graph-expansion ranking module.

The graph-expansion scoring stage assigns each expansion a seed-relative
score: the RRF score of the base seed that produced it
(``signal_scores["graph_seed_score"]``) decayed by graph depth
(``signal_scores["graph_expansion"]`` = ``1/(depth+1)``). These tests pin
the scoring rule and its invariants so a future change to the module can't
silently alter ranking behavior.
"""
from __future__ import annotations

import pytest

from semrag.retrieval.fusion import reciprocal_rank_fusion
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.graph import apply_graph_expansion_ranking


def _result(
    chunk_id: str,
    fused_score: float = 0.0,
    signal_scores: dict[str, float] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="p1",
        relative_path="RTL/m.sv",
        chunk_type="sv_module",
        text="module x; endmodule",
        fused_score=fused_score,
        signal_scores=signal_scores or {},
    )


def _expansion(chunk_id: str, depth_weight: float, seed_score: float) -> RetrievalResult:
    return _result(
        chunk_id,
        fused_score=0.5,
        signal_scores={"graph_expansion": depth_weight, "graph_seed_score": seed_score},
    )


class TestApplyGraphExpansionRanking:
    def test_matches_seed_relative_scoring_rule(self):
        """Each expansion's score must be ``graph_seed_score * depth_weight``,
        merge order and re-sort must match the pipeline stage."""
        base = [
            _result("base_top", fused_score=0.032),
            _result("base_mid", fused_score=0.020),
            _result("base_low", fused_score=0.010),
        ]
        expansions = [
            _expansion("exp_d1", 0.5, 0.032),
            _expansion("exp_d2", 0.33, 0.020),
        ]

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )

        expected = [r.model_copy(deep=True) for r in base]
        for e in expansions:
            dw = e.signal_scores.get("graph_expansion", 0.5)
            seed = e.signal_scores.get("graph_seed_score", 0.0)
            e.fused_score = seed * dw
        expected.extend(expansions)
        expected.sort(key=lambda r: r.fused_score, reverse=True)

        assert [r.chunk_id for r in fused] == [r.chunk_id for r in expected]
        assert [r.fused_score for r in fused] == pytest.approx([r.fused_score for r in expected])

    def test_base_chunk_scores_preserved(self):
        base = [_result("base_top", fused_score=0.032)]
        fused = apply_graph_expansion_ranking(base, [_expansion("exp_d1", 0.5, 0.032)])
        base_result = next(r for r in fused if r.chunk_id == "base_top")
        assert base_result.fused_score == pytest.approx(0.032)

    def test_empty_expansions_is_noop(self):
        base = [_result("base_top", fused_score=0.032)]
        fused = apply_graph_expansion_ranking(base, [])
        assert [r.chunk_id for r in fused] == ["base_top"]
        assert fused[0].fused_score == pytest.approx(0.032)

    def test_expansion_ranked_by_own_seed_score(self):
        """An expansion from a strong seed can outrank a weak base chunk,
        while an expansion from a weak seed stays below it."""
        base = [
            _result("base_top", fused_score=0.032),
            _result("base_mid", fused_score=0.020),
            _result("base_low", fused_score=0.010),
        ]
        expansions = [
            _expansion("exp_strong", 0.5, 0.032),
            _expansion("exp_weak", 0.5, 0.010),
        ]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        ids = [r.chunk_id for r in fused]
        assert ids.index("exp_strong") < ids.index("base_low")
        assert ids.index("exp_weak") > ids.index("base_low")


class TestGraphCandidateCompetition:
    """Graph candidates must compete fairly with normal candidates: able to
    reach the top-k when their seed is strong, but never above the seed."""

    def test_graph_candidate_can_score_above_old_cutoff(self):
        """A depth-1 expansion from the top seed must exceed the old
        ``max_base * 0.5 * 0.5`` cap that kept graph candidates out of the
        top-k."""
        base = [_result("base_top", fused_score=0.078)]
        expansions = [_expansion("exp_d1", 0.5, 0.078)]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        exp = next(r for r in fused if r.chunk_id == "exp_d1")
        old_cap = 0.078 * 0.5 * 0.5
        assert exp.fused_score > old_cap
        assert exp.fused_score == pytest.approx(0.078 * 0.5)

    def test_depth1_expansion_gets_meaningful_score(self):
        base = [_result("base_top", fused_score=0.060)]
        expansions = [_expansion("exp_d1", 0.5, 0.060)]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        exp = next(r for r in fused if r.chunk_id == "exp_d1")
        assert exp.fused_score == pytest.approx(0.030)

    def test_deeper_expansion_scores_lower(self):
        """For an equivalent candidate, depth-2 must score below depth-1."""
        base = [_result("base_top", fused_score=0.060)]
        expansions = [
            _expansion("exp_d1", 0.5, 0.060),
            _expansion("exp_d2", 0.33, 0.060),
        ]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        ids = [r.chunk_id for r in fused]
        assert ids.index("exp_d1") < ids.index("exp_d2")
        d1 = next(r for r in fused if r.chunk_id == "exp_d1").fused_score
        d2 = next(r for r in fused if r.chunk_id == "exp_d2").fused_score
        assert d2 < d1

    def test_strong_base_candidate_still_outranks_weak_graph(self):
        """A strongly matching base chunk must still outrank a graph
        candidate pulled in by a weak seed."""
        base = [
            _result("base_strong", fused_score=0.060),
            _result("base_weak", fused_score=0.010),
        ]
        expansions = [_expansion("exp_from_weak", 0.5, 0.010)]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        ids = [r.chunk_id for r in fused]
        assert ids.index("base_strong") < ids.index("exp_from_weak")

    def test_graph_candidates_not_all_identical_scores(self):
        """Expansions from different seeds must get distinct scores."""
        base = [
            _result("base_top", fused_score=0.060),
            _result("base_mid", fused_score=0.040),
        ]
        expansions = [
            _expansion("exp_from_top", 0.5, 0.060),
            _expansion("exp_from_mid", 0.5, 0.040),
        ]
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )
        scores = {r.chunk_id: r.fused_score for r in fused}
        assert scores["exp_from_top"] != scores["exp_from_mid"]

    def test_base_scores_unchanged_non_graph_ranking_untouched(self):
        base = [
            _result("base_top", fused_score=0.060),
            _result("base_mid", fused_score=0.040),
        ]
        original = {r.chunk_id: r.fused_score for r in base}
        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [_expansion("exp", 0.5, 0.060)],
        )
        for r in fused:
            if r.chunk_id in original:
                assert r.fused_score == pytest.approx(original[r.chunk_id])

    def test_rrf_fusion_unchanged(self):
        """RRF fusion must be unaffected by the graph scoring change."""
        vector = [_result("a", fused_score=0.9), _result("b", fused_score=0.8)]
        lexical = [_result("b", fused_score=0.85), _result("a", fused_score=0.7)]
        fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
        scores = {r.chunk_id: r.fused_score for r in fused}
        # RRF contribution for a list of 2: 1/(60+1) + 1/(60+2) per chunk
        assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)