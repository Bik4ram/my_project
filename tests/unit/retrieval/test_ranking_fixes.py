"""
Tests for the ranking-pipeline fixes:

1. LexicalOverlapReranker must NOT replace fused_score with raw Jaccard.
   Real UVM/architecture chunks with high RRF + intent-boost must beat
   repetitive log chunks whose Jaccard is higher.
2. Hybrid top-k reservation must guarantee that the vector-search #1
   and lexical-search #1 survive into the returned top-k.
3. The ranking trace must capture per-stage scores for every chunk.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.pipeline import (
    ChunkTrace,
    HYBRID_RESERVATION_SLOTS,
    HybridRetriever,
    RankingTrace,
)
from semrag.retrieval.reranking import LexicalOverlapReranker


def _result(
    chunk_id: str,
    *,
    relative_path: str = "RTL/a.sv",
    chunk_type: str = "sv_module",
    fused_score: float = 0.0,
    text: str = "module x; endmodule",
    signal_scores: dict[str, float] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text=text,
        fused_score=fused_score,
        signal_scores=signal_scores or {},
    )


class TestLexicalOverlapRerankerPreservesUpstreamScore:
    """The pre-fix bug: reranker overwrote fused_score with raw Jaccard
    (typically 0.05–0.2). After the fix, fused_score is multiplied by
    (1 + weight * jaccard), so a chunk with fused_score=0.5 and Jaccard=0.1
    scores 0.515, not 0.1."""

    def test_high_rrf_chunk_beats_higher_jaccard_log(self):
        """The bug case: log chunk with high Jaccard but low RRF must NOT
        beat a real UVM chunk with high RRF + low Jaccard. This is the
        exact regression that motivated the fix."""
        # RRF score 0.025 from appearing in both lists; intent-boost
        # makes this 0.025 * 1.3 = 0.0325 for sv_module under ARCH intent.
        uvm = _result(
            "uvm_chunk",
            chunk_type="sv_module",
            fused_score=0.0325,
            text="module i2c_master (input logic clk); endmodule",
        )
        # Repetitive log line — high Jaccard with "uvm i2c_master" query
        # because the words repeat, but it isn't a real RTL chunk.
        log = _result(
            "log_chunk",
            chunk_type="log",
            relative_path="sim/log/x.log",
            fused_score=0.0208,
            text="uvm i2c_master STATE_COV_ITEM uvm i2c_master STATE_COV_ITEM uvm i2c_master",
        )
        results = [log, uvm]
        reranked = LexicalOverlapReranker().rerank("uvm i2c_master", results, top_k=2)
        assert reranked[0].chunk_id == "uvm_chunk", (
            "real UVM chunk must beat log noise after reranking; reranker "
            "destroying fused_score was the original bug"
        )

    def test_score_is_multiplicative_boost_not_replacement(self):
        """After reranking, fused_score should be approximately
        original * (1 + weight * jaccard)."""
        original = 0.04
        result = _result(
            "a",
            fused_score=original,
            text="exact same words as the query",
        )
        reranked = LexicalOverlapReranker(weight=0.5).rerank(
            "exact same words as the query", [result], top_k=1
        )
        r = reranked[0]
        # Jaccard = 1.0 (identical token sets), weight = 0.5
        # expected = 0.04 * (1 + 0.5 * 1.0) = 0.06
        assert r.fused_score == pytest.approx(0.06, rel=1e-3)
        assert r.signal_scores["rerank_lexical_overlap"] == pytest.approx(1.0)

    def test_zero_jaccard_leaves_fused_score_untouched(self):
        """A chunk with zero Jaccard overlap keeps its RRF/intent score
        exactly. No reranker-side penalty for being lexically distant
        from the query (semantic similarity is the RRF signal)."""
        result = _result("a", fused_score=0.04, text="completely unrelated content here")
        reranked = LexicalOverlapReranker().rerank("xyzzy plugh", [result], top_k=1)
        assert reranked[0].fused_score == pytest.approx(0.04)

    def test_zero_fused_score_falls_back_to_jaccard(self):
        """Degenerate case (test fixtures, broken fusion): the reranker
        must still produce a useful ranking by falling back to raw
        Jaccard. Preserves the original test-fixture behavior."""
        high = _result("high", fused_score=0.0, text="start condition assertion clock scl master i2c")
        low = _result("low", fused_score=0.0, text="this text shares almost nothing with the query")
        reranked = LexicalOverlapReranker().rerank(
            "i2c master start condition scl clock assertion", [low, high], top_k=2
        )
        assert reranked[0].chunk_id == "high"


class TestHybridReservation:
    """Hybrid top-k reservation guarantees the #1 from vector AND #1 from
    lexical survive into the returned top-k, even if downstream stages
    demoted them via tie-breaks."""

    def _retriever(self) -> HybridRetriever:
        # Construct a real retriever; we won't call its full pipeline,
        # just the reservation helper.
        return HybridRetriever(session=MagicMock())

    def test_top_vector_survives_when_dropped_from_top_k(self):
        """A chunk that was #1 in vector but absent from lexical and
        below the top-k cut after rerank must be re-inserted at the
        front of the returned list."""
        # vector #1: 'vonly', rank 1, score 0.9 — gets pushed out by
        # 10 higher-scoring chunks from fusion
        vonly = _result("vonly", fused_score=0.015)
        # fused top 10 — all with score 0.020+ to push vonly below top_k=10
        top = [_result(f"top{i}", fused_score=0.030 - i * 0.001) for i in range(10)]
        # fused_pool is the post-RRF / pre-rerank pool; vonly is in it
        # (because vector search surfaced it) but not in top-k.
        fused_pool = top + [vonly]
        vector_results = [vonly] + [_result(f"vt{i}", fused_score=0.5 - i * 0.01) for i in range(20)]
        lexical_results = [_result(f"lt{i}", fused_score=0.030 - i * 0.001) for i in range(10)]

        retriever = self._retriever()
        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_pool=fused_pool,
            slots=2,
        )
        assert vonly.chunk_id in {r.chunk_id for r in out[:HYBRID_RESERVATION_SLOTS]}

    def test_no_op_when_both_already_in_top_k(self):
        """If both signal #1s are already in the top-k, reservation
        doesn't duplicate them and the order is unchanged."""
        top = [_result(f"t{i}", fused_score=0.030 - i * 0.001) for i in range(10)]
        # vector and lexical #1 both equal the same chunk already in top
        vector_results = list(top)
        lexical_results = list(top)
        before_ids = [r.chunk_id for r in top]
        retriever = self._retriever()
        out = retriever._apply_hybrid_reservation(
            top, vector_results=vector_results, lexical_results=lexical_results, slots=2
        )
        assert [r.chunk_id for r in out] == before_ids

    def test_lexical_only_champion_reserved(self):
        """A chunk that topped lexical search but not vector search
        still gets a reservation slot when missing from top-k."""
        top = [_result(f"t{i}", fused_score=0.030 - i * 0.001) for i in range(10)]
        # vector #1 IS in top, but lexical #1 ('lexonly') is not.
        vector_results = list(top)
        lexonly = _result("lexonly", fused_score=0.012)
        lexical_results = [lexonly] + [_result(f"lx{i}", fused_score=0.025 - i * 0.001) for i in range(10)]
        fused_pool = top + [lexonly]
        retriever = self._retriever()
        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_pool=fused_pool,
            slots=2,
        )
        assert out[0].chunk_id == "lexonly"

    def test_empty_input_returns_empty(self):
        retriever = self._retriever()
        assert retriever._apply_hybrid_reservation([], vector_results=[], lexical_results=[]) == []

    def test_top_k_trimmed_to_original_length(self):
        """If reservation pushes past top_k, the result is trimmed back."""
        top = [_result(f"t{i}", fused_score=0.030 - i * 0.001) for i in range(10)]
        vector_results = [_result("vonly", fused_score=0.9)] + [
            _result(f"vt{i}", fused_score=0.5 - i * 0.01) for i in range(20)
        ]
        lexical_results = [_result(f"lt{i}", fused_score=0.030 - i * 0.001) for i in range(10)]
        fused_pool = top + [_result("vonly", fused_score=0.9)]
        retriever = self._retriever()
        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_pool=fused_pool,
            slots=2,
        )
        assert len(out) == len(top)


class TestRankingTrace:
    """The trace records per-chunk scores at every stage."""

    def test_trace_collects_per_chunk(self):
        trace = RankingTrace("test query")
        a = trace.get_or_create("a")
        a.vector_score = 0.9
        a.vector_rank = 1
        a.rrf_score = 0.025
        b = trace.get_or_create("b")
        b.lexical_score = 0.7
        b.lexical_rank = 1
        assert {c.chunk_id for c in trace.chunks()} == {"a", "b"}

    def test_chunk_trace_to_dict_is_json_safe(self):
        ct = ChunkTrace(chunk_id="x", vector_score=0.9, vector_rank=1, reached_llm=True)
        d = ct.to_dict()
        assert d["chunk_id"] == "x"
        assert d["vector_score"] == 0.9
        assert d["reached_llm"] is True


class TestRetrieverEmitsTrace:
    """HybridRetriever wires a RankingTrace through every stage and emits
    it via the optional trace sink."""

    def test_retriever_uses_trace_sink(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod

        class _StubVectorStore:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return []

        class _StubGraph:
            def __init__(self, session):
                pass

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVectorStore)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(pipeline_mod, "PostgresGraphQueries", _StubGraph)

        from semrag.storage.vector.models import VectorSearchFilter

        captured = MagicMock()
        retriever = HybridRetriever(session=MagicMock(), trace_sink=captured)
        # Should not raise even with empty results
        out = retriever.retrieve(
            "q",
            [0.0] * 4,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
        )
        assert out == []
        assert captured.call_count == 1
        trace_arg = captured.call_args[0][0]
        assert isinstance(trace_arg, RankingTrace)
        assert trace_arg.query == "q"


class TestGraphExpansionDoesNotCorruptBaseScores:
    """Regression: graph expansion must NOT use a second RRF fusion that
    recalculates base chunk scores from scratch. After the fix, base chunks
    keep their first-RRF scores and graph expansion chunks are scored
    seed-relative: their producing seed's RRF score (``graph_seed_score``)
    decayed by graph depth (``graph_expansion`` = ``1/(depth+1)``)."""

    def _result(
        self,
        chunk_id: str,
        fused_score: float = 1.0,
        signal_scores: dict | None = None,
        chunk_type: str = "sv_module",
        relative_path: str = "RTL/m.sv",
        source_node_id: str | None = "n1",
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            project_id="p1",
            relative_path=relative_path,
            chunk_type=chunk_type,
            text="module x; endmodule",
            source_node_id=source_node_id,
            fused_score=fused_score,
            signal_scores=signal_scores or {},
        )

    def test_graph_expansion_scores_are_seed_relative(self):
        """Expansions must inherit their producing seed's RRF score decayed
        by depth, and base chunks must keep their first-RRF scores."""
        from semrag.retrieval.ranking.graph import apply_graph_expansion_ranking

        # Base chunks after first RRF: scores ~0.032 to 0.010
        base_chunks = [
            self._result("base_top", fused_score=0.032, source_node_id="n1"),
            self._result("base_mid", fused_score=0.020, source_node_id="n2"),
            self._result("base_low", fused_score=0.010, source_node_id="n3"),
        ]
        # Graph expansion would return new chunks with ``graph_expansion``
        # and ``graph_seed_score`` signal scores set; simulate what
        # ``expand_with_graph_context`` produces with various depths.
        expansion_chunks = [
            RetrievalResult(
                chunk_id="exp_depth1",
                project_id="p1", relative_path="RTL/e1.sv",
                chunk_type="sv_module", text="dependent module",
                source_node_id="e1",
                fused_score=0.5,  # initial score (overridden by pipeline)
                signal_scores={
                    "graph_expansion": 0.5,  # depth=1 -> 1/(1+1)
                    "graph_seed_score": 0.032,  # produced by base_top
                },
            ),
            RetrievalResult(
                chunk_id="exp_depth2",
                project_id="p1", relative_path="RTL/e2.sv",
                chunk_type="sv_module", text="transitive dependent",
                source_node_id="e2",
                fused_score=0.33,  # initial score (overridden by pipeline)
                signal_scores={
                    "graph_expansion": 0.33,  # depth=2 -> 1/(2+1)
                    "graph_seed_score": 0.020,  # produced by base_mid
                },
            ),
        ]

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base_chunks],
            [e.model_copy(deep=True) for e in expansion_chunks],
        )

        # Base chunk RRF scores must be preserved
        for r in fused:
            if r.chunk_id == "base_top":
                assert r.fused_score == pytest.approx(0.032, abs=1e-6)
            elif r.chunk_id == "base_mid":
                assert r.fused_score == pytest.approx(0.020, abs=1e-6)
            elif r.chunk_id == "base_low":
                assert r.fused_score == pytest.approx(0.010, abs=1e-6)

        # Expansion scores are seed_relative * depth_weight
        scores = {r.chunk_id: r.fused_score for r in fused}
        assert scores["exp_depth1"] == pytest.approx(0.032 * 0.5)
        assert scores["exp_depth2"] == pytest.approx(0.020 * 0.33)

        # Depth attenuation: shallower expansion ranks above deeper one.
        ids = [r.chunk_id for r in fused]
        assert ids.index("exp_depth1") < ids.index("exp_depth2")

    def test_graph_expansion_scores_dominate_if_second_rrf_were_used(self):
        """Demonstrate the bug: if a second RRF fusion were used instead
        of the seed-relative merge, expansion chunks would interleave with
        or beat base chunks, proving why the fix is necessary.

        This test simulates what the old second-RRF approach would produce
        by recalculating scores from rank positions alone."""
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        # Same base chunks, same expansions as the test above
        base = [
            self._result("base_top", fused_score=0.032, source_node_id="n1"),
            self._result("base_mid", fused_score=0.020, source_node_id="n2"),
            self._result("base_low", fused_score=0.010, source_node_id="n3"),
        ]
        expansions = [
            RetrievalResult(
                chunk_id="exp_d1", project_id="p1", relative_path="RTL/e1.sv",
                chunk_type="sv_module", text="dep",
                source_node_id="e1", fused_score=0.5,
                signal_scores={"graph_expansion": 0.5},
            ),
        ]

        # What the OLD code did: second RRF fusion recalculates all scores
        old_fused = reciprocal_rank_fusion({"base": base, "graph_expansion": expansions})
        old_order = [r.chunk_id for r in old_fused]

        # With only 4 items (3 base + 1 expansion), a second RRF lets the
        # expansion interleave with or even beat base chunks based purely
        # on rank position, discarding the carefully-combined vector +
        # lexical signal.
        assert old_order.index("exp_d1") < len(old_order) - 1, (
            "Second RRF would have let an expansion beat at least the "
            "lowest base chunk — confirming why we removed it."
        )


class TestRRFDeterminism:
    """RRF must be fully deterministic — same inputs, same outputs."""

    def _result(self, chunk_id: str, score: float = 1.0, signal: str = "vector") -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, project_id="p1", relative_path="RTL/a.sv",
            chunk_type="sv_module", text="text", fused_score=score,
            signal_scores={signal: score},
        )

    def test_identical_inputs_produce_identical_output(self):
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        vector = [self._result("a", 0.9), self._result("b", 0.8), self._result("c", 0.7)]
        lexical = [self._result("b", 0.85), self._result("c", 0.75), self._result("a", 0.65)]

        r1 = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
        r2 = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})

        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        for ra, rb in zip(r1, r2):
            assert ra.fused_score == pytest.approx(rb.fused_score)

    def test_single_list_produces_monotonic_scores(self):
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        results = [self._result(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
        fused = reciprocal_rank_fusion({"vector": results})

        for i in range(len(fused) - 1):
            assert fused[i].fused_score >= fused[i + 1].fused_score, (
                f"RRF scores must be non-increasing: {fused[i].fused_score} < {fused[i + 1].fused_score}"
            )

    def test_all_scores_positive(self):
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        results = [self._result(f"c{i}") for i in range(100)]
        fused = reciprocal_rank_fusion({"vector": results})
        for r in fused:
            assert r.fused_score > 0.0, f"RRF score must be positive: {r.chunk_id} score={r.fused_score}"


class TestStableTopK:
    """Stable top-k must preserve input order for equal scores and be
    fully deterministic."""

    def _result(self, chunk_id: str, score: float = 0.0) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, project_id="p1", relative_path="RTL/a.sv",
            chunk_type="sv_module", text="text", fused_score=score,
        )

    def test_tie_break_by_chunk_id_deterministic(self):
        from semrag.retrieval.pipeline import _stable_top_k

        results = [
            self._result("z", 0.01),
            self._result("a", 0.01),
            self._result("m", 0.01),
        ]
        # All have the same fused_score, so chunk_id breaks ties
        sorted_once = _stable_top_k(results)
        sorted_twice = _stable_top_k(results)
        ids1 = [r.chunk_id for r in sorted_once]
        ids2 = [r.chunk_id for r in sorted_twice]
        assert ids1 == ids2

    def test_top_k_respected(self):
        from semrag.retrieval.pipeline import _stable_top_k

        results = [self._result(f"c{i}", 1.0 - i * 0.1) for i in range(20)]
        top5 = _stable_top_k(results, top_k=5)
        assert len(top5) == 5

    def test_empty_input(self):
        from semrag.retrieval.pipeline import _stable_top_k

        assert _stable_top_k([]) == []


class TestScoreMonotonicity:
    """Key invariant: downstream stages should never reverse clearly-
    stronger upstream evidence unless intentional (intent boost,
    reranker)."""

    def _result(self, chunk_id: str, fused_score: float,
                chunk_type: str = "sv_module", text: str = "some content") -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, project_id="p1", relative_path="RTL/x.sv",
            chunk_type=chunk_type, text=text, fused_score=fused_score,
        )

    def test_intent_boost_does_not_reverse_large_rrf_gap(self):
        """A chunk with 2x RRF score over another chunk must still
        outrank it after intent boost, even if the intent boost favors
        the weaker chunk."""
        from semrag.retrieval.intent import QueryIntent
        from semrag.retrieval.ranking.intent import apply_intent_ranking

        strong = self._result("strong", fused_score=0.030, chunk_type="log", text="log content")
        weak = self._result("weak", fused_score=0.015, chunk_type="sv_module", text="module x; endmodule")
        results = [weak, strong]

        # ARCHITECTURE intent gives sv_module a +0.3 boost
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        assert boosted[0].chunk_id == "strong", (
            "Even with intent boost favoring the weak chunk, the strong "
            "chunk with 2x RRF score must still outrank it."
        )

    def test_intent_boost_can_break_near_ties(self):
        """Intent boost SHOULD be able to reorder near-ties where scores
        are close — that's its purpose."""
        from semrag.retrieval.intent import QueryIntent
        from semrag.retrieval.ranking.intent import apply_intent_ranking

        a = self._result("sv_mod", fused_score=0.020, chunk_type="sv_module", text="module x; endmodule")
        b = self._result("log", fused_score=0.019, chunk_type="log", text="log content")
        results = [a, b]

        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        # sv_module gets +0.3 boost (1.3x), so 0.020 * 1.3 = 0.026
        # log has no boost, stays at 0.019
        assert boosted[0].chunk_id == "sv_mod"

    def test_reranker_does_not_reverse_large_rrf_gap(self):
        """Reranker must not reverse a large RRF gap with Jaccard."""
        from semrag.retrieval.reranking import LexicalOverlapReranker

        # High RRF, low Jaccard (relevant but lexically distant)
        strong = self._result(
            "strong", fused_score=0.030,
            text="scoreboard write_xtn i2c master fsm start stop condition",
        )
        # Low RRF, high Jaccard with query (log that happens to share words)
        weak = self._result(
            "weak", fused_score=0.008,
            text="start stop condition start stop condition start stop condition",
        )
        results = [weak, strong]

        reranked = LexicalOverlapReranker().rerank(
            "start condition i2c master", results, top_k=2
        )
        assert reranked[0].chunk_id == "strong", (
            "Reranker must not let a log chunk with high Jaccard but low "
            "RRF beat a real chunk with high RRF."
        )


class TestDedupStability:
    """Deduplication must not silently drop unique chunks."""

    def _result(self, chunk_id: str, text: str = "unique text",
                relative_path: str = "RTL/a.sv", score: float = 1.0) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, project_id="p1", relative_path=relative_path,
            chunk_type="sv_module", text=text, fused_score=score,
        )

    def test_rrf_merges_duplicate_chunks_from_both_lists(self):
        """When the same chunk appears in both vector and lexical lists,
        RRF must assign it a single entry with accumulated score."""
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        vector = [self._result("a"), self._result("b")]
        lexical = [self._result("b"), self._result("c")]

        fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
        ids = [r.chunk_id for r in fused]
        assert ids.count("b") == 1  # no duplicate
        assert set(ids) == {"a", "b", "c"}

    def test_signal_scores_merged_for_duplicate(self):
        """When a chunk appears in both lists, its signal_scores must
        contain keys from both."""
        from semrag.retrieval.fusion import reciprocal_rank_fusion

        a_vec = self._result("a", score=0.9)
        a_vec.signal_scores = {"vector": 0.9}
        a_lex = self._result("a", score=0.7)
        a_lex.signal_scores = {"lexical": 0.7}

        fused = reciprocal_rank_fusion({"vector": [a_vec], "lexical": [a_lex]})
        result = fused[0]
        assert result.signal_scores.get("vector") == 0.9
        assert result.signal_scores.get("lexical") == 0.7


class TestHybridReservationEdgeCases:
    """Edge cases for hybrid top-k reservation."""

    def _retriever(self) -> HybridRetriever:
        from unittest.mock import MagicMock
        return HybridRetriever(session=MagicMock())

    def _result(self, chunk_id: str, fused_score: float = 0.0) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, project_id="p1", relative_path="RTL/a.sv",
            chunk_type="sv_module", text="text", fused_score=fused_score,
        )

    def test_reservation_same_chunk_wins_both_lists(self):
        """When the same chunk is #1 in both vector and lexical, only
        one slot is reserved (no duplication)."""
        retriever = self._retriever()
        chunk = self._result("champion", 0.9)
        top = [self._result(f"t{i}", 0.05 - i * 0.001) for i in range(10)]
        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=[chunk] + [self._result(f"v{i}") for i in range(5)],
            lexical_results=[chunk] + [self._result(f"l{i}") for i in range(5)],
            fused_pool=top + [chunk],
            slots=2,
        )
        # 'champion' should be in the reserved slots exactly once
        champ_positions = [i for i, r in enumerate(out) if r.chunk_id == "champion"]
        assert len(champ_positions) == 1

    def test_reservation_with_empty_lexical_results(self):
        """Reservation handles the case where lexical search returns
        nothing (e.g. no keyword matches)."""
        retriever = self._retriever()
        top = [self._result(f"t{i}", 0.03 - i * 0.001) for i in range(5)]
        vector_only = self._result("vonly", 0.9)
        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=[vector_only],
            lexical_results=[],
            fused_pool=top + [vector_only],
            slots=2,
        )
        assert vector_only.chunk_id in {r.chunk_id for r in out[:HYBRID_RESERVATION_SLOTS]}
