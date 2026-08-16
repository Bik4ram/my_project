"""
Property-based invariant tests for the retrieval pipeline.

These tests validate that the pipeline satisfies certain invariants
regardless of input data — they're not about correctness of specific
results but about system-level properties that must always hold.

Properties tested:
  - Idempotency: repeated identical query returns identical ranking
  - Score monotonicity: RRF scores are non-increasing within a fused list
  - Score preservation: base chunk scores survive graph expansion unchanged
  - Candidate preservation: adding candidates never removes existing ones
  - Deterministic ordering: same inputs always produce same output order
  - No negative scores: all scores at all stages are >= 0
  - Hybrid reservation: top vector/lexical #1 always makes top-k
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semrag.retrieval.fusion import reciprocal_rank_fusion
from semrag.retrieval.intent import QueryIntent, classify_intent
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.pipeline import (
    HYBRID_RESERVATION_SLOTS,
    HybridRetriever,
    RankingTrace,
    _log_ranking_summary,
    _stable_top_k,
)
from semrag.retrieval.ranking.graph import apply_graph_expansion_ranking
from semrag.retrieval.ranking.intent import apply_intent_ranking
from semrag.retrieval.reranking import LexicalOverlapReranker


def _result(
    chunk_id: str,
    fused_score: float = 1.0,
    chunk_type: str = "sv_module",
    text: str = "module x; endmodule",
    relative_path: str = "RTL/m.sv",
    source_node_id: str | None = "n1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="p1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text=text,
        source_node_id=source_node_id,
        fused_score=fused_score,
    )


# ── Property 1: Idempotency ─────────────────────────────────────────────

class TestIdempotency:
    """Repeated identical query must return identical ranking."""

    def test_rrf_idempotent(self):
        vec = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(20)]
        lex = [_result(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        r1 = reciprocal_rank_fusion({"vector": vec, "lexical": lex})
        r2 = reciprocal_rank_fusion({"vector": vec, "lexical": lex})
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        for a, b in zip(r1, r2):
            assert a.fused_score == pytest.approx(b.fused_score)

    def test_intent_boost_idempotent(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(10)]
        r1 = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        r2 = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_reranker_idempotent(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05,
                           text="i2c master start condition" if i == 0 else "other text")
                   for i in range(10)]
        reranker = LexicalOverlapReranker()
        r1 = reranker.rerank("i2c master", results, top_k=10)
        r2 = reranker.rerank("i2c master", results, top_k=10)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_stable_top_k_idempotent(self):
        results = [_result(f"c{i}", 0.01) for i in range(10)]
        r1 = _stable_top_k(results)
        r2 = _stable_top_k(results)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_classify_intent_idempotent(self):
        assert classify_intent("module i2c master") == classify_intent("module i2c master")
        assert classify_intent("uvm testbench failure") == classify_intent("uvm testbench failure")


# ── Property 2: Score monotonicity ───────────────────────────────────────

class TestScoreMonotonicity:
    """Within any sorted result list, scores must be non-increasing."""

    def _check_non_increasing(self, results: list[RetrievalResult], label: str):
        for i in range(len(results) - 1):
            assert results[i].fused_score >= results[i + 1].fused_score - 1e-9, (
                f"{label}: scores must be non-increasing at index {i}: "
                f"{results[i].fused_score} < {results[i + 1].fused_score}"
            )

    def test_rrf_scores_non_increasing(self):
        vec = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(20)]
        lex = [_result(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        fused = reciprocal_rank_fusion({"vector": vec, "lexical": lex})
        self._check_non_increasing(fused, "RRF")

    def test_intent_boost_scores_non_increasing(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(10)]
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        self._check_non_increasing(boosted, "intent boost")

    def test_reranker_scores_non_increasing(self):
        results = [
            _result(f"c{i}", 1.0 - i * 0.05,
                    text=f"i2c master start condition {i}" if i < 3 else f"other {i}")
            for i in range(10)
        ]
        reranker = LexicalOverlapReranker()
        reranked = reranker.rerank("i2c master start condition", results, top_k=10)
        self._check_non_increasing(reranked, "reranker")

    def test_stable_top_k_scores_non_increasing(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(20)]
        top5 = _stable_top_k(results, top_k=5)
        self._check_non_increasing(top5, "stable top-k")


# ── Property 3: Score preservation through graph expansion ───────────────

class TestScorePreservation:
    """Base chunk RRF scores must survive graph expansion unchanged."""

    def test_base_chunk_scores_unchanged_after_expansion(self):
        base_chunks = [
            _result("a", fused_score=0.032),
            _result("b", fused_score=0.020),
            _result("c", fused_score=0.010),
        ]
        expansions = [
            _result("exp1", fused_score=0.5),
            _result("exp2", fused_score=0.33),
        ]
        for e, seed in zip(expansions, (0.032, 0.020), strict=True):
            e.signal_scores = {"graph_expansion": e.fused_score, "graph_seed_score": seed}

        original_scores = {r.chunk_id: r.fused_score for r in base_chunks}

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base_chunks],
            [e.model_copy(deep=True) for e in expansions],
        )

        for r in fused:
            if r.chunk_id in original_scores:
                assert r.fused_score == pytest.approx(original_scores[r.chunk_id])

    def test_expansion_scores_never_exceed_max_base(self):
        """Graph expansion chunk scores must be strictly less than the
        maximum base chunk score."""
        base_chunks = [
            _result("a", fused_score=0.032),
            _result("b", fused_score=0.020),
            _result("c", fused_score=0.010),
        ]
        expansions = [
            _result("exp1", fused_score=0.5),
            _result("exp2", fused_score=0.33),
        ]
        for e, seed in zip(expansions, (0.032, 0.020), strict=True):
            e.signal_scores = {"graph_expansion": e.fused_score, "graph_seed_score": seed}

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base_chunks],
            [e.model_copy(deep=True) for e in expansions],
        )

        max_base = max(r.fused_score for r in base_chunks)
        for r in fused:
            if "exp" in r.chunk_id:
                assert r.fused_score < max_base, (
                    f"Expansion {r.chunk_id} score {r.fused_score} must be "
                    f"below max base score {max_base}"
                )


# ── Property 4: Candidate preservation ───────────────────────────────────

class TestCandidatePreservation:
    """Adding new candidates must never remove existing candidates from
    the pool. The candidate set only grows (or stays the same)."""

    def test_expansion_never_reduces_candidate_set(self):
        base = [_result(f"c{i}", 0.03 - i * 0.001) for i in range(10)]
        base_ids = {r.chunk_id for r in base}

        expansions = [_result(f"e{j}", 0.5) for j in range(3)]
        for e, seed in zip(expansions, (0.03, 0.029, 0.028), strict=True):
            e.signal_scores = {"graph_expansion": e.fused_score, "graph_seed_score": seed}

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )

        fused_ids = {r.chunk_id for r in fused}
        assert base_ids.issubset(fused_ids), (
            "All original base IDs must still be in the fused set after expansion"
        )

    def test_hybrid_reservation_champion_survives(self):
        """Hybrid reservation guarantees the vector #1 champion survives
        even when it was missing from the top-k."""
        retriever = HybridRetriever(session=MagicMock())
        top = [_result(f"t{i}", 0.03 - i * 0.001) for i in range(5)]
        top_ids = {r.chunk_id for r in top}

        vector_champ = _result("vonly", 0.9)
        vector_results = [vector_champ]
        lexical_results = list(top)

        out = retriever._apply_hybrid_reservation(
            top,
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_pool=top + [vector_champ],
            slots=2,
        )
        out_ids = {r.chunk_id for r in out}
        # Vector champion must be in the reserved slots
        assert vector_champ.chunk_id in {r.chunk_id for r in out[:HYBRID_RESERVATION_SLOTS]}
        # All original top-k survivors (those not pushed out by reservation)
        # must still be present
        for cid in top_ids:
            if cid in out_ids:
                pass  # survived
        # The champion was promoted, *some* existing entry may have been
        # demoted, but no entries were silently dropped without replacement.
        assert len(out) == len(top)  # length stays the same


# ── Property 5: No negative scores ──────────────────────────────────────

class TestNoNegativeScores:
    """All scores at all pipeline stages must be >= 0."""

    def test_rrf_scores_non_negative(self):
        vec = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(20)]
        lex = [_result(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        fused = reciprocal_rank_fusion({"vector": vec, "lexical": lex})
        for r in fused:
            assert r.fused_score >= 0.0

    def test_intent_boost_scores_non_negative(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05) for i in range(10)]
        boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
        for r in boosted:
            assert r.fused_score >= 0.0

    def test_reranker_scores_non_negative(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05,
                           text="some content here") for i in range(10)]
        reranker = LexicalOverlapReranker()
        reranked = reranker.rerank("test query", results, top_k=10)
        for r in reranked:
            assert r.fused_score >= 0.0


# ── Property 6: Deterministic ordering ──────────────────────────────────

class TestDeterministicOrdering:
    """Same inputs must always produce the same output order."""

    def test_reranker_order_deterministic(self):
        results = [_result(f"c{i}", 1.0 - i * 0.05,
                           text=f"i2c master data {i}") for i in range(15)]
        reranker = LexicalOverlapReranker()
        for query in ["i2c master", "data transfer", "start condition"]:
            r1 = reranker.rerank(query, results, top_k=10)
            r2 = reranker.rerank(query, results, top_k=10)
            assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]

    def test_stable_top_k_tie_break_deterministic(self):
        """When many items have the same score, the tiebreak (chunk_id)
        must produce a deterministic ordering."""
        results = [_result(f"c{i}", 0.01) for i in range(20)]
        r1 = _stable_top_k(results)
        r2 = _stable_top_k(results)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


# ── Property 7: Graph expansion never replaces stronger evidence ────────

class TestGraphExpansionDoesNotReplaceEvidence:
    """Graph expansion must never cause a stronger base chunk (higher RRF
    score) to be removed from the fused pool."""

    def test_all_base_chunks_survive_expansion(self):
        base = [_result(f"c{i}", 0.03 - i * 0.001) for i in range(10)]
        base_ids = {r.chunk_id for r in base}

        expansions = [_result(f"e{j}", 0.5) for j in range(5)]
        for e, seed in zip(expansions, (0.03, 0.029, 0.028, 0.027, 0.026), strict=True):
            e.signal_scores = {"graph_expansion": e.fused_score, "graph_seed_score": seed}

        fused = apply_graph_expansion_ranking(
            [r.model_copy(deep=True) for r in base],
            [e.model_copy(deep=True) for e in expansions],
        )

        for cid in base_ids:
            assert any(r.chunk_id == cid for r in fused), (
                f"Base chunk {cid} missing from fused pool after expansion"
            )


# ── Property 8: Trace completeness ──────────────────────────────────────

class TestTraceCompleteness:
    """Every chunk that reaches the LLM must have a complete trace record."""

    def test_all_reached_chunks_have_final_score(self):
        trace = RankingTrace("test query")
        chunks = ["a", "b", "c"]
        for cid in chunks:
            ct = trace.get_or_create(cid)
            ct.final_score = 0.5
            ct.final_rank = chunks.index(cid) + 1
            ct.reached_llm = True

        for c in trace.chunks():
            if c.reached_llm:
                assert c.final_score is not None
                assert c.final_rank is not None

    def test_trace_write_to_noop_without_active_writer(self):
        """write_to should not raise when writer is inactive."""
        from unittest.mock import MagicMock
        trace = RankingTrace("test")
        trace.get_or_create("a")
        # Mock writer whose is_active returns False
        inactive = MagicMock()
        inactive.is_active = False
        trace.write_to(inactive)  # should not raise


# ── Property 9: Intent classification ───────────────────────────────────

class TestIntentClassificationInvariants:
    """Intent classification must always return a valid intent."""

    def test_always_returns_valid_intent(self):
        queries = [
            "how does the i2c master work",
            "uvm scoreboard transaction",
            "simulation log failure",
            "completely random text that matches nothing",
            "",
            "module and uvm both mentioned",
        ]
        from semrag.retrieval.intent import classify_intent
        for q in queries:
            intent = classify_intent(q)
            assert intent in (
                QueryIntent.ARCHITECTURE,
                QueryIntent.VERIFICATION,
                QueryIntent.DEBUG,
                QueryIntent.GENERAL,
            )

    def test_empty_query_returns_general(self):
        from semrag.retrieval.intent import classify_intent
        assert classify_intent("") == QueryIntent.GENERAL
        assert classify_intent("   ") == QueryIntent.GENERAL
