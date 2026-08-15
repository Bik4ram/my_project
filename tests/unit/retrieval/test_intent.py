"""Query-intent classification and rerank boost."""
from __future__ import annotations

import pytest

from semrag.retrieval.intent import QueryIntent, classify_intent
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.intent import apply_intent_ranking, boost_for_intent


def _result(chunk_id: str, chunk_type: str, fused_score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="p",
        chunk_type=chunk_type,
        relative_path="rtl/x.sv",
        text="module x; endmodule",
        fused_score=fused_score,
        signal_scores={"rrf": fused_score},
    )


def test_classify_intent_general_on_empty_query():
    assert classify_intent("") == QueryIntent.GENERAL
    assert classify_intent("what is the meaning of life") == QueryIntent.GENERAL


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what module drives the SDA line?", QueryIntent.ARCHITECTURE),
        ("explain the reset sequence for the FIFO", QueryIntent.ARCHITECTURE),
        ("describe the state machine in the controller", QueryIntent.ARCHITECTURE),
        ("how is the i2c_master module instantiated?", QueryIntent.ARCHITECTURE),
        ("walk me through the UVM sequence", QueryIntent.VERIFICATION),
        ("what does the scoreboard check?", QueryIntent.VERIFICATION),
        ("explain the covergroup coverage points", QueryIntent.VERIFICATION),
        ("what assertions are in the testbench?", QueryIntent.VERIFICATION),
        ("why is the simulation failing?", QueryIntent.DEBUG),
        ("how do I read this uvm_error log?", QueryIntent.DEBUG),
        ("coverage gap dropped on bin xyz", QueryIntent.DEBUG),
    ],
)
def test_classify_intent_recognizes_dv_topics(query: str, expected: QueryIntent):
    assert classify_intent(query) == expected


def test_classify_intent_architecture_wins_ties():
    """A question that overlaps ARCHITECTURE + VERIFICATION lexicons
    lands on architecture — an RAG for design-and-verification repos
    is more often anchored in the design RTL than in the testbench
    alone, and a wrong arch answer is more recoverable than a wrong
    arch-vs-debug misfire."""
    query = "how is the module connected to the sequencer?"
    assert classify_intent(query) == QueryIntent.ARCHITECTURE


def test_apply_intent_ranking_boosts_matching_chunks():
    results = [
        _result("a", "sv_module", fused_score=0.5),
        _result("b", "log", fused_score=0.5),
        _result("c", "sv_assertion", fused_score=0.5),
    ]
    boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
    by_id = {r.chunk_id: r for r in boosted}

    # ``sv_module`` outranks ``sv_assertion`` (smaller boost) and
    # ``log`` (no boost, untouched score).
    assert boosted[0].chunk_id == "a"
    assert boosted[1].chunk_id == "c"
    assert boosted[2].chunk_id == "b"

    # Boosts are multiplicative, capped at max_boost.
    sv_module = by_id["a"]
    assert sv_module.fused_score > 0.5
    assert sv_module.signal_scores["intent_boost"] > 0
    assert pytest.approx(sv_module.fused_score, rel=1e-3) == 0.5 * (1 + sv_module.signal_scores["intent_boost"])


def test_apply_intent_ranking_general_is_noop():
    results = [
        _result("a", "sv_module", fused_score=0.9),
        _result("b", "log", fused_score=0.5),
    ]
    boosted = apply_intent_ranking(results, QueryIntent.GENERAL)
    assert [r.fused_score for r in boosted] == [0.9, 0.5]


def test_apply_intent_ranking_respects_max_boost_cap():
    """Even a maximally-boosted chunk cannot double its score. The
    intent layer is a nudge, not a re-ranker's replacement."""
    results = [_result("a", "sv_module", fused_score=0.5)]
    boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE, max_boost=0.2)
    # 0.5 * (1 + 0.2) = 0.6 — capped at 0.2 * 0.5 = 0.1 above original.
    assert boosted[0].fused_score <= 0.5 * 1.2 + 1e-9


def test_apply_intent_ranking_uses_longest_prefix_match():
    """``sv_module_signature`` should match the ``sv_module_signature``
    entry, not the smaller ``sv_module`` one — long-prefix wins
    over short-prefix."""
    results = [
        _result("a", "sv_module_signature", fused_score=0.5),
        _result("b", "sv_module", fused_score=0.5),
    ]
    boosted = apply_intent_ranking(results, QueryIntent.ARCHITECTURE)
    sig_boost = next(r.signal_scores.get("intent_boost", 0) for r in boosted if r.chunk_id == "a")
    plain_boost = next(r.signal_scores.get("intent_boost", 0) for r in boosted if r.chunk_id == "b")
    assert sig_boost != plain_boost


def test_boost_for_intent_returns_table_for_each_intent():
    for intent in QueryIntent:
        boost_for_intent(intent)  # shouldn't raise; values can be empty (GENERAL)
