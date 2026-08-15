from semrag.retrieval.fusion import reciprocal_rank_fusion
from semrag.retrieval.models import RetrievalResult


def _result(chunk_id: str, score: float = 1.0, signal: str = "vector") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, project_id="proj1", relative_path="RTL/a.sv",
        chunk_type="sv_module", text="text", fused_score=score,
        signal_scores={signal: score},
    )


def test_item_ranked_first_in_both_lists_scores_highest():
    vector = [_result("a"), _result("b"), _result("c")]
    lexical = [_result("a"), _result("c"), _result("b")]

    fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert fused[0].chunk_id == "a"


def test_item_in_only_one_list_still_contributes():
    vector = [_result("a"), _result("b")]
    lexical = [_result("c")]

    fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert {r.chunk_id for r in fused} == {"a", "b", "c"}


def test_item_appearing_in_both_lists_outranks_single_list_item():
    vector = [_result("a"), _result("b")]
    lexical = [_result("b"), _result("a")]
    # "a" and "b" both appear in both lists at different ranks; "c" only
    # appears once, at rank 1 — even so, appearing in both lists should
    # generally outrank appearing in just one, since RRF sums contributions.
    lexical_with_solo = [_result("c")] + lexical

    fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical_with_solo})
    top_two_ids = {fused[0].chunk_id, fused[1].chunk_id}
    assert top_two_ids == {"a", "b"}


def test_signal_scores_are_merged_across_lists():
    vector = [_result("a", score=0.9, signal="vector")]
    lexical = [_result("a", score=0.5, signal="lexical")]

    fused = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    result_a = next(r for r in fused if r.chunk_id == "a")
    assert result_a.signal_scores == {"vector": 0.9, "lexical": 0.5}


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion({"vector": [], "lexical": []}) == []


def test_fused_score_is_deterministic_and_positive():
    fused = reciprocal_rank_fusion({"vector": [_result("a")]}, k=60)
    assert fused[0].fused_score == 1.0 / (60 + 1)
