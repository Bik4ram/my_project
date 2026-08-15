import math

from semrag.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_all_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_half_relevant():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5


def test_precision_at_k_none_relevant():
    assert precision_at_k(["a", "b"], {"x", "y"}, k=2) == 0.0


def test_precision_at_k_truncates_to_k():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "b"}, k=2) == 1.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_recall_at_k_finds_all():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_finds_half():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_recall_at_k_no_relevant_items_defined():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_recall_at_k_relevant_item_outside_k():
    assert recall_at_k(["a", "b"], {"b"}, k=1) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1.0 / 3.0


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_ndcg_worst_ranking_is_lower_than_perfect():
    perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
    worst = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
    assert worst < perfect


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_ndcg_matches_hand_computed_value():
    # retrieved = [rel, irrel, rel], relevant = {rel@1, rel@3}
    # DCG = 1/log2(2) + 0 + 1/log2(4)
    # IDCG (ideal: both relevant ranked first) = 1/log2(2) + 1/log2(3)
    retrieved = ["r1", "x", "r2"]
    relevant = {"r1", "r2"}
    expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = expected_dcg / expected_idcg
    assert abs(ndcg_at_k(retrieved, relevant, k=3) - expected) < 1e-9
