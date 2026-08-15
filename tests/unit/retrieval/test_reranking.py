from unittest.mock import MagicMock, patch

import pytest
import requests

from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.reranking import HTTPRerankerClient, LexicalOverlapReranker, RerankerError


def _result(
    chunk_id: str,
    text: str,
    *,
    fused_score: float = 0.0,
    symbol_name: str | None = None,
    relative_path: str = "RTL/a.sv",
    chunk_type: str = "sv_module",
    category: str = "rtl",
    signal_scores: dict[str, float] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, project_id="proj1", relative_path=relative_path,
        chunk_type=chunk_type, text=text, fused_score=fused_score,
        symbol_name=symbol_name, category=category,
        signal_scores=dict(signal_scores or {}),
    )


class TestLexicalOverlapReranker:
    def test_higher_overlap_ranks_first(self):
        results = [
            _result("low", "this text shares almost nothing with the query"),
            _result("high", "start condition assertion clock scl master i2c"),
        ]
        reranked = LexicalOverlapReranker().rerank(
            "i2c master start condition scl clock assertion", results, top_k=2
        )
        assert reranked[0].chunk_id == "high"

    def test_rerank_score_recorded_in_signal_scores(self):
        results = [_result("a", "start condition scl")]
        reranked = LexicalOverlapReranker().rerank("start condition scl", results, top_k=1)
        assert "rerank_lexical_overlap" in reranked[0].signal_scores
        assert reranked[0].signal_scores["rerank_lexical_overlap"] > 0

    def test_top_k_truncates(self):
        results = [_result(f"c{i}", f"word{i}") for i in range(10)]
        reranked = LexicalOverlapReranker().rerank("word1", results, top_k=3)
        assert len(reranked) == 3

    def test_empty_query_returns_first_top_k_unranked(self):
        results = [_result("a", "text"), _result("b", "text")]
        reranked = LexicalOverlapReranker().rerank("", results, top_k=2)
        assert len(reranked) == 2

    def test_no_token_overlap_scores_zero(self):
        results = [_result("a", "completely unrelated content here")]
        reranked = LexicalOverlapReranker().rerank("xyzzy plugh", results, top_k=1)
        assert reranked[0].signal_scores["rerank_lexical_overlap"] == 0.0


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


class TestHTTPRerankerClient:
    def test_reorders_by_relevance_score(self):
        client = HTTPRerankerClient(base_url="http://localhost:8080", model="reranker-model")
        results = [_result("first", "text a"), _result("second", "text b")]
        # Server says index 1 ("second") is actually more relevant.
        mock_payload = {"results": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]}
        with patch("requests.post", return_value=_mock_response(mock_payload)):
            reranked = client.rerank("query", results, top_k=2)

        assert reranked[0].chunk_id == "second"
        assert reranked[0].signal_scores["rerank_cross_encoder"] == 0.9

    def test_respects_top_k(self):
        client = HTTPRerankerClient(base_url="http://localhost:8080", model="m")
        results = [_result(f"c{i}", f"text {i}") for i in range(5)]
        mock_payload = {"results": [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]}
        with patch("requests.post", return_value=_mock_response(mock_payload)):
            reranked = client.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    def test_empty_results_returns_empty_without_request(self):
        client = HTTPRerankerClient(base_url="http://localhost:8080", model="m")
        with patch("requests.post") as mock_post:
            result = client.rerank("query", [], top_k=5)
        assert result == []
        mock_post.assert_not_called()

    def test_exhausted_retries_raises(self):
        client = HTTPRerankerClient(base_url="http://localhost:8080", model="m", max_retries=2)
        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            with patch("time.sleep"):
                with pytest.raises(RerankerError):
                    client.rerank("query", [_result("a", "text")], top_k=1)


# All three "unrelated" fixtures share text with zero overlap against the
# queries used below, so the Jaccard contribution is 0 and the post-rerank
# score equals the base ``fused_score`` — letting the tests construct exact
# near-ties without depending on token arithmetic.
_UNRELATED = "completely unrelated content here"


class TestMetadataTieBreak:
    def test_existing_jaccard_behavior_unchanged(self):
        # Same inputs, metadata on vs off: identical ordering whenever no
        # metadata rule fires (here the symbol never matches the query).
        query = "describe the main module"
        results = [
            _result("a", _UNRELATED, fused_score=0.0200),
            _result("z", _UNRELATED, fused_score=0.0500),
        ]
        on = LexicalOverlapReranker().rerank(query, results, top_k=2)
        off = LexicalOverlapReranker(enable_metadata_tiebreak=False).rerank(query, results, top_k=2)
        assert [r.chunk_id for r in on] == [r.chunk_id for r in off] == ["z", "a"]
        assert "rerank_metadata_prior" not in on[0].signal_scores

    def test_near_tie_metadata_preference(self):
        query = "describe the i2c_master module"  # structural query
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, symbol_name=None),
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["a", "z"]
        assert reranked[0].signal_scores["rerank_metadata_prior"] == 1.0
        assert "rerank_metadata_prior" not in reranked[1].signal_scores

    def test_materially_better_relevance_always_wins(self):
        # Symbol match is present but the relevance gap (0.03) is far above
        # the 1e-4 epsilon: the better-scoring chunk must stay first and
        # no metadata decision may be recorded.
        query = "describe the i2c_master module"
        results = [
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("b", _UNRELATED, fused_score=0.0500, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["b", "a"]
        assert all("rerank_metadata_prior" not in r.signal_scores for r in reranked)

    def test_structural_query_matching_symbol_preferred_among_near_tied(self):
        query = "describe the i2c_master module"
        results = [
            _result("c", _UNRELATED, fused_score=0.0200, symbol_name="other_mod"),
            _result("b", _UNRELATED, fused_score=0.0200, symbol_name=None),
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=3)
        assert [r.chunk_id for r in reranked] == ["a", "c", "b"]
        assert reranked[0].signal_scores["rerank_metadata_prior"] == 1.0

    def test_architecture_intent_query_unlocks_symbol_preference(self):
        # Not a structural query, but ARCHITECTURE intent (existing logic)
        # identifies it as code-navigation — the same rule applies.
        query = "how does the clock generation work"
        results = [
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="clock_gen"),
            _result("z", _UNRELATED, fused_score=0.0200, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["a", "z"]
        assert reranked[0].signal_scores["rerank_metadata_prior"] == 1.0

    def test_non_structural_query_ignores_symbol_preference(self):
        # DEBUG intent + not structural: the symbol DOES match the query,
        # but the existing logic does not identify this as code-navigation,
        # so the tie-break must not fire.
        query = "why did the i2c_master fail"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert all("rerank_metadata_prior" not in r.signal_scores for r in reranked)

    def test_generated_does_not_globally_dominate(self):
        # (a) No blanket authored-vs-generated rule: a generated chunk with
        # no symbol match ranks purely by the deterministic chunk_id tie-break.
        query = "what is the weather today"  # GENERAL, not structural
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, category="generated"),
            _result("a", _UNRELATED, fused_score=0.0200, category="rtl"),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        # (b) Under a structural query, provenance stays inert: the generated
        # chunk wins because of its matching symbol, not because it is or
        # isn't generated, and the authored chunk is not globally preferred.
        query = "describe the i2c_master module"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, symbol_name=None, category="rtl"),
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master", category="generated"),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["a", "z"]

    def test_missing_metadata_preserves_ordering(self):
        query = "describe the i2c_master module"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200),
            _result("a", _UNRELATED, fused_score=0.0200),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert all("rerank_metadata_prior" not in r.signal_scores for r in reranked)

    def test_disabled_metadata_tiebreak_preserves_ordering(self):
        query = "describe the i2c_master module"
        results = [
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("z", _UNRELATED, fused_score=0.0200, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker(enable_metadata_tiebreak=False).rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert all("rerank_metadata_prior" not in r.signal_scores for r in reranked)

    def test_deterministic_repeated_execution(self):
        query = "describe the i2c_master module"
        results = [
            _result("b", _UNRELATED, fused_score=0.0200, symbol_name=None),
            _result("d", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("c", _UNRELATED, fused_score=0.0200, symbol_name=None),
        ]
        reranker = LexicalOverlapReranker()
        first = [r.chunk_id for r in reranker.rerank(query, results, top_k=4)]
        second = [r.chunk_id for r in reranker.rerank(query, results, top_k=4)]
        assert first == second

    def test_no_source_authority_double_counting(self):
        query = "describe the module"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, signal_scores={"source_authority": 0.3}),
            _result("a", _UNRELATED, fused_score=0.0200),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert reranked[0].signal_scores["source_authority"] == 0.3
        assert "rerank_metadata_prior" not in reranked[0].signal_scores

    def test_no_intent_double_counting(self):
        query = "describe the module"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, signal_scores={"intent_boost": 0.2}),
            _result("a", _UNRELATED, fused_score=0.0200),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert reranked[0].signal_scores["intent_boost"] == 0.2
        assert "rerank_metadata_prior" not in reranked[0].signal_scores

    def test_no_graph_double_counting(self):
        query = "describe the module"
        results = [
            _result("z", _UNRELATED, fused_score=0.0200, signal_scores={"graph_expansion": 0.5}),
            _result("a", _UNRELATED, fused_score=0.0200),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert reranked[0].signal_scores["graph_expansion"] == 0.5
        assert "rerank_metadata_prior" not in reranked[0].signal_scores

    def test_no_diversity_duplication_inside_reranker(self):
        # Same file + same category, near-tied: the reranker must NOT dedup
        # them and must not introduce a diversity decision of its own.
        query = "describe the module"
        results = [
            _result("a", _UNRELATED, fused_score=0.0200, relative_path="RTL/a.sv"),
            _result("b", _UNRELATED, fused_score=0.0200, relative_path="RTL/a.sv"),
        ]
        reranked = LexicalOverlapReranker().rerank(query, results, top_k=2)
        assert len(reranked) == 2
        assert all("diversity" not in r.signal_scores for r in reranked)

    def test_near_tie_epsilon_bounds_reordering(self):
        # A gap just inside epsilon is still a near-tie and may reorder;
        # a gap above epsilon never is.
        epsilon = 1e-4
        query = "describe the i2c_master module"
        results_in = [
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("z", _UNRELATED, fused_score=0.0201, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker(near_tie_epsilon=epsilon).rerank(query, results_in, top_k=2)
        # 0.0201 - 0.0200 = 1e-4 <= epsilon: preference allowed.
        assert [r.chunk_id for r in reranked] == ["a", "z"]

        results_out = [
            _result("a", _UNRELATED, fused_score=0.0200, symbol_name="i2c_master"),
            _result("z", _UNRELATED, fused_score=0.0210, symbol_name=None),
        ]
        reranked = LexicalOverlapReranker(near_tie_epsilon=epsilon).rerank(query, results_out, top_k=2)
        # 0.0210 - 0.0200 = 1e-3 > epsilon: relevance wins, no preference.
        assert [r.chunk_id for r in reranked] == ["z", "a"]
        assert all("rerank_metadata_prior" not in r.signal_scores for r in reranked)
