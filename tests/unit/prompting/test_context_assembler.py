from semrag.reasoning.prompting.context_assembler import assemble_context
from semrag.reasoning.prompting.token_estimator import HeuristicTokenEstimator
from semrag.retrieval.models import RetrievalResult


def _result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, project_id="proj1", relative_path=f"RTL/{chunk_id}.sv",
        chunk_type="sv_module", text=text, fused_score=score,
    )


def test_all_results_fit_within_generous_budget():
    results = [_result("a", "short text", 1.0), _result("b", "also short", 0.9)]
    included, context_text, used = assemble_context(results, budget_tokens=1000, estimator=HeuristicTokenEstimator())
    assert len(included) == 2
    assert "short text" in context_text
    assert "also short" in context_text


def test_lower_ranked_results_dropped_when_budget_is_tight():
    results = [
        _result("a", "x" * 40, 1.0),
        _result("b", "y" * 40, 0.9),
        _result("c", "z" * 40, 0.8),
    ]
    included, _, used = assemble_context(results, budget_tokens=15, estimator=HeuristicTokenEstimator())
    assert len(included) == 1
    assert included[0].chunk_id == "a"
    assert used <= 15


def test_zero_budget_includes_nothing():
    results = [_result("a", "some text", 1.0)]
    included, context_text, used = assemble_context(results, budget_tokens=0, estimator=HeuristicTokenEstimator())
    assert included == []
    assert context_text == ""
    assert used == 0


def test_empty_results_produce_empty_context():
    included, context_text, used = assemble_context([], budget_tokens=1000, estimator=HeuristicTokenEstimator())
    assert included == []
    assert context_text == ""


def test_citations_are_included_in_context_text():
    results = [_result("a", "module body text", 1.0)]
    _, context_text, _ = assemble_context(results, budget_tokens=1000, estimator=HeuristicTokenEstimator())
    assert "[Source: RTL/a.sv]" in context_text


def test_never_truncates_a_chunk_mid_text():
    results = [
        _result("big", "x" * 1000, 1.0),
        _result("small", "y" * 10, 0.9),
    ]
    included, context_text, _ = assemble_context(results, budget_tokens=20, estimator=HeuristicTokenEstimator())
    assert [r.chunk_id for r in included] == ["small"]
    assert "x" not in context_text
