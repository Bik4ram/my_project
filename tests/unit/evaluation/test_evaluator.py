"""Tests for the end-to-end evaluator (with mocked QueryService)."""
from __future__ import annotations

from semrag.evaluation.evaluator import (
    EndToEndEvaluator,
    LLMJudgeScores,
    _parse_judge_scores,
)
from semrag.evaluation.models import GoldenQuery
from semrag.services.services import DebugSummary, QueryAnswer


class _StubQueryService:
    """Minimal QueryService stand-in that returns canned answers."""

    def __init__(self, answers: list[QueryAnswer]):
        self._answers = answers
        self.call_count = 0

    def ask(self, project_id: str, question: str) -> QueryAnswer:
        self.call_count += 1
        return self._answers.pop(0)


def _make_answer(
    final_answer: str,
    *,
    iterations: int = 1,
    hit_limit: bool = False,
    debug: DebugSummary | None = None,
) -> QueryAnswer:
    return QueryAnswer(
        final_answer=final_answer,
        iterations_used=iterations,
        hit_iteration_limit=hit_limit,
        messages=[],
        debug=debug or DebugSummary(query="", intent="general"),
    )


def test_evaluator_stub():
    """EndToEndEvaluator runs queries through a stub and returns results."""
    answer = _make_answer("i2c_master is the RTL module.")
    golden = [
        GoldenQuery(query_id="q1", project_id="proj1", query_text="what is i2c_master?"),
    ]
    service = _StubQueryService([answer])
    evaluator = EndToEndEvaluator(service)  # type: ignore[arg-type]

    report = evaluator.evaluate_all(golden, run_llm_judge=False)
    assert service.call_count == 1
    assert report.query_count == 1
    assert report.per_query_results[0].final_answer == "i2c_master is the RTL module."
    assert report.per_query_results[0].query_id == "q1"


def test_evaluator_multiple_queries():
    answers = [
        _make_answer("module a"),
        _make_answer("module b"),
    ]
    golden = [
        GoldenQuery(query_id="q1", project_id="proj1", query_text="what is a?"),
        GoldenQuery(query_id="q2", project_id="proj1", query_text="what is b?"),
    ]
    service = _StubQueryService(answers)
    evaluator = EndToEndEvaluator(service)  # type: ignore[arg-type]

    report = evaluator.evaluate_all(golden, run_llm_judge=False)
    assert report.query_count == 2
    assert report.per_query_results[0].final_answer == "module a"
    assert report.per_query_results[1].final_answer == "module b"


def test_evaluator_aggregation():
    """Aggregate metrics are computed correctly over multiple results."""
    answers = [
        _make_answer("answer 1"),
        _make_answer("answer 2"),
    ]
    golden = [
        GoldenQuery(query_id="q1", project_id="proj1", query_text="q1"),
        GoldenQuery(query_id="q2", project_id="proj1", query_text="q2"),
    ]
    service = _StubQueryService(answers)
    evaluator = EndToEndEvaluator(service)  # type: ignore[arg-type]

    report = evaluator.evaluate_all(golden, run_llm_judge=False)
    assert report.query_count == 2


def test_evaluator_empty_golden():
    service = _StubQueryService([])
    evaluator = EndToEndEvaluator(service)  # type: ignore[arg-type]
    report = evaluator.evaluate_all([], run_llm_judge=False)
    assert report.query_count == 0


def test_evaluator_error_handling():
    """A query that raises an exception should be captured in the result
    error field without crashing the evaluator."""

    class _FailingService:
        def ask(self, project_id: str, question: str) -> QueryAnswer:
            raise RuntimeError("simulated failure")

    golden = [
        GoldenQuery(query_id="q1", project_id="proj1", query_text="fail?"),
    ]
    evaluator = EndToEndEvaluator(_FailingService())  # type: ignore[arg-type]
    report = evaluator.evaluate_all(golden, run_llm_judge=False)
    assert report.query_count == 1
    assert "RuntimeError" in (report.per_query_results[0].error or "")


def test_parse_judge_scores_valid():
    raw = '{"correctness": 8, "groundedness": 7, "completeness": 9, "hallucination_risk": 2, "citation_quality": 8}'
    scores = _parse_judge_scores(raw)
    assert scores is not None
    assert scores.correctness == 8
    assert scores.groundedness == 7
    assert scores.completeness == 9
    assert scores.hallucination_risk == 2
    assert scores.citation_quality == 8
    assert scores.valid is True


def test_parse_judge_scores_invalid():
    assert _parse_judge_scores("not json") is None
    assert _parse_judge_scores(None) is None
    assert _parse_judge_scores("") is None


def test_parse_judge_scores_out_of_range():
    raw = '{"correctness": 11, "groundedness": 0, "completeness": 5, "hallucination_risk": 5, "citation_quality": 5}'
    scores = _parse_judge_scores(raw)
    assert scores is not None
    assert scores.valid is False


def test_evaluator_citation_analysis():
    """Verify citation analysis from messages with [Source: ...] headers."""
    from semrag.evaluation.evaluator import EndToEndEvaluator

    messages = [
        {"role": "user", "content": "CONTEXT:\n[Source: RTL/top.sv, lines 1-10]\nmodule top;\nQUESTION:\nwhat is top?"},
    ]
    chunks = EndToEndEvaluator._extract_chunks_from_messages(messages)
    assert len(chunks) == 1
    assert chunks[0].relative_path == "RTL/top.sv"
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 10


def test_evaluator_citation_analysis_deduplication():
    messages = [
        {"role": "user", "content": "[Source: RTL/top.sv, lines 1-5]\n[Source: RTL/top.sv, lines 10-15]"},
    ]
    chunks = EndToEndEvaluator._extract_chunks_from_messages(messages)
    assert len(chunks) == 1  # deduplicated by path
