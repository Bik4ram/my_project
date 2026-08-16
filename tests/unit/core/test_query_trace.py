"""Tests for QueryTrace serialization and shape."""
from __future__ import annotations

from semrag.core.query_trace import QueryTrace, QueryTraceStage


def test_query_trace_defaults():
    trace = QueryTrace()
    assert trace.run_id == ""
    assert trace.project_id == ""
    assert trace.query_text == ""
    assert trace.preprocessing.started_at == 0.0
    assert trace.retrieval.data == {}
    assert trace.final_answer is None
    assert trace.iterations_used == 0
    assert trace.hit_iteration_limit is False


def test_query_trace_incremental_population():
    trace = QueryTrace(run_id="r1", project_id="p1", query_text="test query")

    trace.preprocessing.data = {"is_structural": True, "file_index_included": True}
    trace.intent_classification.data = {"intent": "structural"}

    trace.embedding.data = {"model": "nomic-embed-text", "provider": "ollama"}
    trace.retrieval.data = {"query": "test query", "chunks": []}

    trace.context_assembly.data = {"included_count": 5, "excluded_count": 2}
    trace.token_estimation.data = {"estimated_total_tokens": 1500}

    trace.llm_stage.data = {"provider": "ollama", "model": "qwen2.5:7b"}
    trace.final_answer = "The answer is 42."
    trace.iterations_used = 1
    trace.total_duration_ms = 1234.5

    assert trace.run_id == "r1"
    assert trace.preprocessing.data["is_structural"] is True
    assert trace.retrieval.data["query"] == "test query"
    assert trace.final_answer == "The answer is 42."


def test_query_trace_stage_durations():
    stage = QueryTraceStage()
    stage.started_at = 1000.0
    stage.completed_at = 1001.5
    stage.duration_ms = 1500.0
    stage.data = {"key": "value"}

    assert stage.duration_ms == 1500.0
    assert stage.data["key"] == "value"


def test_query_trace_citation_extraction_stage():
    trace = QueryTrace()
    trace.citation_extraction.data = {
        "cited_paths": ["RTL/top.sv"],
        "valid_count": 1,
        "fabricated_count": 0,
        "fabrication_rate": 0.0,
    }
    assert trace.citation_extraction.data["valid_count"] == 1
    assert trace.citation_extraction.data["fabrication_rate"] == 0.0


def test_query_trace_json_serialization():
    trace = QueryTrace(
        run_id="r2",
        project_id="p2",
        query_text="what is the architecture",
        timestamp=1700000000.0,
    )
    trace.retrieval.data = {"query": "what is the architecture", "chunks": []}
    trace.final_answer = "The architecture is..."
    trace.total_duration_ms = 5000.0

    serialized = trace.model_dump(mode="json")
    assert serialized["run_id"] == "r2"
    assert serialized["project_id"] == "p2"
    assert serialized["final_answer"] == "The architecture is..."
    assert serialized["total_duration_ms"] == 5000.0

    # Round-trip
    restored = QueryTrace.model_validate(serialized)
    assert restored.run_id == "r2"
    assert restored.total_duration_ms == 5000.0


def test_query_trace_stage_data_persistence():
    trace = QueryTrace()
    trace.preprocessing.data = {
        "is_structural": False,
        "file_index_included": False,
        "file_index_length": 0,
    }
    trace.intent_classification.data = {"intent": "general"}
    trace.embedding.data = {"model": "test-model", "provider": "test-provider"}
    trace.query_expansion.data = {"expanded_queries": []}

    serialized = trace.model_dump(mode="json")
    assert serialized["preprocessing"]["data"]["is_structural"] is False
    assert serialized["intent_classification"]["data"]["intent"] == "general"
    assert serialized["embedding"]["data"]["model"] == "test-model"
    assert serialized["query_expansion"]["data"]["expanded_queries"] == []


def test_query_trace_extra_fields_ignored():
    trace = QueryTrace.model_validate({
        "run_id": "r1",
        "unknown_field": "should be ignored",
        "preprocessing": {"data": {"extra_nested": "ignored"}},
    })
    assert trace.run_id == "r1"
    # extra fields should be silently ignored by model_config
    assert not hasattr(trace, "unknown_field")


def test_query_trace_all_stages_present():
    """Ensure all 22 pipeline stages are represented in the model."""
    trace = QueryTrace()
    assert hasattr(trace, "preprocessing")  # Stages 1-2
    assert hasattr(trace, "intent_classification")  # Stage 3
    assert hasattr(trace, "query_expansion")  # Stage 4
    assert hasattr(trace, "embedding")  # Stage 5
    assert hasattr(trace, "retrieval")  # Stages 6-15
    assert hasattr(trace, "context_assembly")  # Stage 16
    assert hasattr(trace, "prompt_construction")  # Stage 17
    assert hasattr(trace, "token_estimation")  # Stage 18
    assert hasattr(trace, "llm_stage")  # Stages 19-20
    assert hasattr(trace, "citation_extraction")  # Stage 21
    # Stage 22 is directly on the trace (final_answer, iterations_used, etc.)
    assert hasattr(trace, "final_answer")
    assert hasattr(trace, "iterations_used")
    assert hasattr(trace, "hit_iteration_limit")


def test_query_trace_stage_defaults():
    stage = QueryTraceStage()
    assert stage.data == {}
    assert stage.duration_ms == 0.0
    assert stage.started_at == 0.0
    assert stage.completed_at == 0.0


def test_query_trace_empty_string_fields():
    trace = QueryTrace(
        run_id="",
        project_id="",
        query_text="",
    )
    serialized = trace.model_dump(mode="json")
    assert serialized["run_id"] == ""
    assert serialized["project_id"] == ""
    assert serialized["query_text"] == ""
