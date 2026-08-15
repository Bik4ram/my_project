"""Tests for the Prometheus metrics module."""
from __future__ import annotations

import pytest

from semrag.core import metrics as metrics_module


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the metrics registry between tests so observations don't leak."""
    metrics_module.reset_registry()
    yield
    metrics_module.reset_registry()


def test_render_metrics_returns_valid_exposition():
    """A fresh registry renders Prometheus exposition format."""
    body, content_type = metrics_module.render_metrics()
    assert content_type.startswith("text/plain")
    # Up-time is a single sample by default; the body is non-empty.
    assert b"semrag_uptime_seconds" in body or b"semrag_health_up" in body


def test_observe_http_increments_counters():
    metrics_module.observe_http(
        method="GET",
        path="/v1/projects/{project_id}/query",
        status=200,
        duration_s=0.123,
        request_bytes=128,
        response_bytes=2048,
    )
    body, _ = metrics_module.render_metrics()
    # Counter name and label values appear in the exposition format.
    assert b'semrag_http_requests_total' in body
    assert b'semrag_http_request_duration_seconds' in body
    assert b'semrag_http_request_size_bytes' in body
    assert b'semrag_http_response_size_bytes' in body
    # The status label must be a string (Prometheus forbids ints).
    assert b'status="200"' in body


def test_observe_retrieval_records_histogram():
    metrics_module.observe_retrieval("vector", 0.045)
    metrics_module.observe_rerank(0.123)
    metrics_module.observe_embedding("ollama", 0.456)
    metrics_module.observe_llm("ollama", "qwen2.5:7b", 1.234)
    metrics_module.observe_tool("read_file", 0.05)
    body, _ = metrics_module.render_metrics()
    assert b'semrag_rag_retrieval_duration_seconds_count{stage="vector"}' in body
    assert b'semrag_rag_rerank_duration_seconds_count' in body
    assert b'semrag_rag_embedding_duration_seconds_count{provider="ollama"}' in body
    # Prometheus renders labels in alphabetical order, so the model label
    # appears before the provider label.
    assert b'semrag_rag_llm_duration_seconds_count{model="qwen2.5:7b",provider="ollama"}' in body
    assert b'semrag_rag_tool_duration_seconds_count{tool="read_file"}' in body


def test_cache_and_rate_limit_helpers():
    metrics_module.inc_cache_hit("memory")
    metrics_module.inc_cache_miss("memory")
    metrics_module.inc_cache_eviction("memory")
    metrics_module.inc_rate_limit_accepted()
    metrics_module.inc_rate_limit_rejected()
    body, _ = metrics_module.render_metrics()
    assert b'semrag_cache_hits_total{backend="memory"}' in body
    assert b'semrag_cache_misses_total{backend="memory"}' in body
    assert b'semrag_cache_evictions_total{backend="memory"}' in body
    assert b'semrag_rate_limit_accepted_total' in body
    assert b'semrag_rate_limit_rejected_total' in body


def test_health_and_uptime_helpers():
    metrics_module.set_health_up("database", True)
    metrics_module.set_health_up("llm", False)
    metrics_module.set_uptime(123.45)
    body, _ = metrics_module.render_metrics()
    assert b'semrag_health_up{component="database"} 1.0' in body
    assert b'semrag_health_up{component="llm"} 0.0' in body
    assert b'semrag_uptime_seconds 123.45' in body


def test_indexing_job_helpers():
    metrics_module.INDEXING_JOBS_QUEUED.labels(project_id="p1").inc()
    metrics_module.INDEXING_JOBS_COMPLETED.labels(project_id="p1").inc()
    metrics_module.INDEXING_JOBS_FAILED.labels(project_id="p2").inc()
    metrics_module.INDEXING_DURATION.labels(project_id="p1").observe(5.5)
    body, _ = metrics_module.render_metrics()
    assert b'semrag_indexing_jobs_queued_total{project_id="p1"}' in body
    assert b'semrag_indexing_jobs_completed_total{project_id="p1"}' in body
    assert b'semrag_indexing_jobs_failed_total{project_id="p2"}' in body


def test_reset_registry_clears_observations():
    metrics_module.set_uptime(99.0)
    body1, _ = metrics_module.render_metrics()
    assert b'semrag_uptime_seconds 99.0' in body1

    metrics_module.reset_registry()
    body2, _ = metrics_module.render_metrics()
    # After reset, the gauge is back to its initial state (the
    # default Counter/Histogram/Gauge value, which is 0 / no
    # observations).
    assert b'semrag_uptime_seconds 99.0' not in body2


def test_http_exception_counter_labels_exc_class():
    metrics_module.inc_http_exception("POST", "/v1/projects/{project_id}/query", "ValueError")
    body, _ = metrics_module.render_metrics()
    assert b'semrag_http_exceptions_total' in body
    assert b'exception="ValueError"' in body