"""Tests for the /metrics endpoint and the health gauges populated by /readyz."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semrag.api import health as health_module
from semrag.api.app import create_app
from semrag.core import metrics as metrics_module


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the metrics registry before each test so observations don't leak."""
    metrics_module.reset_registry()
    yield
    metrics_module.reset_registry()


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("SEMRAG_API_KEYS", "test-api-key-12345")
    monkeypatch.setenv("SEMRAG_EMBEDDING_PROVIDER", "tfidf")
    app = create_app()
    with TestClient(app) as client:
        yield client


def _ok_db() -> tuple[bool, str]:
    return True, "ok"


def test_metrics_endpoint_returns_prometheus_exposition(api_client, monkeypatch):
    monkeypatch.setattr(health_module, "_check_database", _ok_db)
    response = api_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.content
    # Standard Wave 3 metric names are present.
    assert b"semrag_http_requests_total" in body
    assert b"semrag_uptime_seconds" in body
    assert b"semrag_health_up" in body


def test_metrics_endpoint_disabled_returns_404(monkeypatch):
    monkeypatch.setenv("SEMRAG_METRICS_ENABLED", "false")
    from semrag.core.settings import reset_settings_cache
    reset_settings_cache()
    try:
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/metrics")
            assert response.status_code == 404
    finally:
        monkeypatch.delenv("SEMRAG_METRICS_ENABLED", raising=False)
        reset_settings_cache()


def test_readyz_populates_health_gauges(api_client, monkeypatch):
    monkeypatch.setattr(health_module, "_check_database", _ok_db)
    response = api_client.get("/readyz")
    assert response.status_code == 200
    body, _ = metrics_module.render_metrics()
    # After /readyz, the database health gauge is 1.
    assert b'semrag_health_up{component="database"} 1.0' in body


def test_readyz_failure_marks_gauges_zero(api_client, monkeypatch):
    def _bad_db():
        return False, "simulated outage"

    monkeypatch.setattr(health_module, "_check_database", _bad_db)
    response = api_client.get("/readyz")
    assert response.status_code == 503
    body, _ = metrics_module.render_metrics()
    assert b'semrag_health_up{component="database"} 0.0' in body


def test_metrics_endpoint_records_http_request(monkeypatch, api_client):
    monkeypatch.setattr(health_module, "_check_database", _ok_db)
    # A regular request through the API should record metrics
    # (request count, latency, etc.) — verify by issuing a request
    # and inspecting the exposition body.
    api_client.get("/livez")
    body, _ = metrics_module.render_metrics()
    assert b'semrag_http_requests_total{method="GET",path="/livez",status="200"}' in body
    assert b'semrag_http_request_duration_seconds_count' in body


def test_uptime_gauge_present(monkeypatch, api_client):
    monkeypatch.setattr(health_module, "_check_database", _ok_db)
    api_client.get("/metrics")
    body, _ = metrics_module.render_metrics()
    # The uptime gauge is set every time /metrics is hit.
    assert b"semrag_uptime_seconds" in body