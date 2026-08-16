"""
Tests for the production health endpoints.

These tests intentionally exercise the routes against the existing
`create_app()` factory so the wiring (path, response shape, status
code) is verified end-to-end. The `_check_database` probe is
overridden in tests that don't have a live Postgres so the suite
stays green in CI without a database.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semrag.api import health as health_module
from semrag.api.app import create_app
from semrag.api.health import _check_database, readyz


def _ok_db() -> tuple[bool, str]:
    return True, "ok"


def _bad_db() -> tuple[bool, str]:
    return False, "simulated outage"


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("SEMRAG_API_KEYS", "test-api-key-12345")
    monkeypatch.setenv("SEMRAG_EMBEDDING_PROVIDER", "tfidf")
    app = create_app()
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# /livez
# ---------------------------------------------------------------------------

def test_livez_returns_alive(api_client):
    response = api_client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_livez_does_not_touch_dependencies(api_client, monkeypatch):
    """Liveness must not require the database or any other dependency."""
    def _raise():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(health_module, "_check_database", _raise)
    response = api_client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


# ---------------------------------------------------------------------------
# /version
# ---------------------------------------------------------------------------

def test_version_returns_expected_shape(api_client):
    response = api_client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["application"] == "semrag"
    assert isinstance(body["version"], str)
    assert "python_version" in body
    assert "environment" in body
    assert "git_commit" in body
    assert "build_timestamp" in body


def test_version_uses_package_metadata(monkeypatch):
    """When the package is installed, the version comes from importlib.metadata."""
    monkeypatch.setattr(health_module, "_package_version", lambda: "9.9.9")
    result = health_module.version()
    assert result["version"] == "9.9.9"
    assert result["application"] == "semrag"


def test_version_falls_back_when_metadata_missing(monkeypatch):
    """When the package metadata isn't available, /version returns 0.0.0
    rather than erroring out — useful for tarball installs."""
    monkeypatch.setattr(health_module, "_package_version", lambda: None)
    result = health_module.version()
    assert result["version"] == "0.0.0"


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------

def test_readyz_returns_200_when_all_checks_pass(monkeypatch, api_client):
    monkeypatch.setattr(health_module, "_check_database", _ok_db)
    response = api_client.get("/readyz")
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"] == "ok"
    assert "llm" in body["checks"]
    assert "embedding" in body["checks"]
    assert "job_executor" in body["checks"]
    assert "cache" in body["checks"]
    assert "rate_limiter" in body["checks"]
    assert "version" in body
    assert "timestamp" in body


def test_readyz_returns_503_when_check_fails(monkeypatch, api_client):
    monkeypatch.setattr(health_module, "_check_database", _bad_db)
    response = api_client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"] == "simulated outage"


def test_readyz_exercises_real_database_check_when_db_is_up():
    """When a real Postgres is reachable, the default check should
    succeed. Falls back to skipping if no DB is available."""
    try:
        ok, msg = _check_database()
    except Exception:
        pytest.skip("database not reachable in this environment")
    assert ok is True
    assert msg == "ok"


def test_readyz_function_returns_unhealthy_when_db_down(monkeypatch):
    monkeypatch.setattr(health_module, "_check_database", _bad_db)
    from fastapi import Response

    response = Response()
    body = readyz(response)
    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"]["database"] == "simulated outage"


def test_health_alias_still_works(api_client):
    """The pre-existing /health endpoint is preserved for backwards
    compatibility (older Docker HEALTHCHECK directives depend on it)."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}