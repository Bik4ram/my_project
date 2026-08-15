"""
API test fixtures.

Overrides get_db_session to yield the same transactional db_session
fixture every other live-Postgres test in this series uses (rolled back
on teardown, never committed) - so API tests get the same isolation
guarantee without a second database connection strategy. SEMRAG_API_KEYS
is set to a known test value per-test since require_api_key reads it
live from the environment on every call (not cached at import time),
which is exactly what makes this override possible without reimporting
any module.

The job store defaults to ``database`` in production. Tests need the
in-memory store (no Postgres dependency), so the fixture pins
``SEMRAG_JOB_STORE_BACKEND=memory`` and calls
``reset_job_store_for_tests()`` so the next ``get_job_store`` rebuilds
the module-level singletons with the freshly-read setting. Tests that
specifically exercise the database-backed job store override this
fixture.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semrag.api.app import create_app
from semrag.api.dependencies import (
    get_db_session,
    reset_job_store_for_tests,
)
from semrag.core.settings import reset_settings_cache

TEST_API_KEY = "test-api-key-12345"


@pytest.fixture
def api_client(db_session, monkeypatch):
    monkeypatch.setenv("SEMRAG_API_KEYS", TEST_API_KEY)
    monkeypatch.setenv("SEMRAG_EMBEDDING_PROVIDER", "tfidf")
    # Pin the in-memory job store for the duration of the test. The
    # dependency singleton was built at module import time against
    # whatever the default was then; ``reset_job_store_for_tests``
    # rebuilds it now that the env var is set.
    monkeypatch.setenv("SEMRAG_JOB_STORE_BACKEND", "memory")
    reset_settings_cache()
    reset_job_store_for_tests()

    app = create_app()

    def _override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = _override_get_db_session

    # Deliberately does NOT set a default Authorization header on the
    # client — httpx merges per-call headers with client defaults rather
    # than letting an empty per-call dict clear them, so a client-wide
    # default would make it impossible for any test to genuinely send an
    # unauthenticated request. Tests that need auth pass `auth_headers`
    # explicitly instead (see below) — caught by
    # test_protected_endpoint_rejects_missing_auth actually failing
    # against the first version of this fixture, not assumed to work.
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}
