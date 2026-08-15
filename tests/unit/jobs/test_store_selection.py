"""
Tests for JobStore backend selection.

These tests pin the contract for ``SEMRAG_JOB_STORE_BACKEND``:

- Production default must be ``database`` (durable, multi-replica).
- Tests / dev opt back into ``memory`` via the env var.
- The module-level dependency singleton rebuilds when a test
  resets it via ``reset_job_store_for_tests``.
- The startup validator refuses ``memory`` in production with the
  same kind of error it raises for ``api_key_store=memory``.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

import pytest

from semrag.core.settings import Settings, reset_settings_cache


def test_production_default_is_database_store():
    """The default ``job_store_backend`` for a fresh Settings must be
    the database-backed implementation. The previous default of
    ``memory`` silently lost every queued/running job on restart and
    made multi-replica deployments race to enqueue / claim / re-enqueue
    the same work — the class of bug the persistent job store project
    ships to fix.
    """
    # Make sure no leftover env var influences this assertion.
    os.environ.pop("SEMRAG_JOB_STORE_BACKEND", None)
    reset_settings_cache()
    settings = Settings()
    assert settings.job_store_backend == "database"


def test_explicit_memory_backend_is_accepted():
    """The ``memory`` backend must remain available for tests / dev.
    Production validation rejects it, but the field still parses so a
    developer can opt in locally.
    """
    os.environ["SEMRAG_JOB_STORE_BACKEND"] = "memory"
    try:
        reset_settings_cache()
        settings = Settings()
        assert settings.job_store_backend == "memory"
    finally:
        del os.environ["SEMRAG_JOB_STORE_BACKEND"]
        reset_settings_cache()


def test_production_rejects_in_memory_job_store():
    """In-memory job store loses queued/running jobs on restart and is
    per-replica; production must use the database store or refuse to
    start.
    """
    from semrag.api.startup_validation import (
        RuntimeConfigurationError,
        validate_runtime_config,
    )

    settings = Settings(
        database_url="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag",
        embedding_provider="tfidf",
        embedding_base_url="",
        llm_provider="mock",
        llm_base_url="",
        llm_model="",
        environment="production",
        api_key_store="database",
        job_store_backend="memory",
        api_keys="real-prod-key-12345",
    )
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        validate_runtime_config(settings)
    assert "SEMRAG_JOB_STORE_BACKEND must be 'database' in production" in str(exc_info.value)


def test_production_accepts_database_job_store():
    """The database job store passes the production check."""
    from semrag.api.startup_validation import validate_runtime_config

    settings = Settings(
        database_url="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag",
        embedding_provider="tfidf",
        embedding_base_url="",
        llm_provider="mock",
        llm_base_url="",
        llm_model="",
        environment="production",
        api_key_store="database",
        job_store_backend="database",
        api_keys="real-prod-key-12345",
    )
    # Should not raise.
    validate_runtime_config(settings)


def test_build_job_store_returns_postgres_by_default(monkeypatch):
    """With the production default (``database``), the dependency wiring
    must return a ``PostgresJobStore``. Built via a stub session
    factory so the test never opens a real database connection.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase, PostgresJobStore
    from semrag.jobs.store import InMemoryJobStore
    from semrag.api import dependencies as deps

    # Build a session factory pointed at in-memory SQLite (the
    # PostgresJobStore round-trip is exercised in the dedicated
    # PostgresJobStore tests; here we only care about which class
    # the wiring picks).
    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Inject the stub session factory into the module so
    # _build_job_store uses it. This decouples the test from a
    # real Postgres server.
    monkeypatch.setattr(deps, "_session_factory", factory)
    monkeypatch.delenv("SEMRAG_JOB_STORE_BACKEND", raising=False)
    reset_settings_cache()

    store = deps._build_job_store(deps.get_settings())
    assert isinstance(store, PostgresJobStore)
    assert not isinstance(store, InMemoryJobStore)


def test_build_job_store_returns_memory_when_configured(monkeypatch):
    """When ``SEMRAG_JOB_STORE_BACKEND=memory``, the wiring must return
    an ``InMemoryJobStore`` even though Postgres is reachable.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase
    from semrag.jobs.store import InMemoryJobStore
    from semrag.api import dependencies as deps

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(deps, "_session_factory", factory)

    monkeypatch.setenv("SEMRAG_JOB_STORE_BACKEND", "memory")
    reset_settings_cache()
    try:
        store = deps._build_job_store(deps.get_settings())
        assert isinstance(store, InMemoryJobStore)
    finally:
        monkeypatch.delenv("SEMRAG_JOB_STORE_BACKEND", raising=False)
        reset_settings_cache()


def test_reset_job_store_for_tests_rebuilds_singleton(monkeypatch):
    """``reset_job_store_for_tests`` must drop the cached JobStore and
    rebuild it from the freshly-read settings. This is what makes the
    test fixture pattern of ``monkeypatch.setenv`` + ``reset_*`` work
    when the production default is now ``database``.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase
    from semrag.jobs.store import InMemoryJobStore
    from semrag.api import dependencies as deps

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(deps, "_session_factory", factory)

    # Force the production default and confirm we get a PostgresJobStore.
    monkeypatch.delenv("SEMRAG_JOB_STORE_BACKEND", raising=False)
    reset_settings_cache()
    deps.reset_job_store_for_tests()
    store_database = deps.get_job_store()
    assert type(store_database).__name__ == "PostgresJobStore"

    # Now flip the env var, reset, and confirm we get an in-memory store.
    monkeypatch.setenv("SEMRAG_JOB_STORE_BACKEND", "memory")
    reset_settings_cache()
    deps.reset_job_store_for_tests()
    store_memory = deps.get_job_store()
    assert isinstance(store_memory, InMemoryJobStore)

    # Cleanup
    monkeypatch.delenv("SEMRAG_JOB_STORE_BACKEND", raising=False)
    reset_settings_cache()
    deps.reset_job_store_for_tests()


# ---------------------------------------------------------------------------
# Cross-store contract parity — the same job flows through both stores.
# This is what makes "swap the backend in tests" honest: if the in-memory
# store silently drops a behaviour the database store relies on, the
# parity test fails before it ever reaches production.
# ---------------------------------------------------------------------------


def _run_lifecycle(store) -> None:
    """Drive a job through the same lifecycle on whatever store is
    passed in. Used by the parity tests to assert identical observable
    behaviour across backends."""
    from semrag.jobs.models import Job, JobState

    job = Job(
        job_id="parity-job-1",
        project_id="parity-proj",
        state=JobState.QUEUED,
        root_path="/tmp",
    )
    store.create_job(job)

    fetched = store.get_job("parity-job-1")
    assert fetched is not None
    assert fetched.state == JobState.QUEUED

    assert store.update_job("parity-job-1", state=JobState.RUNNING, progress=42)
    refreshed = store.get_job("parity-job-1")
    assert refreshed.state == JobState.RUNNING
    assert refreshed.progress == 42

    assert store.delete_job("parity-job-1")
    assert store.get_job("parity-job-1") is None


def test_in_memory_store_full_lifecycle():
    """Sanity check on the parity fixture's reference backend."""
    from semrag.jobs.store import InMemoryJobStore

    _run_lifecycle(InMemoryJobStore())


def test_postgres_store_full_lifecycle_parity():
    """The PostgresJobStore must exhibit the same lifecycle behaviour
    as the in-memory store. Uses SQLite for the SQLAlchemy engine so
    the test does not need a Postgres server, but the production
    path is identical.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase, PostgresJobStore

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    _run_lifecycle(PostgresJobStore(session_factory=Session))  # type: ignore[arg-type]


def test_postgres_job_store_survives_replica_restart():
    """The whole point of the PostgresJobStore is that two API
    replicas (or the same replica after a restart) see the same job
    rows. Two store instances sharing a session factory must round-
    trip a job through them; the in-memory store can never do this.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.models import Job, JobState
    from semrag.jobs.postgres_store import _JobBase, PostgresJobStore

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Replica A enqueues a job.
    replica_a = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    job = Job(
        job_id="restart-survives-1",
        project_id="proj",
        state=JobState.QUEUED,
        root_path="/tmp",
    )
    replica_a.create_job(job)

    # Replica B comes up after a "restart" — fresh store instance, same
    # database. Must see the job.
    replica_b = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    fetched = replica_b.get_job("restart-survives-1")
    assert fetched is not None
    assert fetched.state == JobState.QUEUED
    assert fetched.project_id == "proj"

    # Replica B claims the job (transitions to RUNNING); Replica A
    # must observe that transition.
    assert replica_b.update_job(
        "restart-survives-1", state=JobState.RUNNING, progress=10
    )
    refreshed = replica_a.get_job("restart-survives-1")
    assert refreshed.state == JobState.RUNNING
    assert refreshed.progress == 10
