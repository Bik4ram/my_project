"""
Per-request database session dependency and job system dependencies.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Callable

from fastapi import Depends
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from semrag.core.settings import get_settings
from semrag.ingestion.embedding.factory import build_embedding_provider
from semrag.jobs.executor import ExecutorFactory, ThreadPoolJobExecutor
from semrag.jobs.manager import JobManager
from semrag.jobs.store import InMemoryJobStore, JobStore
from semrag.reasoning.llm.factory import build_llm
from semrag.services.services import QueryService
from semrag.storage.postgres.session import build_engine, build_session_factory

# Database session
_settings = get_settings()
_engine = build_engine(settings=_settings)
_session_factory = build_session_factory(_engine, settings=_settings)


def get_db_session() -> Generator[Session, None, None]:
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _build_job_store(settings) -> JobStore:
    """Pick the job store backend based on configuration.

    The selection is made at module load (consistent with the in-memory
    design that already pre-dates this change) — a process that runs
    with ``database`` semantics keeps them; a process that runs with
    ``memory`` semantics keeps those. Tests that want to swap stores
    call ``reset_job_store_for_tests`` after monkeypatching
    ``SEMRAG_JOB_STORE_BACKEND`` so the next ``get_job_store`` rebuilds
    with the new configuration. The single-replica in-memory mode is
    still available for dev/test by setting
    ``SEMRAG_JOB_STORE_BACKEND=memory``.
    """
    if settings.job_store_backend == "database":
        from semrag.jobs.postgres_store import PostgresJobStore

        return PostgresJobStore(_session_factory)
    return InMemoryJobStore()


# Job system dependencies (singletons)
_settings = get_settings()
_job_store = _build_job_store(_settings)
_job_manager = JobManager(_job_store)
_job_executor = ExecutorFactory.create_executor(_settings)


def reset_job_store_for_tests() -> None:
    """Drop the cached ``JobStore`` and ``JobManager`` singletons.

    Tests call this after monkeypatching ``SEMRAG_JOB_STORE_BACKEND``
    so the next ``get_job_store`` / ``get_job_manager`` call rebuilds
    with the freshly-read setting rather than the module-level store
    captured at import time. Production never hits this path.
    """
    global _job_store, _job_manager
    _job_store = _build_job_store(get_settings())
    _job_manager = JobManager(_job_store)


def get_engine() -> Engine:
    """The shared SQLAlchemy Engine. Used by health checks so they
    exercise the same connection pool production traffic uses, not a
    side-channel pool that could mask a real outage."""
    return _engine


def get_job_store() -> JobStore:
    return _job_store


def get_job_manager() -> JobManager:
    return _job_manager


def get_job_executor() -> ThreadPoolJobExecutor:  # Note: we return the concrete type for now, but ideally we'd return JobExecutor
    """The shared JobExecutor. Lazily re-creates after shutdown so
    tests that run multiple lifespans against the same module-level
    singletons (each TestClient startup calls the lifespan) don't
    see a perpetually-shut-down executor. Production never sees a
    re-creation — the process exits after the first shutdown."""
    global _job_executor
    if _job_executor is None or getattr(_job_executor, "_shutdown", False):
        _job_executor = ExecutorFactory.create_executor(_settings)
    return _job_executor  # type: ignore


def get_session_factory() -> Callable[[], Session]:
    """Return a callable that produces a new SQLAlchemy session."""
    return lambda: _session_factory()


def shutdown_resources() -> None:
    """Gracefully tear down every long-lived resource the process owns.

    Called from the FastAPI lifespan on shutdown so a SIGTERM gives each
    subsystem a chance to drain:

    - Job executor — waits for in-flight jobs (wait=True), then refuses
      new submissions via the _shutdown flag.
    - SQLAlchemy engine — closes the connection pool. Any in-flight
      request that still holds a Session will fail at commit time; the
      request has already returned by then in normal operation.

    Idempotent: a second call after the executor is already shut down
    is a no-op rather than a runtime error. Container restarts that
    double-invoke shutdown (uvicorn worker reload) stay quiet.

    The executor reference is dropped to None on shutdown so a
    subsequent get_job_executor() call lazily creates a fresh one.
    Production never sees this path (the process exits after one
    shutdown); tests do, when they create a second TestClient after
    the first one's lifespan ran.
    """
    global _job_executor
    executor = _job_executor
    if executor is not None and not getattr(executor, "_shutdown", False):
        try:
            executor.shutdown(wait=True)
        except Exception:  # noqa: BLE001 — never let shutdown raise
            pass
    engine = _engine
    if engine is not None:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass
    _job_executor = None


# Backwards-compatible alias for the previous shutdown function name.
shutdown_job_executor = shutdown_resources


def get_query_service(db: Session = Depends(get_db_session)) -> QueryService:
    """Dependency that provides a QueryService instance."""
    settings = get_settings()
    embedding_provider = build_embedding_provider(settings)
    llm = build_llm(settings)
    return QueryService(db, embedding_provider, llm, settings)