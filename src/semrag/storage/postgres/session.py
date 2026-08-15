"""
Engine/session factory.

Connection string is read from `SEMRAG_DATABASE_URL`, defaulting to the
local-development URL matching Part 1's `docker-compose.yml` Postgres
service. No hardcoded production credentials anywhere in source — this is
the one place a connection string is assembled, so it's also the one
place that needs to change for a different deployment target.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg2://semrag:semrag@localhost:5432/semrag"


def get_database_url() -> str:
    return os.environ.get("SEMRAG_DATABASE_URL", DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None, settings: Settings | None = None):
    url = database_url or get_database_url()
    if settings is None:
        # Fallback to default pool settings if settings not provided
        return create_engine(url, pool_pre_ping=True)
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_timeout=settings.db_pool_timeout,
    )


def build_session_factory(engine=None, settings: Settings | None = None) -> sessionmaker[Session]:
    if engine is None:
        engine = build_engine(settings=settings)
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None = None, settings: Settings | None = None):
    """Standard unit-of-work context manager: commits on clean exit, rolls
    back on exception, always closes. Every repository method in this
    package takes a `Session` argument rather than managing its own —
    callers control transaction boundaries (e.g. Part 6's IR builder call
    and Part 5's file-sync call can share one transaction), this module
    just provides the default way to get one when a caller doesn't need
    that control."""
    factory = session_factory or build_session_factory(settings=settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
