"""
Shared fixtures for storage tests.

These tests require a live PostgreSQL instance — matching Part 1's
docker-compose Postgres service in a real deployment. `SEMRAG_TEST_DATABASE_URL`
points at it; the default assumes the same local setup used to develop
this part (`semrag`/`semrag_test`@localhost, database `semrag_test`,
already migrated via `alembic upgrade head`).

Isolation strategy: each test runs inside a transaction that's rolled back
on teardown (`session.rollback()` in the fixture's finally block, never
committed) — so tests don't need to truncate tables themselves or worry
about ordering/interference between tests, and the real `semrag_test`
database stays empty across test runs without needing a fixture that
drops/recreates the schema every time (expensive) or truncates every
table by hand (easy to forget one as new tables get added).
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from semrag.storage.postgres.session import build_engine

TEST_DATABASE_URL = os.environ.get(
    "SEMRAG_TEST_DATABASE_URL",
    "postgresql+psycopg2://semrag:semrag_test@localhost:5432/semrag_test",
)


@pytest.fixture(scope="session")
def pg_engine():
    engine = build_engine(TEST_DATABASE_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not available at {TEST_DATABASE_URL}: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(pg_engine):
    connection = pg_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
