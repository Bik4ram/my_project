"""
Fixtures for Neo4j-backed tests.

These tests are written and ready but were NOT run in this delivery — no
Neo4j instance was reachable in this sandbox (see `neo4j_store.py`'s
module docstring). The skip-gracefully pattern here matches
`tests/unit/conftest.py`'s Postgres fixture exactly, so the moment a real
Neo4j instance is reachable (e.g. via the `docker-compose.yml` Part 16
will eventually build), these tests run for real with zero code changes —
just point `SEMRAG_TEST_NEO4J_URI` at it.
"""
from __future__ import annotations

import os

import pytest

from semrag.storage.graph.neo4j_store import Neo4jGraphStore

NEO4J_URI = os.environ.get("SEMRAG_TEST_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("SEMRAG_TEST_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("SEMRAG_TEST_NEO4J_PASSWORD", "semrag_test")


@pytest.fixture(scope="session")
def neo4j_store():
    store = Neo4jGraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        store._driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available at {NEO4J_URI}: {exc}")
    store.ensure_constraints()
    yield store
    store.close()
