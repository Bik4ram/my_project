"""
Tests for the new persistent / distributed backend pieces.

Each test is small and uses only the in-memory implementations or
fakeness so that the test suite doesn't require Postgres or Redis to
be running. The Postgres / Redis implementations are exercised
through interface contracts (they share the same API as the in-memory
implementations) and through direct construction with a stub client.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from datetime import timezone
from typing import Any, Dict, List, Optional

import pytest

from semrag.api.api_key import APIKey, APIKeyStatus, hash_api_key
from semrag.api.api_key_store import PostgresAPIKeyStore
from semrag.api.query_cache import TTLQueryCache, RedisQueryCache, QueryCache
from semrag.api.rate_limit import MemoryRateLimiter, RedisRateLimiter, RateLimiter
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.redis_cache import RedisEmbeddingCache
from semrag.jobs.manager import JobManager
from semrag.jobs.models import Job, JobState
from semrag.jobs.postgres_store import PostgresJobStore, IndexingJobRow
from semrag.jobs.store import InMemoryJobStore, JobStore


# ---------------------------------------------------------------------------
# F1: hash_api_key
# ---------------------------------------------------------------------------


def test_hash_api_key_is_deterministic():
    """Hashing the same key twice must yield the same digest."""
    assert hash_api_key("foo") == hash_api_key("foo")


def test_hash_api_key_collision_resistant():
    """Different keys must produce different digests."""
    assert hash_api_key("foo") != hash_api_key("bar")


def test_hash_api_key_does_not_return_plaintext():
    """The digest must not be the plaintext key."""
    assert hash_api_key("secret") != "secret"


# ---------------------------------------------------------------------------
# F1: PostgresAPIKeyStore — interface parity with the in-memory store
# ---------------------------------------------------------------------------


def test_postgres_api_key_store_round_trip_in_memory():
    """The PostgresAPIKeyStore round-trips against a real SQLAlchemy
    SQLite engine. SQLite gives us genuine WHERE-clause semantics so
    the tests don't need a custom stub; ``JSON`` columns degrade to
    plain text under SQLite, which is enough for the round-trip."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.api.api_key_store import APIKeyRow, _APIKeyBase

    engine = create_engine("sqlite:///:memory:")
    _APIKeyBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = PostgresAPIKeyStore(session_factory=Session)  # type: ignore[arg-type]
    api_key = APIKey(
        key="plaintext-bearer",
        name="test",
        created_at=dt.datetime.now(timezone.utc).replace(tzinfo=None),
    )
    asyncio.run(store.create_key(api_key))

    fetched = asyncio.run(store.get_key("plaintext-bearer"))
    assert fetched is not None
    assert fetched.name == "test"

    fetched.description = "updated"
    assert asyncio.run(store.update_key("plaintext-bearer", fetched))

    assert asyncio.run(store.delete_key("plaintext-bearer"))
    assert asyncio.run(store.get_key("plaintext-bearer")) is None


def test_postgres_api_key_store_duplicate_rejected():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.api.api_key_store import _APIKeyBase

    engine = create_engine("sqlite:///:memory:")
    _APIKeyBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = PostgresAPIKeyStore(session_factory=Session)  # type: ignore[arg-type]
    api_key = APIKey(
        key="dup",
        name="first",
        created_at=dt.datetime.now(timezone.utc).replace(tzinfo=None),
    )
    asyncio.run(store.create_key(api_key))
    with pytest.raises(ValueError):
        asyncio.run(store.create_key(api_key))


def test_postgres_api_key_store_list_keys():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.api.api_key_store import _APIKeyBase

    engine = create_engine("sqlite:///:memory:")
    _APIKeyBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = PostgresAPIKeyStore(session_factory=Session)  # type: ignore[arg-type]
    asyncio.run(store.create_key(APIKey(key="a", name="a", created_at=dt.datetime.now(timezone.utc).replace(tzinfo=None))))
    asyncio.run(store.create_key(APIKey(key="b", name="b", created_at=dt.datetime.now(timezone.utc).replace(tzinfo=None))))
    keys = asyncio.run(store.list_keys())
    assert len(keys) == 2


# ---------------------------------------------------------------------------
# F2: PostgresJobStore — interface parity
# ---------------------------------------------------------------------------


def test_postgres_job_store_lifecycle():
    """create -> get -> update -> delete covers the lifecycle."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    store.create_job(Job(job_id="j1", project_id="proj", state=JobState.QUEUED, root_path=""))
    fetched = store.get_job("j1")
    assert fetched is not None
    assert fetched.state == JobState.QUEUED

    assert store.update_job("j1", state=JobState.RUNNING, progress=42)
    refreshed = store.get_job("j1")
    assert refreshed.state == JobState.RUNNING
    assert refreshed.progress == 42

    assert store.delete_job("j1")
    assert store.get_job("j1") is None


def test_postgres_job_store_list_filter():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    store = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    store.create_job(Job(job_id="a1", project_id="proj1", state=JobState.QUEUED, root_path=""))
    store.create_job(Job(job_id="a2", project_id="proj2", state=JobState.QUEUED, root_path=""))
    only_proj1 = store.list_jobs(project_id="proj1")
    assert {j.job_id for j in only_proj1} == {"a1"}


def test_postgres_job_store_persists_across_instances():
    """Two stores sharing a session factory see the same data — this is
    the multi-replica contract that the in-memory store cannot fulfil."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from semrag.jobs.postgres_store import _JobBase

    engine = create_engine("sqlite:///:memory:")
    _JobBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    store_a = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    store_a.create_job(
        Job(job_id="j1", project_id="proj", state=JobState.QUEUED, root_path="")
    )
    store_b = PostgresJobStore(session_factory=Session)  # type: ignore[arg-type]
    assert store_b.get_job("j1") is not None


# ---------------------------------------------------------------------------
# F3: Redis-backed components — stubbed redis client
# ---------------------------------------------------------------------------


class _StubRedis:
    """A minimal in-memory stub that mirrors the parts of redis-py used
    by the wrappers. Not a complete Redis emulation — just enough to
    prove the cache / rate-limit / embedding-cache wrappers behave
    correctly under the public interface.
    """

    def __init__(self):
        self.kv: Dict[str, Any] = {}
        self.sets: Dict[str, set] = {}
        self.expires: Dict[str, int] = {}

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value):
        self.kv[key] = value

    def setex(self, key, ttl, value):
        self.kv[key] = value
        self.expires[key] = ttl

    def expire(self, key, ttl):
        self.expires[key] = ttl
        return 1

    def mget(self, keys):
        return [self.kv.get(k) for k in keys]

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return list(self.sets.get(key, set()))

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.sets.pop(k, None)
            self.expires.pop(k, None)
        return 1

    def pipeline(self):
        return _StubPipeline(self)

    def scan_iter(self, match="*", count=10):
        prefix = match.rstrip("*")
        for key in list(self.kv.keys()) + list(self.sets.keys()):
            if key.startswith(prefix):
                yield key

    async def ascan_iter(self, match="*", count=10):
        for key in self.scan_iter(match=match, count=count):
            yield key

    async def adelete(self, *keys):
        return self.delete(*keys)

    def register_script(self, source):
        def _runner(keys, args):
            now, window, limit, member = args
            key = keys[0]
            entries = [e for e in self.kv.get(key, []) if now - float(e.split(":")[0]) <= window]
            if len(entries) >= limit:
                return 0
            entries.append(f"{now}:{member}")
            self.kv[key] = entries
            return 1

        async def _arunner(keys, args):
            return _runner(keys, args)

        return _arunner


class _StubPipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def __getattr__(self, name):
        def _op(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self

        return _op

    def execute(self):
        results = []
        for name, args, kwargs in self._ops:
            method = getattr(self._client, name)
            results.append(method(*args, **kwargs))
        return results


def test_redis_query_cache_set_and_get():
    """Round-trip a QueryResponse through the Redis cache."""
    from semrag.api.schemas import QueryResponse, ChatMessage

    client = _StubRedis()
    cache = RedisQueryCache(client=client, ttl_seconds=10)
    response = QueryResponse(
        query="q",
        messages=[ChatMessage(role="user", content="hi")],
        included_chunks=[],
        excluded_chunk_count=0,
        estimated_total_tokens=10,
        from_cache=False,
    )
    cache.set("k1", "proj1", response)
    fetched = cache.get("k1")
    assert fetched is not None
    assert fetched.query == "q"


def test_redis_query_cache_invalidate_project():
    """invalidate_project removes only the project's keys."""
    from semrag.api.schemas import QueryResponse, ChatMessage

    client = _StubRedis()
    cache = RedisQueryCache(client=client, ttl_seconds=10)
    response = QueryResponse(
        query="q",
        messages=[ChatMessage(role="user", content="hi")],
        included_chunks=[],
        excluded_chunk_count=0,
        estimated_total_tokens=10,
        from_cache=False,
    )
    cache.set("k1", "projA", response)
    cache.set("k2", "projB", response)
    cache.invalidate_project("projA")
    assert cache.get("k1") is None
    assert cache.get("k2") is not None


def test_redis_rate_limiter_window():
    """The limiter allows up to N requests per window."""
    client = _StubRedis()
    limiter: RateLimiter = RedisRateLimiter(
        client=client, requests_per_window=2, window_seconds=60.0
    )
    assert asyncio.run(limiter.check_and_record("k"))
    assert asyncio.run(limiter.check_and_record("k"))
    assert asyncio.run(limiter.check_and_record("k")) is False
    asyncio.run(limiter.reset())


def test_redis_embedding_cache_round_trip():
    """The Redis embedding cache behaves like the in-memory one over its
    public interface."""
    client = _StubRedis()
    cache = RedisEmbeddingCache(client=client, model_name="m1")
    cache.set("h1", [1.0, 2.0, 3.0])
    assert cache.get("h1") == [1.0, 2.0, 3.0]
    cache.set_many({"h2": [4.0], "h3": [5.0, 6.0]})
    out = cache.get_many(["h1", "h2", "h3", "h4"])
    assert out["h1"] == [1.0, 2.0, 3.0]
    assert out["h2"] == [4.0]
    assert out["h3"] == [5.0, 6.0]
    assert "h4" not in out


# ---------------------------------------------------------------------------
# F4: WorkerService claim/run loop
# ---------------------------------------------------------------------------


def test_worker_service_claims_queued_jobs():
    """The worker service must atomically transition QUEUED -> RUNNING
    and run the job."""

    class _StubStore(JobStore):
        def __init__(self):
            self.updated: List[Any] = []

        def create_job(self, job):
            pass

        def get_job(self, job_id):
            return None

        def update_job(self, job_id, **kwargs):
            self.updated.append((job_id, kwargs))
            return True

        def delete_job(self, job_id):
            return True

        def list_jobs(self, project_id=None):
            return [
                Job(job_id="j1", project_id="p", state=JobState.QUEUED, root_path="")
            ]

    store = _StubStore()
    from semrag.worker import WorkerService

    service = WorkerService(
        job_store=store,
        session_factory=lambda: None,
        settings=None,
        poll_interval_seconds=0.001,
    )
    job = service.claim_next_queued()
    assert job is not None
    assert job.job_id == "j1"
    # The store observed the transition to RUNNING.
    assert ("j1", {"state": JobState.RUNNING}) in store.updated


def test_worker_service_skips_already_claimed():
    """A second poll should not double-claim the same job."""

    class _StubStore(JobStore):
        def __init__(self):
            self._calls = 0

        def create_job(self, job):
            pass

        def get_job(self, job_id):
            return None

        def update_job(self, job_id, **kwargs):
            return True

        def delete_job(self, job_id):
            return True

        def list_jobs(self, project_id=None):
            self._calls += 1
            return [
                Job(job_id="j1", project_id="p", state=JobState.QUEUED, root_path="")
            ]

    from semrag.worker import WorkerService

    store = _StubStore()
    service = WorkerService(
        job_store=store,
        session_factory=lambda: None,
        settings=None,
        poll_interval_seconds=0.001,
    )
    first = service.claim_next_queued()
    second = service.claim_next_queued()
    assert first is not None
    # Second claim should return None because the in-flight registry
    # marks j1 as already taken, even though the store still says QUEUED.
    assert second is None


# ---------------------------------------------------------------------------
# F5: Transaction safety — implicit rollback on failure
# ---------------------------------------------------------------------------


def test_worker_session_rolls_back_on_failure():
    """The worker pipeline must call ``session.rollback()`` on any
    exception inside the indexing pipeline so partial state doesn't
    persist.
    """
    from semrag.jobs.worker import indexing_worker

    class _RecordingSession:
        def __init__(self):
            self.actions: List[str] = []
            self.rolled_back = False

        def close(self):
            self.actions.append("close")

        def rollback(self):
            self.rolled_back = True
            self.actions.append("rollback")

        def commit(self):
            self.actions.append("commit")

    record = _RecordingSession()

    def factory():
        return record

    # Patch the scanner to raise before any real work happens.
    from semrag.ingestion.scanner.orchestrator import Scanner

    original_scan = Scanner.scan

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated scanner failure")

    Scanner.scan = _boom  # type: ignore
    try:
        from semrag.jobs.manager import JobManager as _JM

        job_manager = _JM(InMemoryJobStore())
        job, _ = job_manager.create_job(project_id="p", root_path="/tmp")
        job_manager.update_job(job.job_id, state=JobState.RUNNING)

        try:
            indexing_worker(
                job_id=job.job_id,
                project_id="p",
                root_path="/tmp",
                job_manager=job_manager,
                settings=None,
                session_factory=factory,
                embedding_provider_factory=lambda s: None,
            )
        except Exception:
            pass

        assert record.rolled_back, f"session.rollback() was not called: {record.actions}"
        assert "close" in record.actions
    finally:
        Scanner.scan = original_scan  # type: ignore


# ---------------------------------------------------------------------------
# F6: Cache invalidation on /index
# ---------------------------------------------------------------------------


def test_index_endpoint_invalidates_cache_on_enqueue():
    """A 202 from /index must drop the project's cached queries."""
    import os
    from fastapi.testclient import TestClient

    from semrag.api.app import create_app
    from semrag.api.api_key import (
        APIKey,
        get_api_key_store,
        reset_api_key_store_for_tests,
    )
    from semrag.api.query_cache import TTLQueryCache
    from semrag.api.schemas import QueryResponse, ChatMessage
    from semrag.core.settings import reset_settings_cache

    # Force the in-memory store. Production default is now ``database``,
    # which would try to connect to Postgres and fail in this test.
    os.environ["SEMRAG_API_KEY_STORE"] = "memory"
    os.environ["SEMRAG_JOB_STORE_BACKEND"] = "memory"
    reset_settings_cache()
    reset_api_key_store_for_tests()
    from semrag.api.dependencies import reset_job_store_for_tests
    reset_job_store_for_tests()

    import semrag.api.query_cache as qc_module
    qc_module._shared_cache = TTLQueryCache(ttl_seconds=60)

    cache = qc_module.get_query_cache()
    response = QueryResponse(
        query="q",
        messages=[ChatMessage(role="user", content="hi")],
        included_chunks=[],
        excluded_chunk_count=0,
        estimated_total_tokens=10,
        from_cache=False,
    )
    cache.set("pre_existing_key", "test_project", response)
    assert cache.get("pre_existing_key") is not None

    import asyncio
    import datetime as dt
    from datetime import timezone as _tz
    store = get_api_key_store()
    api_key = APIKey(
        key="integration-key",
        name="integration",
        created_at=dt.datetime.now(timezone.utc).replace(tzinfo=None),
    )
    asyncio.run(store.create_key(api_key))

    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/v1/projects/test_project/index",
        headers={"Authorization": "Bearer integration-key", "Content-Type": "application/json"},
        json={"root_path": project_root},
    )
    assert r.status_code in (200, 202), f"got {r.status_code}: {r.text}"
    assert cache.get("pre_existing_key") is None
