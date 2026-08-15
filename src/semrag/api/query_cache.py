"""
Query result cache.

TTL-based, keyed by a hash of (project_id, query text, top_k,
and the retrieval flags) - Part 1's "caching" requirement for the API
layer, scoped honestly the same way rate_limit.py's limiter is: correct
for one running process, not a shared cache across replicas (Redis would
be the right call there, Part 16's concern). A short default TTL (5
minutes) reflects that the underlying index changes whenever a project is
re-scanned - a long-lived cache would risk serving stale retrieval
results after a re-index without any invalidation signal wired to it.

A per-project key index is kept alongside the cache entries specifically
so invalidate_project() can be precise (clear only that project's cached
queries) rather than clearing the entire cache on every re-index, which
would needlessly discard other projects' still-valid cached results.

Wave 3: cache hit / miss / eviction decisions feed the
``semrag_cache_*`` Prometheus counters so the cache hit rate is
observable from a Grafana panel without scraping application logs.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal

from semrag.api.schemas import QueryResponse
from semrag.core import metrics as metrics_module
from semrag.core.settings import get_settings


def cache_key(
    project_id: str,
    query: str,
    top_k: int,
    enable_graph_expansion: bool,
    enable_reranking: bool,
    enable_intent_boost: bool,
) -> str:
    raw = (
        f"{project_id}|{query}|{top_k}|{enable_graph_expansion}"
        f"|{enable_reranking}|{enable_intent_boost}"
    )
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


class QueryCache(ABC):
    """Abstract base class for a query cache."""

    @abstractmethod
    def get(self, key: str) -> QueryResponse | None:
        """Get a cached query response by key, if present and not expired."""
        pass

    @abstractmethod
    def set(self, key: str, project_id: str, response: QueryResponse) -> None:
        """Cache a query response."""
        pass

    @abstractmethod
    def invalidate_project(self, project_id: str) -> None:
        """Remove all cached queries for the given project."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the entire cache."""
        pass


class TTLQueryCache(QueryCache):
    """In-memory TTL-based query cache."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, QueryResponse]] = {}
        self._keys_by_project: dict[str, set[str]] = defaultdict(set)

    def get(self, key: str) -> QueryResponse | None:
        entry = self._store.get(key)
        if entry is None:
            metrics_module.inc_cache_miss("memory")
            return None
        expires_at, response = entry
        if time.monotonic() > expires_at:
            self._evict(key)
            metrics_module.inc_cache_miss("memory")
            metrics_module.inc_cache_eviction("memory")
            return None
        metrics_module.inc_cache_hit("memory")
        return response

    def set(self, key: str, project_id: str, response: QueryResponse) -> None:
        self._store[key] = (time.monotonic() + self._ttl, response)
        self._keys_by_project[project_id].add(key)

    def invalidate_project(self, project_id: str) -> None:
        for key in self._keys_by_project.pop(project_id, set()):
            self._store.pop(key, None)
            metrics_module.inc_cache_eviction("memory")

    def clear(self) -> None:
        self._store.clear()
        self._keys_by_project.clear()

    def _evict(self, key: str) -> None:
        self._store.pop(key, None)
        for keys in self._keys_by_project.values():
            keys.discard(key)


# Redis key layout — shared between the two Redis-backed caches so that
# a project invalidation can do a single scan against the prefix:
#   semrag:query:cache:{key}            -> JSON-encoded QueryResponse
#   semrag:query:project_keys:{project} -> SET of cache keys per project
_QUERY_KEY_PREFIX = "semrag:query:cache:"
_PROJECT_KEYS_PREFIX = "semrag:query:project_keys:"


class RedisQueryCache(QueryCache):
    """Redis-backed query cache.

    TTL is enforced by Redis (`SETEX`); per-project invalidation is
    precise via a SET of cache keys per project. The values are the
    JSON-encoded ``QueryResponse`` — model_dump() round-trips cleanly
    through pydantic for retrieval, and the body is human-readable if
    an operator ever needs to inspect a key directly.
    """

    def __init__(self, client, ttl_seconds: float):
        self._client = client
        self._ttl = ttl_seconds

    def _key(self, key: str) -> str:
        return _QUERY_KEY_PREFIX + key

    def _project_set(self, project_id: str) -> str:
        return _PROJECT_KEYS_PREFIX + project_id

    def get(self, key: str) -> QueryResponse | None:
        raw = self._client.get(self._key(key))
        if raw is None:
            metrics_module.inc_cache_miss("redis")
            return None
        metrics_module.inc_cache_hit("redis")
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            # Corrupt value — drop it and treat as miss.
            self._client.delete(self._key(key))
            metrics_module.inc_cache_miss("redis")
            metrics_module.inc_cache_eviction("redis")
            return None
        return QueryResponse.model_validate(payload)

    def set(self, key: str, project_id: str, response: QueryResponse) -> None:
        payload = response.model_dump_json()
        pipe = self._client.pipeline()
        pipe.setex(self._key(key), int(self._ttl), payload)
        pipe.sadd(self._project_set(project_id), key)
        # Track the project-set member too so it gets cleaned up at the
        # same TTL as the cache entry; otherwise stale project indexes
        # accumulate forever.
        pipe.expire(self._project_set(project_id), int(self._ttl))
        pipe.execute()

    def invalidate_project(self, project_id: str) -> None:
        members = self._client.smembers(self._project_set(project_id))
        if not members:
            return
        keys = [self._key(m.decode() if isinstance(m, bytes) else m) for m in members]
        pipe = self._client.pipeline()
        pipe.delete(*keys)
        pipe.delete(self._project_set(project_id))
        pipe.execute()
        for _ in members:
            metrics_module.inc_cache_eviction("redis")

    def clear(self) -> None:
        # Wipe everything under our prefixes. SCAN keeps this safe to
        # run on a shared Redis instance where other apps' keys live.
        for prefix in (_QUERY_KEY_PREFIX, _PROJECT_KEYS_PREFIX):
            for key in self._client.scan_iter(match=prefix + "*", count=200):
                self._client.delete(key)


def get_shared_query_cache() -> QueryCache:
    """Return a query cache instance based on the current settings.

    Backend selection is exact: ``backend=redis`` requires a configured
    ``SEMRAG_REDIS_URL`` and will raise ``RuntimeError`` if the URL is
    empty rather than silently downgrading to the in-memory cache. An
    operator who wants best-effort behavior should pick ``backend=memory``;
    selecting Redis means "I require the distributed property, do not
    fall back." See ``require_redis_client`` in ``redis_client.py``.
    """
    settings = get_settings()
    if settings.query_cache_backend == "redis":
        from semrag.api.redis_client import require_redis_client

        client = require_redis_client()
        return RedisQueryCache(client=client, ttl_seconds=settings.query_cache_ttl_seconds)
    return TTLQueryCache(ttl_seconds=settings.query_cache_ttl_seconds)


# Module-level singleton used by the FastAPI dependency. ``get_query_cache``
# below returns this; tests override via ``app.dependency_overrides``.
_shared_cache: QueryCache | None = None


def get_query_cache() -> QueryCache:
    """FastAPI dependency for the shared query cache.

    A module-level singleton is created lazily so tests that override
    settings between cases get a fresh instance after the cache is
    cleared. Production runs through this path once per process; the
    lazy initialization is a one-time cost.
    """
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = get_shared_query_cache()
    return _shared_cache


def reset_shared_cache() -> None:
    """Drop the module-level singleton. Used by tests after settings
    mutation so the next ``get_query_cache`` builds a fresh instance."""
    global _shared_cache
    _shared_cache = None