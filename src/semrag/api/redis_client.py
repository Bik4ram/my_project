"""
Redis client factory.

Single chokepoint for turning ``Settings.redis_url`` into either a sync
or async ``redis`` connection. Imported lazily by the Redis query cache,
rate limiter, and embedding cache so a deployment that doesn't use
Redis never pays the import cost.

Two flavors of client are exposed:

- ``get_redis_client`` / ``require_redis_client`` — sync ``redis.Redis``.
  Used by the query cache (``QueryCache`` ABC is sync) and the embedding
  cache (``EmbeddingCache`` Protocol is sync). Both run in synchronous
  FastAPI request handlers and from synchronous pipeline code.
- ``get_async_redis_client`` / ``require_async_redis_client`` — async
  ``redis.asyncio.Redis``. Used by the rate limiter so its declared
  ``async def check_and_record`` is actually non-blocking; the previous
  implementation wrapped sync ``redis.Redis`` and would freeze the
  FastAPI worker thread under load.

``require_*`` variants raise ``RuntimeError`` if ``redis_url`` is empty
— this is the no-silent-fallback rule for backends that the operator
*configured* to be Redis. ``get_*`` variants return ``None`` so the
caller can choose; only the builders wired to a specific backend
(``query_cache_backend=redis`` etc.) use the ``require`` form.
"""
from __future__ import annotations

from typing import Any

from semrag.core.settings import get_settings


def _import_redis() -> Any:
    """Lazy import of redis-py. ``redis`` is a declared dependency for
    production deployments that select a Redis backend, but the import
    is deferred so a deployment that doesn't use Redis never pays the
    module-load cost."""
    try:
        import redis  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Redis backend requested but the `redis` package is not "
            "installed. Install it with `pip install redis` to enable "
            "Redis-backed caches, rate limiters, and embedding caches."
        ) from exc
    return redis


def _import_async_redis() -> Any:
    """Lazy import of redis.asyncio. Same rationale as ``_import_redis``
    — the package is declared in pyproject.toml, but the import is
    deferred to keep the import graph small for deployments that don't
    touch the rate limiter's Redis path."""
    try:
        import redis.asyncio  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Redis async backend requested but the `redis` package is not "
            "installed. Install it with `pip install redis` (>=4.2) to "
            "enable the Redis-backed rate limiter."
        ) from exc
    return redis.asyncio


def _require_url() -> str:
    """Shared helper: read ``SEMRAG_REDIS_URL`` and raise if empty.

    Returning ``None`` here would let callers silently pick a memory
    fallback, defeating the operator's explicit choice of a Redis
    backend. The setting name is included verbatim in the error so a
    misconfigured deployment is one click away from the fix in any
    operator's runbook.
    """
    settings = get_settings()
    url = settings.redis_url
    if not url:
        raise RuntimeError(
            "Redis backend requested but SEMRAG_REDIS_URL is empty. "
            "Set SEMRAG_REDIS_URL=redis://host:port/db or switch the "
            "backend to a non-Redis value (memory, file, database)."
        )
    return url


def get_redis_client() -> Any:
    """Build a sync Redis client from the configured URL.

    Returns ``None`` if no URL is configured. Use the ``require_`` form
    in backend builders that should fail fast when Redis is selected.
    """
    settings = get_settings()
    url = settings.redis_url
    if not url:
        return None
    redis = _import_redis()
    return redis.Redis.from_url(url, decode_responses=False)


def get_async_redis_client() -> Any:
    """Build an async ``redis.asyncio.Redis`` client from the configured
    URL. Returns ``None`` if no URL is configured; callers that need
    a hard failure should use ``require_async_redis_client`` instead.
    """
    settings = get_settings()
    url = settings.redis_url
    if not url:
        return None
    redis_asyncio = _import_async_redis()
    return redis_asyncio.Redis.from_url(url, decode_responses=False)


def require_redis_client() -> Any:
    """Build a sync Redis client or raise.

    The builder pattern for ``backend=redis`` on the query cache and
    embedding cache uses this so a misconfigured deployment fails at
    first use rather than silently downgrading to in-memory.
    """
    url = _require_url()
    redis = _import_redis()
    return redis.Redis.from_url(url, decode_responses=False)


def require_async_redis_client() -> Any:
    """Build an async Redis client or raise. Used by the rate-limiter
    builder for the same fail-fast reason as ``require_redis_client``.
    """
    url = _require_url()
    redis_asyncio = _import_async_redis()
    return redis_asyncio.Redis.from_url(url, decode_responses=False)
