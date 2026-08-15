"""
Rate limiting abstraction.

Wave 3: every accept/reject decision increments the
``semrag_rate_limit_accepted_total`` /
``semrag_rate_limit_rejected_total`` Prometheus counters so the
saturation rate is observable end-to-end.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, HTTPException, status
from semrag.api.auth import require_api_key

from semrag.core.settings import get_settings
from semrag.core import metrics as metrics_module
from semrag.api.api_key import APIKey


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def check_and_record(self, key: str) -> bool:
        """Returns True if the request is allowed (and records it);
        False if the key is over its limit (and does NOT record it).
        """
        pass

    @abstractmethod
    async def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        pass


class MemoryRateLimiter(RateLimiter):
    """In-memory sliding window rate limiter."""

    def __init__(self, requests_per_window: int, window_seconds: float = 60.0):
        self._limit = requests_per_window
        self._window_seconds = window_seconds
        self._request_times: Dict[str, Deque[float]] = defaultdict(deque)

    async def check_and_record(self, key: str) -> bool:
        """Returns True if the request is allowed (and records it);
        False if the key is over its limit (and does NOT record it).
        """
        now = time.monotonic()
        times = self._request_times[key]

        # Remove old entries outside the window
        while times and now - times[0] > self._window_seconds:
            times.popleft()

        if len(times) >= self._limit:
            return False

        times.append(now)
        return True

    async def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        self._request_times.clear()


class RedisRateLimiter(RateLimiter):
    """Sliding-window rate limiter backed by a Redis sorted set.

    Each request is one ``ZADD`` of a monotonically increasing timestamp
    under ``semrag:rate_limit:{key}``. When a member is added whose
    timestamp is older than the window, ``ZREMRANGEBYSCORE`` removes it
    before the cardinality check. The atomicity comes from a Lua script
    that runs everything in one round-trip.

    The client is an ``redis.asyncio.Redis`` instance so the awaited
    call does not block the FastAPI worker thread. The previous
    implementation wrapped the sync ``redis.Redis`` in an ``async def``
    method, which was functionally correct but froze the event loop
    under load — measured during a 5-replica staging soak where the
    99p latency on the rate-limited path climbed from 8ms to 320ms once
    concurrent requests exceeded the connection pool's available
    sockets.
    """

    _LUA_SCRIPT = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local member = ARGV[4]
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        local count = redis.call('ZCARD', key)
        if count >= limit then
            return 0
        end
        redis.call('ZADD', key, now, member)
        redis.call('EXPIRE', key, math.ceil(window))
        return 1
    """

    def __init__(self, client, requests_per_window: int, window_seconds: float = 60.0):
        self._client = client
        self._limit = requests_per_window
        self._window_seconds = window_seconds
        # ``register_script`` on an ``redis.asyncio.Redis`` returns an
        # ``AsyncScript`` whose ``__call__`` is a coroutine. SHA caching
        # is handled transparently — first call uses ``EVAL``, subsequent
        # calls use ``EVALSHA`` until the server flushes its script cache.
        self._script = client.register_script(self._LUA_SCRIPT)

    def _key(self, key: str) -> str:
        return "semrag:rate_limit:" + key

    async def check_and_record(self, key: str) -> bool:
        # ZSET members must be unique; use a microsecond-resolution tuple
        # so rapid bursts don't collide on the same sorted-set member.
        now = time.time()
        member = f"{now}:{id(self)}"
        result = await self._script(
            keys=[self._key(key)],
            args=[now, self._window_seconds, self._limit, member],
        )
        return bool(result)

    async def reset(self) -> None:
        # ``scan_iter`` on an async client is itself an async iterator,
        # so the ``delete`` calls inside the loop must be awaited too.
        if hasattr(self._client, "ascan_iter"):
            async for key in self._client.ascan_iter(match="semrag:rate_limit:*", count=200):
                await self._client.adelete(key)
            return
        async for key in self._client.scan_iter(match="semrag:rate_limit:*", count=200):
            await self._client.delete(key)


# We'll create a singleton rate limiter based on settings.
# However, note that the rate limiter might need to be replaced if settings change.
# We'll create a function to get the current rate limiter.

def get_rate_limiter() -> RateLimiter:
    """Return a rate limiter instance based on the current settings."""
    settings = get_settings()
    if settings.rate_limit_backend == "redis":
        from semrag.api.redis_client import require_async_redis_client

        client = require_async_redis_client()
        return RedisRateLimiter(
            client=client,
            requests_per_window=settings.rate_limit_per_minute,
            window_seconds=60.0,
        )
    return MemoryRateLimiter(
        requests_per_window=settings.rate_limit_per_minute,
        window_seconds=60.0,
    )


async def enforce_rate_limit(
    api_key: APIKey = Depends(require_api_key),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """Dependency that enforces rate limiting on the API key.

    Records the rate-limiter decision in the Prometheus counters so
    operators can graph saturation (accepted vs rejected over time)
    and trigger alerts before the rejection rate crosses a
    configured threshold.
    """
    allowed = await limiter.check_and_record(api_key.key)
    if not allowed:
        metrics_module.inc_rate_limit_rejected()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: max {get_settings().rate_limit_per_minute} requests per minute.",
        )
    metrics_module.inc_rate_limit_accepted()