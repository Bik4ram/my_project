import asyncio
import time

from semrag.api.rate_limit import MemoryRateLimiter


def test_allows_requests_under_limit():
    limiter = MemoryRateLimiter(requests_per_window=3, window_seconds=60)
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is True


def test_rejects_requests_over_limit():
    limiter = MemoryRateLimiter(requests_per_window=2, window_seconds=60)
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is False


def test_rejected_request_does_not_count_against_window():
    limiter = MemoryRateLimiter(requests_per_window=1, window_seconds=60)
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is False
    assert asyncio.run(limiter.check_and_record("key1")) is False


def test_different_keys_have_independent_limits():
    limiter = MemoryRateLimiter(requests_per_window=1, window_seconds=60)
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key2")) is True


def test_old_requests_fall_out_of_window():
    limiter = MemoryRateLimiter(requests_per_window=1, window_seconds=0.05)
    assert asyncio.run(limiter.check_and_record("key1")) is True
    assert asyncio.run(limiter.check_and_record("key1")) is False
    time.sleep(0.1)
    assert asyncio.run(limiter.check_and_record("key1")) is True