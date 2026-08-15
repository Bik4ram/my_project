import time

from semrag.api.query_cache import TTLQueryCache, cache_key
from semrag.api.schemas import QueryResponse


def _response(query: str = "q") -> QueryResponse:
    return QueryResponse(
        query=query, messages=[], included_chunks=[], excluded_chunk_count=0, estimated_total_tokens=10,
    )


def test_cache_key_is_deterministic():
    k1 = cache_key("proj1", "query text", 10, True, True, True)
    k2 = cache_key("proj1", "query text", 10, True, True, True)
    assert k1 == k2


def test_cache_key_differs_by_query():
    k1 = cache_key("proj1", "query A", 10, True, True, True)
    k2 = cache_key("proj1", "query B", 10, True, True, True)
    assert k1 != k2


def test_get_set_roundtrip():
    cache = TTLQueryCache(ttl_seconds=60)
    key = cache_key("proj1", "q", 10, True, True, True)
    assert cache.get(key) is None
    cache.set(key, "proj1", _response())
    assert cache.get(key) is not None


def test_entry_expires_after_ttl():
    cache = TTLQueryCache(ttl_seconds=0.05)
    key = cache_key("proj1", "q", 10, True, True, True)
    cache.set(key, "proj1", _response())
    assert cache.get(key) is not None
    time.sleep(0.1)
    assert cache.get(key) is None


def test_invalidate_project_clears_only_that_projects_entries():
    cache = TTLQueryCache(ttl_seconds=60)
    key_a = cache_key("proj_a", "q", 10, True, True, True)
    key_b = cache_key("proj_b", "q", 10, True, True, True)
    cache.set(key_a, "proj_a", _response())
    cache.set(key_b, "proj_b", _response())

    cache.invalidate_project("proj_a")

    assert cache.get(key_a) is None
    assert cache.get(key_b) is not None
