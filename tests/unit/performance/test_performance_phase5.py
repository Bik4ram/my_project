"""
Phase 5 — Performance & Scalability tests.

These cover the bulk/batch additions to the storage layer, the
persistent embedding cache, the streaming ingestion helper, and the
pool-tuning knobs exposed via Settings. Pure-unit tests; no DB or
network required.
"""
from __future__ import annotations

import os

import pytest

from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.embedding.cache import (
    InMemoryEmbeddingCache,
    JSONFileEmbeddingCache,
)
from semrag.ingestion.streaming import stream_chunks_in_windows


@pytest.fixture
def clean_env(monkeypatch):
    """Strip SEMRAG_ env vars so tests start from a known baseline."""
    from semrag.core.settings import reset_settings_cache

    for key in list(os.environ):
        if key.startswith("SEMRAG_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield monkeypatch
    reset_settings_cache()


# ---------------------------------------------------------------------------
# Persistent embedding cache: batch writes + flush-on-demand
# ---------------------------------------------------------------------------

def test_json_file_cache_set_many_defers_disk_writes(tmp_path: Path):
    cache = JSONFileEmbeddingCache(tmp_path, model_name="m")
    # set_many should not raise and should leave the cache dirty until
    # flush() is called explicitly.
    cache.set_many({"h1": [0.1, 0.2], "h2": [0.3, 0.4]})
    # The cache should be readable from memory immediately.
    assert cache.get("h1") == [0.1, 0.2]
    assert cache.get("h2") == [0.3, 0.4]
    # A fresh instance loaded from disk should NOT yet see these writes.
    fresh = JSONFileEmbeddingCache(tmp_path, model_name="m")
    assert fresh.get("h1") is None
    # After flush(), the writes become visible on disk.
    cache.flush()
    fresh = JSONFileEmbeddingCache(tmp_path, model_name="m")
    assert fresh.get("h1") == [0.1, 0.2]
    assert fresh.get("h2") == [0.3, 0.4]


def test_json_file_cache_get_many_returns_only_hits(tmp_path: Path):
    cache = JSONFileEmbeddingCache(tmp_path, model_name="m")
    cache.set("exists", [1.0, 2.0])
    hits = cache.get_many(["exists", "missing", "exists"])
    assert hits == {"exists": [1.0, 2.0]}


def test_in_memory_cache_get_many_returns_only_hits():
    cache = InMemoryEmbeddingCache()
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    assert cache.get_many(["a", "missing", "b"]) == {"a": [1.0], "b": [2.0]}


def test_in_memory_cache_set_many_is_atomic_for_caller():
    cache = InMemoryEmbeddingCache()
    cache.set_many({"x": [0.1], "y": [0.2]})
    assert cache.get("x") == [0.1]
    assert cache.get("y") == [0.2]


def test_in_memory_cache_flush_is_noop():
    cache = InMemoryEmbeddingCache()
    cache.set("k", [1.0])
    cache.flush()  # must not raise
    assert cache.get("k") == [1.0]


# ---------------------------------------------------------------------------
# Streaming ingestion helper
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        project_id="p1",
        relative_path="RTL/a.sv",
        chunk_type="sv_module",
        text=text,
    )


def test_streaming_returns_single_window_for_small_inputs():
    chunks = [_chunk(f"c{i}") for i in range(3)]
    windows = list(stream_chunks_in_windows(chunks, window_size=10))
    assert len(windows) == 1
    assert len(windows[0]) == 3


def test_streaming_splits_into_windows():
    chunks = [_chunk(f"c{i}") for i in range(7)]
    windows = list(stream_chunks_in_windows(chunks, window_size=3))
    assert [len(w) for w in windows] == [3, 3, 1]


def test_streaming_with_zero_window_size_returns_full_list():
    chunks = [_chunk(f"c{i}") for i in range(4)]
    windows = list(stream_chunks_in_windows(chunks, window_size=0))
    assert len(windows) == 1
    assert len(windows[0]) == 4


# ---------------------------------------------------------------------------
# Settings: pool tuning knobs are exposed and type-safe
# ---------------------------------------------------------------------------

def test_pool_settings_have_sane_defaults(clean_env):
    from semrag.core.settings import Settings
    s = Settings()
    assert s.db_pool_size > 0
    assert s.db_max_overflow >= 0
    assert s.db_pool_recycle > 0
    assert s.db_pool_timeout > 0
    assert s.max_concurrent_parse_workers >= 1


def test_pool_settings_are_tunable(clean_env):
    from semrag.core.settings import Settings
    s = Settings(db_pool_size=20, db_max_overflow=40, db_pool_recycle=600, db_pool_timeout=10, max_concurrent_parse_workers=8)
    assert s.db_pool_size == 20
    assert s.db_max_overflow == 40
    assert s.db_pool_recycle == 600
    assert s.db_pool_timeout == 10
    assert s.max_concurrent_parse_workers == 8
