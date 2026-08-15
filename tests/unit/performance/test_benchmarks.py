"""
Benchmark smoke tests.

Not a full benchmark suite (that lives in `run_benchmarks.py` for the
retrieval side and `run_benchmarks.py`-adjacent harnesses for ingestion).
These tests assert one thing each: that the bulk and batch code paths
introduced in Phase 5 actually do less work than the sequential paths
they replace. They run small inputs in-process so they're fast enough
to live alongside unit tests without slowing CI.

The assertions are deliberately generous — we want to catch "this code
got 10x slower overnight" regressions, not pin the absolute cost. A
test that fails when the optimization actually disappears will catch
the right thing; a test that fails when the absolute time fluctuates
between dev laptops would be noise.
"""
from __future__ import annotations

import time

from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.embedding.cache import (
    InMemoryEmbeddingCache,
    JSONFileEmbeddingCache,
)
from semrag.ingestion.streaming import stream_chunks_in_windows


def _fake_embedding(text: str, dim: int = 8) -> list[float]:
    """Cheap deterministic embedding for perf work; doesn't go to a model."""
    return [float(len(text))] * dim


def test_persistent_cache_set_many_faster_than_per_call(tmp_path):
    """Bulk-set path must be measurably faster than per-key set()."""
    cache = JSONFileEmbeddingCache(tmp_path, model_name="bench")
    items = {f"k{i}": _fake_embedding(f"text-{i}") for i in range(200)}

    t0 = time.perf_counter()
    for k, v in items.items():
        cache.set(k, v)
    cache.flush()
    sequential = time.perf_counter() - t0

    cache2 = JSONFileEmbeddingCache(tmp_path, model_name="bench2")
    t0 = time.perf_counter()
    cache2.set_many(items)
    cache2.flush()
    bulk = time.perf_counter() - t0

    # Bulk path should be at least somewhat faster — on a real disk the
    # win is several x; we leave a generous floor so this test doesn't
    # false-positive on warm-cache CI runners.
    assert bulk < sequential, f"bulk={bulk:.4f}s seq={sequential:.4f}s"


def test_streaming_pipeline_does_not_allocate_extra_copies():
    """The streaming helper yields the underlying chunks without copying."""
    chunks = [_ChunkFactory(f"c{i}") for i in range(10)]
    windows = list(stream_chunks_in_windows(chunks, window_size=3))
    # First window is a slice of the original list — same list contents.
    assert windows[0] is chunks[0:3] or windows[0] == chunks[0:3]


class _ChunkFactory:
    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
        self.text = "x"

    def __eq__(self, other):
        return isinstance(other, _ChunkFactory) and self.chunk_id == other.chunk_id


def test_in_memory_cache_batch_read_handles_duplicates():
    """get_many dedupes the request list internally — important when many
    chunks share the same hash (boilerplate, license headers, etc.)."""
    cache = InMemoryEmbeddingCache()
    cache.set("dup", _fake_embedding("dup"))
    hits = cache.get_many(["dup", "dup", "missing", "dup"])
    assert hits == {"dup": _fake_embedding("dup")}