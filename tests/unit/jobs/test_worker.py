"""Regression tests for the indexing worker module."""
from __future__ import annotations

from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.jobs import worker as worker_module


def test_worker_resolves_in_memory_embedding_cache():
    """The worker instantiates ``InMemoryEmbeddingCache`` at runtime; it
    must be importable in the worker module namespace (was a NameError
    when the import was missing)."""
    assert worker_module.InMemoryEmbeddingCache is InMemoryEmbeddingCache


def test_worker_exposes_indexing_worker_entrypoint():
    assert callable(worker_module.indexing_worker)