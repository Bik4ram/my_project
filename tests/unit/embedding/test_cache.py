from pathlib import Path

from semrag.ingestion.embedding.cache import (
    InMemoryEmbeddingCache,
    JSONFileEmbeddingCache,
    text_hash,
)


def test_text_hash_deterministic():
    assert text_hash("hello world") == text_hash("hello world")
    assert text_hash("hello world") != text_hash("hello there")


def test_in_memory_cache_roundtrip():
    cache = InMemoryEmbeddingCache()
    assert cache.get("abc") is None
    cache.set("abc", [0.1, 0.2, 0.3])
    assert cache.get("abc") == [0.1, 0.2, 0.3]


def test_json_file_cache_roundtrip(tmp_path: Path):
    cache = JSONFileEmbeddingCache(tmp_path, model_name="tfidf-lsa-16d")
    cache.set("hash1", [0.1, 0.2])
    assert cache.get("hash1") == [0.1, 0.2]


def test_json_file_cache_persists_across_instances(tmp_path: Path):
    cache1 = JSONFileEmbeddingCache(tmp_path, model_name="tfidf-lsa-16d")
    cache1.set("hash1", [0.5, 0.6])

    cache2 = JSONFileEmbeddingCache(tmp_path, model_name="tfidf-lsa-16d")
    assert cache2.get("hash1") == [0.5, 0.6]


def test_json_file_cache_namespaced_by_model_name(tmp_path: Path):
    cache_a = JSONFileEmbeddingCache(tmp_path, model_name="model-a")
    cache_a.set("hash1", [1.0])

    cache_b = JSONFileEmbeddingCache(tmp_path, model_name="model-b")
    assert cache_b.get("hash1") is None  # different model's cache is separate
