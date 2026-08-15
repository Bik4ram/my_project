"""Embedding provider factory — single entry point for CLI and API."""
from __future__ import annotations

from semrag.core.observability import get_logger
from semrag.core.settings import Settings
from semrag.ingestion.embedding.interfaces import EmbeddingProvider

logger = get_logger(__name__)


class EmbeddingConfigurationError(RuntimeError):
    pass


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    if settings is None:
        from semrag.core.settings import get_settings
        settings = get_settings()

    name = settings.embedding_provider
    common = {
        "model": settings.embedding_model,
        "dimension": settings.embedding_dimension,
        "timeout_seconds": settings.embedding_timeout_seconds,
        "api_key": settings.embedding_api_key,
    }

    if name == "ollama":
        from semrag.ingestion.embedding.ollama_provider import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider(base_url=settings.embedding_base_url, **common)
    if name == "openai_compatible":
        from semrag.ingestion.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider
        return OpenAICompatibleEmbeddingProvider(base_url=settings.embedding_base_url, **common)
    if name == "tfidf":
        from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
        return TfidfEmbeddingProvider(dimension=settings.embedding_dimension)
    raise EmbeddingConfigurationError(f"Unknown embedding provider: {name!r}")


def build_embedding_cache(settings: Settings | None = None):
    """Return an embedding cache based on the configured backend.

    Backend selection is exact: ``backend=redis`` requires a configured
    ``SEMRAG_REDIS_URL`` and will raise ``RuntimeError`` if the URL is
    empty rather than silently downgrading to the JSONFile cache. An
    operator who wants best-effort behavior should pick
    ``backend=jsonfile``; selecting Redis means "I require the
    distributed property, do not fall back." See ``require_redis_client``
    in ``redis_client.py``.
    """
    if settings is None:
        from semrag.core.settings import get_settings
        settings = get_settings()

    if settings.embedding_cache_backend == "memory":
        from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
        return InMemoryEmbeddingCache()
    elif settings.embedding_cache_backend == "jsonfile":
        from pathlib import Path
        from semrag.ingestion.embedding.cache import JSONFileEmbeddingCache
        return JSONFileEmbeddingCache(Path(settings.embedding_cache_dir), model_name=settings.embedding_model)
    elif settings.embedding_cache_backend == "redis":
        from semrag.api.redis_client import require_redis_client
        from semrag.ingestion.embedding.redis_cache import RedisEmbeddingCache

        client = require_redis_client()
        return RedisEmbeddingCache(client=client, model_name=settings.embedding_model)
    else:
        raise ValueError(f"Unknown embedding cache backend: {settings.embedding_cache_backend!r}")