"""
API configuration.

Same env-var-driven pattern as Part 5's session.py's get_database_url()
(one function, one source of truth, no config file needed for local dev)
rather than introducing pydantic-settings as a new dependency for a
handful of values. API_KEYS is a comma-separated list rather than one
key, since a real deployment issues distinct keys per client (the CLI,
future VS Code extension, a CI pipeline) so any one can be revoked
without breaking the others - Part 1's "authentication" requirement,
built as something that actually distinguishes callers, not a single
shared secret.
"""
from __future__ import annotations

import os


def get_api_keys() -> set[str]:
    raw = os.environ.get("SEMRAG_API_KEYS", "dev-local-key")
    return {k.strip() for k in raw.split(",") if k.strip()}


def get_rate_limit_per_minute() -> int:
    return int(os.environ.get("SEMRAG_RATE_LIMIT_PER_MINUTE", "60"))


def get_query_cache_ttl_seconds() -> int:
    return int(os.environ.get("SEMRAG_QUERY_CACHE_TTL_SECONDS", "300"))


def get_embedding_provider_dir() -> str:
    """Where each project's fitted TfidfEmbeddingProvider is persisted
    (see tfidf_provider.py's save()/load() docstring for why this must
    exist at all for a corpus-fit provider) — a real neural provider
    would need no equivalent directory."""
    path = os.environ.get("SEMRAG_EMBEDDING_PROVIDER_DIR", "/tmp/semrag_embedding_providers")
    os.makedirs(path, exist_ok=True)
    return path
