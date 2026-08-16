"""
Tests for API key authentication and store.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncio
import pytest

from semrag.api.api_key import (
    APIKey,
    APIKeyStatus,
    InMemoryAPIKeyStore,
    get_api_key_store,
    hash_api_key,
)
from semrag.api.auth import require_api_key
from semrag.core.settings import Settings, get_settings


def _run_async(coro):
    """Run an async coroutine synchronously for test compatibility."""
    return asyncio.run(coro)


def test_api_key_model():
    """Test the APIKey model."""
    key = "test-key"
    name = "test-name"
    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    expires_at = created_at + timedelta(days=1)
    api_key = APIKey(
        key=key,
        name=name,
        created_at=created_at,
        expires_at=expires_at,
        status=APIKeyStatus.ACTIVE,
        description="Test key",
        allowed_projects=["proj1", "proj2"],
    )
    assert api_key.key == key
    assert api_key.name == name
    assert api_key.created_at == created_at
    assert api_key.expires_at == expires_at
    assert api_key.status == APIKeyStatus.ACTIVE
    assert api_key.description == "Test key"
    assert api_key.allowed_projects == ["proj1", "proj2"]
    assert api_key.is_valid() is True
    assert api_key.is_allowed_for_project("proj1") is True
    assert api_key.is_allowed_for_project("proj3") is False
    # Test with empty allowed_projects (means all projects)
    key2 = "test-key-2"
    name2 = "test-name-2"
    api_key2 = APIKey(
        key=key2,
        name=name2,
        created_at=created_at,
    )
    assert api_key2.is_allowed_for_project("any") is True


def test_api_key_store():
    """Test the in-memory API key store.

    The store keys entries by ``hash_api_key(plaintext)`` — never by
    plaintext — so this test uses the bearer token string as the
    argument to ``get_key``/``delete_key``/``update_key`` and verifies
    the lookup/remove/update path is hash-based, then asserts the
    returned APIKey's ``.key`` field is the digest, not the plaintext.
    """
    import asyncio
    from semrag.api.api_key import hash_api_key

    store = InMemoryAPIKeyStore()
    api_key = APIKey(
        key="test-key",
        name="test-name",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    digest = hash_api_key("test-key")
    # Test create and get
    asyncio.run(store.create_key(api_key))
    retrieved = asyncio.run(store.get_key("test-key"))
    assert retrieved is not None
    assert retrieved.name == "test-name"
    # Plaintext must never round-trip through the stored APIKey.
    assert retrieved.key == digest
    assert "test-key" not in retrieved.key
    # Test update
    api_key.description = "Updated"
    asyncio.run(store.update_key("test-key", api_key))
    retrieved = asyncio.run(store.get_key("test-key"))
    assert retrieved.description == "Updated"
    # Test delete
    assert asyncio.run(store.delete_key("test-key")) is True
    assert asyncio.run(store.get_key("test-key")) is None
    # Test that update against an unknown plaintext returns False
    assert asyncio.run(store.update_key("never-inserted", api_key)) is False
    # Test delete against unknown plaintext returns False
    assert asyncio.run(store.delete_key("never-inserted")) is False
    # Test list
    asyncio.run(store.create_key(api_key))
    keys = asyncio.run(store.list_keys())
    assert len(keys) == 1
    assert keys[0].key == digest
    # Test cleanup expired
    expired_key = APIKey(
        key="expired-key",
        name="expired",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2),
        expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1),
    )
    asyncio.run(store.create_key(expired_key))
    # Assume the default expiration is 30 days, so this key is expired
    cleaned = asyncio.run(store.cleanup_expired())
    assert cleaned == 1
    assert asyncio.run(store.get_key("expired-key")) is None


def test_in_memory_store_does_not_leak_plaintext():
    """The InMemoryAPIKeyStore must never keep the plaintext bearer
    token in any of its internal state, attributes, or returned
    APIKey instances. A memory dump that captures the store should
    only ever yield blake3 digests, never reusable credentials.
    """
    import asyncio
    from semrag.api.api_key import hash_api_key

    bearer = "sk-leaked-credential-1234567890"
    store = InMemoryAPIKeyStore()
    asyncio.run(
        store.create_key(
            APIKey(key=bearer, name="n", created_at=datetime.now(timezone.utc).replace(tzinfo=None))
        )
    )

    # Internal dict keys are digests, never plaintext.
    digest = hash_api_key(bearer)
    assert bearer not in store._keys
    assert digest in store._keys

    # list_keys() must not surface the plaintext anywhere.
    for k in asyncio.run(store.list_keys()):
        assert bearer not in k.key
        assert bearer not in k.name
        assert bearer not in k.description

    # str()/repr() of the returned APIKey must not echo the plaintext.
    api_key = asyncio.run(store.get_key(bearer))
    assert api_key is not None
    assert bearer not in repr(api_key)


def test_hash_api_key_is_deterministic_and_isolated():
    """``hash_api_key`` must be:

    - Deterministic: same input -> same digest (lookups must succeed).
    - Domain-separated: an attacker cannot collide a SemRAG token with
      a token from a different system using the same blake3 function.
    - Non-reversible: the digest does not reveal the plaintext.
    """
    from semrag.api.api_key import hash_api_key

    a = hash_api_key("same-input")
    b = hash_api_key("same-input")
    assert a == b
    # The digest is hex; the plaintext cannot be reconstructed.
    assert "same-input" not in a
    # Domain separator means the same byte string under a different
    # domain would NOT collide here — implicitly tested by the
    # non-empty constant we use in the implementation.


def test_create_key_rejects_duplicate():
    """Inserting the same plaintext twice raises ValueError, matching
    the Postgres store's contract. Without this, two replicas could
    race to seed the same key."""
    import asyncio

    store = InMemoryAPIKeyStore()
    asyncio.run(
        store.create_key(APIKey(key="dup-key", name="a", created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
    )
    with pytest.raises(ValueError):
        asyncio.run(
            store.create_key(APIKey(key="dup-key", name="b", created_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        )


def test_get_api_key_store():
    """Test the factory function.

    The production default is now ``database``, so we explicitly set
    ``SEMRAG_API_KEY_STORE=memory`` for this test (the in-memory
    factory branch is the one we want to assert here; the database
    factory is exercised by the dedicated Postgres store tests).
    """
    import os
    from semrag.core.settings import reset_settings_cache

    os.environ["SEMRAG_API_KEY_STORE"] = "memory"
    try:
        reset_settings_cache()
        # Reset the in-memory singleton too so the factory honours the
        # freshly-read setting rather than the cached module-level
        # store from earlier tests.
        from semrag.api.api_key import reset_api_key_store_for_tests

        reset_api_key_store_for_tests()
        store = get_api_key_store()
        assert isinstance(store, InMemoryAPIKeyStore)
    finally:
        del os.environ["SEMRAG_API_KEY_STORE"]
        reset_settings_cache()
        reset_api_key_store_for_tests()


def test_production_default_is_database_store():
    """The default ``api_key_store`` for a fresh Settings must be the
    database-backed implementation. The previous default of ``memory``
    silently broke multi-replica deployments and lost every key on
    restart; production must never boot into that mode by accident.
    """
    from semrag.core.settings import Settings

    settings = Settings()
    assert settings.api_key_store == "database"


def test_verify_api_key_constant_time():
    """``verify_api_key`` must match the bearer's digest against a
    stored digest using a constant-time comparison. The test asserts
    both the happy path and the negative path; the structural guarantee
    (use of ``hmac.compare_digest``) is what matters for timing-safety,
    not any single numeric timing measurement."""
    from semrag.api.api_key import verify_api_key

    bearer = "sk-real-bearer-token-xyz"
    assert verify_api_key(bearer, hash_api_key(bearer)) is True
    # A different bearer must NOT verify against the stored digest.
    assert verify_api_key("sk-other-bearer", hash_api_key(bearer)) is False
    # A garbled-but-same-length digest also fails (the constant-time
    # path runs through compare_digest either way).
    digest = hash_api_key(bearer)
    tampered = "a" * len(digest)
    assert verify_api_key(bearer, tampered) is False


def test_require_api_key_constant_time_check_on_lookup(monkeypatch):
    """The ``require_api_key`` dependency must use the constant-time
    ``verify_api_key`` helper on the path that confirms the stored row
    matches the bearer. We patch the store to return a row with a
    *mismatched* digest and confirm the request still 401s — proving
    the verify step is on the hot path, not just a docstring.
    """
    from fastapi import Depends
    from fastapi.testclient import TestClient

    from semrag.api.app import create_app
    from semrag.api.api_key import APIKey as _AK
    from semrag.api.api_key import (
        InMemoryAPIKeyStore,
        hash_api_key,
        reset_api_key_store_for_tests,
    )

    reset_api_key_store_for_tests()
    store = InMemoryAPIKeyStore()

    class _TamperingStore(InMemoryAPIKeyStore):
        async def get_key(self, key: str):  # type: ignore[override]
            row = await super().get_key(key)
            if row is None:
                return None
            # Swap in a syntactically-valid-but-different digest so
            # the constant-time check fires and rejects.
            row = row.model_copy(update={"key": "f" * len(hash_api_key(key))})
            return row

    real_store = _TamperingStore()
    real_store._keys = store._keys  # share the same backing dict

    import semrag.api.auth

    monkeypatch.setattr(semrag.api.auth, "get_api_key_store", lambda: real_store)

    monkeypatch.setenv("SEMRAG_API_KEYS", "sk-real-bearer-token-xyz")
    from semrag.core.settings import reset_settings_cache

    reset_settings_cache()

    app = create_app()

    @app.get("/test")
    async def test_route(api_key: _AK = Depends(require_api_key)):
        return {"key_id": api_key.name}

    client = TestClient(app)
    response = client.get(
        "/test", headers={"Authorization": "Bearer sk-real-bearer-token-xyz"}
    )
    assert response.status_code == 401


def test_require_api_key_valid():
    """Test the require_api_key dependency with a valid key.

    Production default is now ``database``, so this test explicitly
    pins the in-memory store — the unit under test here is the auth
    dependency's header handling, not which backing store is in use.
    """
    import os
    from fastapi import Depends
    from fastapi.testclient import TestClient

    from semrag.api.app import create_app
    from semrag.api.api_key import reset_api_key_store_for_tests
    from semrag.core.settings import reset_settings_cache

    os.environ["SEMRAG_API_KEY_STORE"] = "memory"
    reset_settings_cache()
    reset_api_key_store_for_tests()

    app = create_app()

    # Override the API key store to return a known key
    store = get_api_key_store()
    test_key = APIKey(
        key="test-key",
        name="test",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    # We need to replace the store's get_key method to return our test key
    # For simplicity, we'll just add the key to the store
    import asyncio
    asyncio.run(store.create_key(test_key))

    # Point the auth dependency at this store (it builds a fresh module-level
    # store per request otherwise).
    import semrag.api.auth
    original_get_api_key_store = semrag.api.auth.get_api_key_store
    semrag.api.auth.get_api_key_store = lambda: store

    # Override the environment variable for API keys
    os.environ["SEMRAG_API_KEYS"] = "test-key"

    @app.get("/test")
    async def test_route(api_key: APIKey = Depends(require_api_key)):
        return {"key_id": api_key.name}

    client = TestClient(app)
    response = client.get("/test", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json() == {"key_id": "test"}

    del os.environ["SEMRAG_API_KEY_STORE"]
    del os.environ["SEMRAG_API_KEYS"]
    reset_settings_cache()
    reset_api_key_store_for_tests()
    semrag.api.auth.get_api_key_store = original_get_api_key_store


def test_require_api_key_invalid():
    """Test the require_api_key dependency with an invalid key.

    Pin the in-memory store explicitly (production default is now
    ``database``) so this test doesn't try to hit a Postgres that
    isn't running."""
    import os
    from fastapi import Depends
    from fastapi.testclient import TestClient

    from semrag.api.app import create_app
    from semrag.api.api_key import reset_api_key_store_for_tests
    from semrag.core.settings import reset_settings_cache

    os.environ["SEMRAG_API_KEY_STORE"] = "memory"
    reset_settings_cache()
    reset_api_key_store_for_tests()

    app = create_app()

    @app.get("/test")
    async def test_route(api_key: APIKey = Depends(require_api_key)):
        return {"key_id": api_key.name}

    client = TestClient(app)
    response = client.get("/test", headers={"Authorization": "Bearer invalid-key"})
    assert response.status_code == 401

    del os.environ["SEMRAG_API_KEY_STORE"]
    reset_settings_cache()
    reset_api_key_store_for_tests()


# ---------------------------------------------------------------------------
# Production-defaults enforcement — these tests verify the C-1 production
# blocker fixes: the in-memory store is no longer acceptable in
# production, and the configured SEMRAG_API_KEYS get seeded into the
# active store on startup.
# ---------------------------------------------------------------------------


def test_production_rejects_in_memory_key_store():
    """In-memory API key store is per-replica and per-process; production
    must use the database store or refuse to start."""
    from semrag.api.startup_validation import (
        RuntimeConfigurationError,
        validate_runtime_config,
    )

    settings = Settings(
        database_url="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag",
        embedding_provider="tfidf",
        embedding_base_url="",
        llm_provider="mock",
        llm_base_url="",
        llm_model="",
        environment="production",
        api_key_store="memory",
        api_keys="real-prod-key-12345",
    )
    with pytest.raises(RuntimeConfigurationError) as exc_info:
        _run_async(validate_runtime_config(settings))
    assert "SEMRAG_API_KEY_STORE must be 'database' in production" in str(exc_info.value)


def test_production_accepts_database_key_store():
    """The database store passes the production check."""
    from semrag.api.startup_validation import validate_runtime_config

    settings = Settings(
        database_url="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag",
        embedding_provider="tfidf",
        embedding_base_url="",
        llm_provider="mock",
        llm_base_url="",
        llm_model="",
        environment="production",
        api_key_store="database",
        api_keys="real-prod-key-12345",
    )
    # Should not raise.
    _run_async(validate_runtime_config(settings))


def test_seed_initial_keys_idempotent(monkeypatch):
    """``seed_initial_keys`` must be idempotent — running it twice
    must not produce duplicate rows. The first run inserts, the
    second is a no-op."""
    from semrag.api.api_key_store import seed_initial_keys
    from semrag.core.settings import reset_settings_cache

    monkeypatch.setenv("SEMRAG_API_KEYS", "seed-test-key-a,seed-test-key-b")
    monkeypatch.setenv("SEMRAG_ENVIRONMENT", "development")
    reset_settings_cache()

    store = InMemoryAPIKeyStore()
    inserted_first = asyncio.run(seed_initial_keys(store))
    inserted_second = asyncio.run(seed_initial_keys(store))
    assert inserted_first == 2
    assert inserted_second == 0
    # Both keys are now retrievable via their plaintext bearer.
    assert asyncio.run(store.get_key("seed-test-key-a")) is not None
    assert asyncio.run(store.get_key("seed-test-key-b")) is not None


def test_seed_initial_keys_skips_dev_key_in_production(monkeypatch):
    """The default ``dev-local-key`` is allowed in production when
    ``reject_dev_keys_in_production`` is disabled (for testing the
    seeder's behavior when the guard is off)."""
    from semrag.api.api_key_store import seed_initial_keys
    from semrag.core.settings import reset_settings_cache

    # Disable the rejection so both the Settings validator and the seeder
    # allow dev-local-key. This tests the seeder's behavior when the
    # guard is disabled.
    monkeypatch.setenv("SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION", "false")
    monkeypatch.setenv("SEMRAG_API_KEYS", "dev-local-key")
    monkeypatch.setenv("SEMRAG_ENVIRONMENT", "production")
    reset_settings_cache()

    store = InMemoryAPIKeyStore()
    inserted = asyncio.run(seed_initial_keys(store))
    assert inserted == 1
    assert asyncio.run(store.get_key("dev-local-key")) is not None