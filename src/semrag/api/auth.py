"""
API key authentication.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from semrag.api.api_key import APIKey, APIKeyStore, get_api_key_store, verify_api_key


async def get_api_key_store_dependency() -> APIKeyStore:
    """Dependency that returns the API key store."""
    return get_api_key_store()


async def require_api_key(
    authorization: str | None = Header(default=None),
    store: APIKeyStore = Depends(get_api_key_store_dependency),
) -> APIKey:
    """Validate the API key and return the APIKey object if valid.

    This dependency also updates the last_used_at timestamp for the key.

    Lookup path is constant-time by construction: the bearer is hashed
    once on entry (``hash_api_key`` is keyed blake3 over a domain
    separator — fixed-cost for any input length) and the store lookup
    matches by digest (in-memory dict key, or SQL ``key_hash = digest``
    equality). ``verify_api_key`` re-checks the digest in constant
    time after fetch as a defense-in-depth confirmation that the
    stored row's digest really matches the bearer the request sent —
    protects against any future store implementation that might do
    case-sensitive substring matching instead of a fixed-width compare.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <api_key>",
        )

    api_key = authorization.removeprefix("Bearer ").strip()
    stored_key = await store.get_key(api_key)
    if stored_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    # Defense-in-depth: re-confirm the stored digest matches the bearer
    # via constant-time comparison. The store's lookup already used the
    # digest, but a misconfigured future store could regress to a
    # non-constant-time compare and we don't want that to silently
    # become the source of a side channel.
    if not verify_api_key(api_key, stored_key.key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    if not stored_key.is_valid():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Update last_used_at
    stored_key.last_used_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await store.update_key(api_key, stored_key)

    return stored_key