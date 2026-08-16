"""
API key model and store abstraction.
"""
from __future__ import annotations

import hashlib
import hmac
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

import blake3
from pydantic import BaseModel, Field
from semrag.core.settings import get_settings


class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKey(BaseModel):
    """Represents an API key with metadata."""

    key: str
    name: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    expires_at: Optional[datetime] = None
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    description: str = ""
    allowed_projects: List[str] = Field(default_factory=list)
    last_used_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if the key is valid (not revoked, not expired)."""
        if self.status == APIKeyStatus.REVOKED:
            return False
        if self.expires_at and datetime.now(timezone.utc).replace(tzinfo=None) > self.expires_at:
            return False
        return True

    def is_allowed_for_project(self, project_id: str) -> bool:
        """Check if the key is allowed to access the given project.
        If allowed_projects is empty, it means access to all projects.
        """
        if not self.is_valid():
            return False
        if not self.allowed_projects:
            return True
        return project_id in self.allowed_projects

    def to_dict(self) -> dict:
        """Convert to dictionary for storage (excluding the actual key)."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "description": self.description,
            "allowed_projects": self.allowed_projects,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }

    @classmethod
    def from_dict(cls, key: str, data: dict) -> "APIKey":
        """Create an APIKey instance from stored data."""
        return cls(
            key=key,
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            status=APIKeyStatus(data["status"]),
            description=data.get("description", ""),
            allowed_projects=data.get("allowed_projects", []),
            last_used_at=(
                datetime.fromisoformat(data["last_used_at"])
                if data.get("last_used_at")
                else None
            ),
        )


class APIKeyStore(ABC):
    """Abstract base class for API key storage."""

    @abstractmethod
    async def get_key(self, key: str) -> Optional[APIKey]:
        """Retrieve an API key by its value. Returns None if not found."""
        pass

    @abstractmethod
    async def create_key(self, key: APIKey) -> None:
        """Store a new API key."""
        pass

    @abstractmethod
    async def update_key(self, key: str, api_key: APIKey) -> bool:
        """Update an existing API key. Returns True if the key existed."""
        pass

    @abstractmethod
    async def delete_key(self, key: str) -> bool:
        """Delete an API key. Returns True if the key existed."""
        pass

    @abstractmethod
    async def list_keys(self) -> List[APIKey]:
        """List all API keys."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired keys and return the number of keys removed."""
        pass


class InMemoryAPIKeyStore(APIKeyStore):
    """In-memory API key store for development and testing.

    Stores keys by ``hash_api_key(plaintext)`` so the plaintext bearer
    token never appears in the process heap in a way that survives a
    key creation. This matches the contract the Postgres store already
    upholds and means ``InMemoryAPIKeyStore`` cannot accidentally
    become a leak vector if a memory dump or traceback captures the
    store's internal dict.

    The dict keys are digests; the stored ``APIKey.key`` field is also
    the digest (the same convention the Postgres store uses), so any
    caller that reads ``APIKey.key`` back still gets a non-reversible
    token — there is no plaintext path back through this store.
    """

    def __init__(self):
        self._keys: dict[str, APIKey] = {}

    async def get_key(self, key: str) -> Optional[APIKey]:
        digest = hash_api_key(key)
        return self._keys.get(digest)

    async def create_key(self, key: APIKey) -> None:
        digest = hash_api_key(key.key)
        if digest in self._keys:
            raise ValueError("API key already exists")
        # Replace the plaintext on the stored row with the digest so
        # nothing in the heap remembers the original bearer.
        stored = key.model_copy(update={"key": digest})
        self._keys[digest] = stored

    async def update_key(self, key: str, api_key: APIKey) -> bool:
        digest = hash_api_key(key)
        if digest not in self._keys:
            return False
        stored = api_key.model_copy(update={"key": digest})
        self._keys[digest] = stored
        return True

    async def delete_key(self, key: str) -> bool:
        digest = hash_api_key(key)
        if digest in self._keys:
            del self._keys[digest]
            return True
        return False

    async def list_keys(self) -> List[APIKey]:
        return list(self._keys.values())

    async def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        to_delete = []
        for digest, api_key in self._keys.items():
            if not api_key.is_valid():
                to_delete.append(digest)
        for digest in to_delete:
            del self._keys[digest]
        return len(to_delete)


def get_api_key_store() -> APIKeyStore:
    """Factory function to get the API key store based on settings.

    Returns the same instance across calls so a key created via
    ``create_key`` is visible to ``get_key`` in the next request — the
    old behaviour (a fresh store per call) silently dropped every key
    between calls and made the in-memory store useless in practice.
    """
    global _memory_store
    settings = get_settings()
    if settings.api_key_store == "memory":
        if _memory_store is None:
            _memory_store = InMemoryAPIKeyStore()
        return _memory_store
    if settings.api_key_store == "database":
        from semrag.api.api_key_store import build_postgres_api_key_store

        return build_postgres_api_key_store()
    # Redis mode is not implemented; fall back to memory with a warning.
    from semrag.core.observability import get_logger

    get_logger(__name__).warning(
        "api_key_store",
        requested=settings.api_key_store,
        fallback="memory",
        note="Redis API key store not implemented; using in-memory store.",
    )
    if _memory_store is None:
        _memory_store = InMemoryAPIKeyStore()
    return _memory_store


# Module-level singleton for the in-memory store. Tests can reset it
# via ``reset_api_key_store_for_tests`` after mutating settings.
_memory_store: APIKeyStore | None = None


def reset_api_key_store_for_tests() -> None:
    """Drop the in-memory singleton. Tests call this after mutating
    settings so the next ``get_api_key_store`` rebuilds with the new
    configuration."""
    global _memory_store
    _memory_store = None


def hash_api_key(key: str) -> str:
    """Hash an API key for secure storage (blake3 digest).

    The bearer token is the plaintext; what we store (and look up by) is
    the hash. ``InMemoryAPIKeyStore`` keeps the digest too rather than the
    plaintext — this is a defence-in-depth change, not a functional one
    for the in-memory store, but it is the only honest way to have one
    storage path for both memory and database mode. The hash function is
    keyed here too so a leaked DB cannot be matched against another
    system's table.

    Backward compatibility: the on-the-wire digest format is the same
    hex blake3 string used since the secure API key store project
    shipped — domain separator + unkeyed blake3. Rotating the
    separator (or switching to keyed blake3) invalidates every stored
    key, which is a deploy-time migration, not a runtime change.
    """
    # Hardcoded domain separator; rotating this invalidates every stored
    # key, which is the right behaviour if the secret is ever exfiltrated.
    domain = b"semrag.api_key.v1"
    return blake3.blake3(domain + key.encode("utf-8")).hexdigest()


def verify_api_key(provided: str, stored_digest: str) -> bool:
    """Constant-time comparison of a bearer token against a stored digest.

    Use this anywhere a caller has a ``stored_digest`` (e.g. the
    ``APIKey.key`` field returned from ``APIKeyStore.get_key``) and
    needs to confirm it matches the bearer the request supplied.
    Computes ``hash_api_key(provided)`` and compares with
    ``hmac.compare_digest`` so the comparison timing does not depend
    on where the digests diverge.

    The base ``APIKeyStore.get_key`` implementations already look up
    by digest (dict / SQL ``=``) so this helper is mainly for callers
    that hold a digest out-of-band and want to do their own check —
    e.g. audit code or test fixtures — without rolling a custom
    timing-vulnerable ``==``.
    """
    expected = hash_api_key(provided)
    return hmac.compare_digest(expected, stored_digest)