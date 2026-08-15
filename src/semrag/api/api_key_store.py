"""
Postgres-backed API key store.

The findings that motivated this:

- ``InMemoryAPIKeyStore`` lives in the API process. Each replica holds its
  own copy. Creating a key on replica A makes it invisible to replica B,
  and an API key created in dev is gone on the next process restart.
- The seed keys from ``SEMRAG_API_KEYS`` are loaded only into memory; an
  operator that needs to revoke a static key without restarting every
  replica has no path.
- For ``database`` mode we store a hash, not the plaintext. The bearer
  token is the plaintext, the lookup is by hash, so a DB compromise does
  not yield reusable tokens.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from semrag.api.api_key import APIKey, APIKeyStatus, APIKeyStore, hash_api_key
from semrag.core.settings import get_settings

_logger = logging.getLogger(__name__)


def _row_to_api_key(row) -> APIKey:
    return APIKey(
        key=row.key_hash,
        name=row.name,
        created_at=row.created_at,
        expires_at=row.expires_at,
        status=APIKeyStatus(row.status),
        description=row.description or "",
        allowed_projects=list(row.allowed_projects or []),
        last_used_at=row.last_used_at,
    )


class PostgresAPIKeyStore(APIKeyStore):
    """Persistent API key store backed by Postgres.

    The actual bearer token is never stored. ``hash_api_key`` produces a
    blake3 digest of the token; that digest is the primary key. Lookups
    hash the incoming bearer before querying, so a database dump cannot
    be replayed against the auth path.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory

    def _session(self) -> Session:
        return self._session_factory()

    async def get_key(self, key: str) -> Optional[APIKey]:
        digest = hash_api_key(key)
        with self._session() as session:
            row = session.execute(
                select(APIKeyRow).where(APIKeyRow.key_hash == digest)
            ).scalar_one_or_none()
            if row is None:
                return None
            return _row_to_api_key(row)

    async def create_key(self, key: APIKey) -> None:
        digest = hash_api_key(key.key)
        with self._session() as session:
            existing = session.execute(
                select(APIKeyRow).where(APIKeyRow.key_hash == digest)
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError("API key already exists")
            session.add(
                APIKeyRow(
                    key_hash=digest,
                    name=key.name,
                    created_at=key.created_at,
                    expires_at=key.expires_at,
                    status=key.status.value,
                    description=key.description,
                    allowed_projects=list(key.allowed_projects),
                    last_used_at=key.last_used_at,
                )
            )
            session.commit()

    async def update_key(self, key: str, api_key: APIKey) -> bool:
        digest = hash_api_key(key)
        with self._session() as session:
            row = session.execute(
                select(APIKeyRow).where(APIKeyRow.key_hash == digest)
            ).scalar_one_or_none()
            if row is None:
                return False
            row.name = api_key.name
            row.expires_at = api_key.expires_at
            row.status = api_key.status.value
            row.description = api_key.description
            row.allowed_projects = list(api_key.allowed_projects)
            row.last_used_at = api_key.last_used_at
            session.commit()
            return True

    async def delete_key(self, key: str) -> bool:
        digest = hash_api_key(key)
        with self._session() as session:
            row = session.execute(
                select(APIKeyRow).where(APIKeyRow.key_hash == digest)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    async def list_keys(self) -> List[APIKey]:
        with self._session() as session:
            rows = session.execute(select(APIKeyRow)).scalars().all()
            return [_row_to_api_key(r) for r in rows]

    async def cleanup_expired(self) -> int:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._session() as session:
            rows = session.execute(
                select(APIKeyRow).where(APIKeyRow.expires_at.is_not(None))
            ).scalars().all()
            removed = 0
            for row in rows:
                if row.expires_at is not None and now > row.expires_at:
                    session.delete(row)
                    removed += 1
            if removed:
                session.commit()
            return removed


# SQLAlchemy model — kept inline (rather than in storage/postgres/models.py)
# because the table is API-plane only and the Postgres package is reserved
# for domain data. Pydantic-style declarative base keeps the model
# independent of the main Base so importing this module never forces
# the rest of the API surface to load the wizardry of the storage package.


class _APIKeyBase(DeclarativeBase):
    pass


class APIKeyRow(_APIKeyBase):
    __tablename__ = "api_keys"

    key_hash: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    description: Mapped[Optional[str]] = mapped_column(String(1024))
    allowed_projects: Mapped[Optional[list]] = mapped_column(JSON)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))


def build_postgres_api_key_store() -> PostgresAPIKeyStore:
    """Factory used by ``get_api_key_store`` when persistence is requested."""
    from semrag.storage.postgres.session import build_session_factory

    settings = get_settings()
    return PostgresAPIKeyStore(build_session_factory(settings=settings))


def seed_initial_keys(store: APIKeyStore) -> int:
    """Insert the keys from ``SEMRAG_API_KEYS`` into the store when they
    aren't already present.

    Idempotent: keys that already exist (matched by hash) are skipped,
    so the helper is safe to call on every process start. This is what
    closes the "operator sets SEMRAG_API_KEYS but they vanish on every
    restart because the in-memory store loses them" gap that motivated
    moving to Postgres — without seeding, the new Postgres store would
    start empty and the operator's keys wouldn't work.

    Returns the number of NEW keys inserted (0 means "all configured
    keys were already present", which is the common case after the
    first boot)."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from semrag.api.api_key import APIKey, APIKeyStatus
    from semrag.core.settings import get_settings

    settings = get_settings()
    existing_digests: set[str] = set()
    # Best-effort digest pre-population so we don't try to insert a key
    # that's already there (which would raise). list_keys() returns the
    # stored APIKey objects whose `.key` field is the digest (set on
    # insertion by the stores), so we can read them directly.
    try:
        for k in asyncio.run(store.list_keys()):
            existing_digests.add(k.key)
    except Exception:  # noqa: BLE001 — seeding must never block startup
        _logger.warning("seed_initial_keys: list_keys failed; continuing", exc_info=True)

    seeded = 0
    for raw in settings.api_key_list():
        if raw == "dev-local-key" and settings.environment == "production":
            # The validator already rejects this, but a second line of
            # defence at the seeder is cheap and clear.
            _logger.warning("seed_initial_keys: skipping dev-local-key in production")
            continue
        digest = hash_api_key(raw)
        if digest in existing_digests:
            continue
        try:
            asyncio.run(
                store.create_key(
                    APIKey(
                        key=raw,
                        name=f"seeded-{digest[:8]}",
                        created_at=_dt.now(_tz.utc).replace(tzinfo=None),
                        expires_at=_dt.now(_tz.utc).replace(tzinfo=None) + _td(days=settings.api_key_expire_days)
                        if settings.api_key_expire_days > 0
                        else None,
                        status=APIKeyStatus.ACTIVE,
                        description="Seeded from SEMRAG_API_KEYS at startup",
                    )
                )
            )
            seeded += 1
        except ValueError:
            # Race: another replica seeded it first. Fine.
            continue
        except Exception:  # noqa: BLE001
            _logger.warning(
                "seed_initial_keys: insert failed for hashed digest %s...", digest[:8],
                exc_info=True,
            )
    if seeded:
        _logger.info("seed_initial_keys: inserted %d new keys", seeded)
    return seeded
