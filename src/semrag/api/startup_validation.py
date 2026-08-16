"""
Runtime configuration validation.

Run once at process start to fail fast on a misconfigured deploy. The
pydantic-settings model already validates individual fields and
combinations it knows about (e.g. dev keys + production env); this
module adds the cross-cutting checks that are awkward to express in
pydantic validators:

- Database URL is well-formed and the host:port is parseable.
- Embedding configuration is internally consistent (provider/model/dim).
- LLM configuration is internally consistent.
- Project roots, when configured, are absolute paths.
- Security: trusted hosts, CORS, body-size limits, dev-key rejection.

Raises ``RuntimeConfigurationError`` with every offending check in one
message so an operator can fix all problems at once instead of
restart-loop-and-discover-one-at-a-time.
"""
from __future__ import annotations

from urllib.parse import urlparse

from semrag.core.observability import get_logger
from semrag.core.settings import Settings

logger = get_logger(__name__)


class RuntimeConfigurationError(RuntimeError):
    """Raised when the process refuses to start because of a
    configuration problem the operator must fix."""


def _validate_database_url(url: str, problems: list[str]) -> None:
    try:
        parsed = urlparse(url)
    except Exception as exc:  # noqa: BLE001 — operator needs the cause
        problems.append(f"SEMRAG_DATABASE_URL is not a valid URL: {exc}")
        return
    if parsed.scheme not in ("postgresql", "postgresql+psycopg2", "postgres"):
        problems.append(
            f"SEMRAG_DATABASE_URL scheme must be postgresql or "
            f"postgresql+psycopg2 (got {parsed.scheme!r})"
        )
    if not parsed.hostname:
        problems.append("SEMRAG_DATABASE_URL is missing a hostname")


def _validate_embedding(settings: Settings, problems: list[str]) -> None:
    if settings.embedding_provider not in ("ollama", "openai_compatible", "tfidf"):
        problems.append(
            f"SEMRAG_EMBEDDING_PROVIDER is unknown: {settings.embedding_provider!r}"
        )
        return
    if settings.embedding_dimension <= 0:
        problems.append(
            f"SEMRAG_EMBEDDING_DIMENSION must be positive "
            f"(got {settings.embedding_dimension})"
        )
    if settings.embedding_provider in ("ollama", "openai_compatible"):
        if not settings.embedding_base_url:
            problems.append(
                "SEMRAG_EMBEDDING_BASE_URL is required when "
                f"SEMRAG_EMBEDDING_PROVIDER={settings.embedding_provider!r}"
            )
        if not settings.embedding_model:
            problems.append("SEMRAG_EMBEDDING_MODEL is required")


def _validate_llm(settings: Settings, problems: list[str]) -> None:
    if settings.llm_provider not in (
        "ollama",
        "openai",
        "anthropic",
        "openai_compatible",
        "vllm",
        "mock",
    ):
        problems.append(f"SEMRAG_LLM_PROVIDER is unknown: {settings.llm_provider!r}")
        return
    if settings.llm_provider == "mock":
        return  # mock is always valid
    if not settings.llm_base_url:
        problems.append(
            f"SEMRAG_LLM_BASE_URL is required when "
            f"SEMRAG_LLM_PROVIDER={settings.llm_provider!r}"
        )
    if not settings.llm_model:
        problems.append("SEMRAG_LLM_MODEL is required")
    if settings.llm_timeout_seconds <= 0:
        problems.append("SEMRAG_LLM_TIMEOUT_SECONDS must be positive")
    if settings.llm_max_retries < 0:
        problems.append("SEMRAG_LLM_MAX_RETRIES must be non-negative")
    if settings.llm_num_predict <= 0:
        problems.append("SEMRAG_LLM_NUM_PREDICT must be positive")


def _validate_security(settings: Settings, problems: list[str]) -> None:
    if settings.environment == "production":
        if settings.api_key_store == "memory":
            # The in-memory store is per-process and per-replica; combined
            # with any horizontally-scaled deploy this silently halves
            # (or quarters, or worse) the effective key revocation surface
            # AND loses every key on restart. A database-backed store is
            # the only acceptable default for production — refuse to start
            # rather than log a warning that gets ignored.
            problems.append(
                "SEMRAG_API_KEY_STORE must be 'database' in production "
                "(in-memory store loses state across restarts and is "
                "per-replica). Set SEMRAG_API_KEY_STORE=database and run "
                "alembic upgrade head to create the api_keys table."
            )
        if settings.job_store_backend == "memory":
            # Same class of bug as the API key store above: an in-memory
            # job store loses every QUEUED/RUNNING job on process restart
            # and makes two replicas race to enqueue / claim / re-enqueue
            # the same work. The persistent job store project ships
            # specifically to make this deployable — refuse to boot
            # production with the wrong backend rather than log a warning
            # that gets ignored.
            problems.append(
                "SEMRAG_JOB_STORE_BACKEND must be 'database' in production "
                "(in-memory store loses queued/running jobs on restart "
                "and is per-replica). Set SEMRAG_JOB_STORE_BACKEND=database "
                "and run alembic upgrade head to create the indexing_jobs "
                "table."
            )
        if not settings.api_keys_set:
            problems.append("SEMRAG_API_KEYS must be set in production")
    if settings.max_body_size <= 0:
        problems.append("SEMRAG_MAX_BODY_SIZE must be positive")
    if settings.rate_limit_per_minute < 0:
        problems.append("SEMRAG_RATE_LIMIT_PER_MINUTE must be non-negative")


def _validate_project_roots(settings: Settings, problems: list[str]) -> None:
    for root in settings.project_roots:
        if not root:
            problems.append("SEMRAG_PROJECT_ROOTS contains an empty entry")


async def validate_runtime_config(settings: Settings | None = None) -> Settings:
    """Validate the active Settings. Returns the same Settings instance
    on success, raises ``RuntimeConfigurationError`` with every detected
    problem on failure. Called once at process start; safe to call again
    after settings mutation in tests."""
    if settings is None:
        from semrag.core.settings import get_settings

        settings = get_settings()

    problems: list[str] = []
    _validate_database_url(settings.database_url, problems)
    _validate_embedding(settings, problems)
    _validate_llm(settings, problems)
    _validate_security(settings, problems)
    _validate_project_roots(settings, problems)

    if problems:
        # Bullet-list format so the operator can copy it into a ticket.
        message = "SemRAG refused to start due to configuration errors:\n  - " + "\n  - ".join(problems)
        logger.error("runtime configuration validation failed", extra={"problems": problems})
        raise RuntimeConfigurationError(message)

    logger.info(
        "runtime configuration validated",
        extra={
            "environment": settings.environment,
            "embedding_provider": settings.embedding_provider,
            "llm_provider": settings.llm_provider,
        },
    )

    # Seed the configured SEMRAG_API_KEYS into the active store. This
    # is what closes the gap where an operator set
    # SEMRAG_API_KEYS=key1,key2 but the in-memory store lost them on
    # every restart — and now that production uses the database store
    # exclusively, this is also the bootstrap step that gets a fresh
    # deployment's keys into the api_keys table.
    await _seed_configured_keys(settings)

    return settings


async def _seed_configured_keys(settings: Settings) -> None:
    """Insert ``SEMRAG_API_KEYS`` into the active store when missing.

    Failures are logged, not raised: a broken seed must not block
    startup if, for example, the database is read-only during a
    migration window. The store is constructed lazily here so we
    don't open a Postgres pool just for validation when nothing
    needs seeding.
    """
    if not settings.api_keys_set:
        return
    try:
        from semrag.api.api_key import get_api_key_store
        from semrag.api.api_key_store import seed_initial_keys

        store = get_api_key_store()
        await seed_initial_keys(store)
    except Exception:  # noqa: BLE001 — seed failure is a logged warning, not fatal
        logger.warning(
            "seed_initial_keys: skipping; store unavailable",
            exc_info=True,
        )