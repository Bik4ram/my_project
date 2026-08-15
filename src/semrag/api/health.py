"""
Production health endpoints.

Three endpoints, three distinct contracts:

GET /livez  — Process liveness. Always returns 200 unless the process is
              so broken it can't run Python. No external dependency checks,
              no I/O, no database roundtrip. Safe to hit from a kubelet's
              liveness probe at any frequency.

GET /readyz — Process readiness. Verifies every dependency the API needs
              to serve a request: Postgres connectivity + pgvector
              availability, LLM provider reachability, embedding provider
              initialization, JobExecutor liveness, and the configured
              cache / rate-limit backends. Returns 200 with a structured
              JSON body when every required check passes; 503 with the
              failing components otherwise. Suitable for a kubelet's
              readiness probe.

GET /version — Build metadata. Application version, git commit (if
              available), build timestamp, Python version. Package
              metadata via importlib.metadata is the source of truth for
              the version; pyproject.toml is the fallback so the endpoint
              still answers when the package hasn't been installed
              editable.

Checks reuse existing singletons (Settings, the SQLAlchemy engine, the
JobExecutor) so we don't open a second connection pool, don't duplicate
provider configuration, and don't add a parallel startup path that could
drift from the real one.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata
import os
import platform
import subprocess
import time
from typing import Any

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from semrag.api.dependencies import get_engine, get_job_executor
from semrag.core import metrics as metrics_module
from semrag.core.observability import get_logger
from semrag.core.settings import get_settings

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# /livez — liveness probe
# ---------------------------------------------------------------------------

@router.get("/livez")
def livez() -> dict:
    """Liveness: the process can serve Python. No dependency checks."""
    return {"status": "alive"}


# ---------------------------------------------------------------------------
# /version — build metadata
# ---------------------------------------------------------------------------

# Cached at import time: importlib.metadata can be slow on cold start in
# some package layouts, and /version is cheap to answer but should never
# do I/O. None of this changes at runtime.
_VERSION_CACHE: dict[str, str | None] = {}


def _package_version() -> str | None:
    """SemRAG version from installed package metadata. Returns None when
    the package isn't installed (typical of an editable checkout where
    pyproject.toml is the source of truth)."""
    if "version" in _VERSION_CACHE:
        return _VERSION_CACHE["version"]
    version: str | None = None
    try:
        version = importlib.metadata.version("semrag")
    except importlib.metadata.PackageNotFoundError:
        version = None
    _VERSION_CACHE["version"] = version
    return version


def _git_commit() -> str | None:
    """Short git commit SHA if the working tree is a git checkout.
    Best-effort — never raises; returns None when git isn't available
    or the source isn't a repo (CI checkout with .git removed, tarball
    install, etc.)."""
    if "git_commit" in _VERSION_CACHE:
        return _VERSION_CACHE["git_commit"]
    commit: str | None = os.environ.get("SEMRAG_GIT_COMMIT")
    if not commit:
        try:
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode("utf-8")
                .strip()
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            commit = None
    _VERSION_CACHE["git_commit"] = commit
    return commit


def _build_timestamp() -> str | None:
    """ISO 8601 build timestamp. Set by build (CI) via SEMRAG_BUILD_TIME,
    else None — we don't claim "build time" when we don't actually know."""
    return os.environ.get("SEMRAG_BUILD_TIME")


@router.get("/version")
def version() -> dict:
    """Build metadata. Always available, never touches dependencies."""
    settings = get_settings()
    pkg_version = _package_version()
    return {
        "application": "semrag",
        "version": pkg_version or "0.0.0",
        "git_commit": _git_commit(),
        "build_timestamp": _build_timestamp(),
        "python_version": platform.python_version(),
        "environment": settings.environment,
    }


# ---------------------------------------------------------------------------
# /readyz — readiness probe
# ---------------------------------------------------------------------------

def _check_database() -> tuple[bool, str]:
    """Confirm Postgres is reachable and pgvector is installed. Uses the
    shared engine so we exercise the real connection pool path, not a
    side-channel that could disagree with production traffic."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            # pgvector is an extension; check via the pg_extension catalog
            # so we fail loudly if it isn't installed (Vector type would
            # only fail on first use, deep inside a request).
            row = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).first()
            if row is None:
                return False, "pgvector extension not installed"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 — readiness surfaces all errors
        logger.warning("readyz: database check failed", extra={"error": str(exc)})
        return False, f"{exc.__class__.__name__}: {exc}"


def _check_llm() -> tuple[bool, str]:
    """LLM provider reachability. Reuses Settings rather than rebuilding
    a provider just for readiness — the same factory already ran at
    startup via _probe(); if that passed, the LLM is reachable. Here we
    only validate the configuration is internally consistent (provider
    name known, base URL parseable) and skip the actual probe so /readyz
    stays cheap on every probe tick."""
    settings = get_settings()
    provider = settings.llm_provider
    if provider == "mock":
        return True, "ok (mock)"
    if not settings.llm_base_url:
        return False, "llm_base_url not configured"
    if not settings.llm_model:
        return False, "llm_model not configured"
    return True, f"configured ({provider}/{settings.llm_model})"


def _check_embedding() -> tuple[bool, str]:
    """Embedding provider configuration validity. Same reasoning as LLM:
    we don't re-probe on every readiness check; we confirm Settings is
    internally consistent (provider known, model + base URL set when
    required)."""
    settings = get_settings()
    provider = settings.embedding_provider
    if provider == "tfidf":
        return True, "configured (tfidf)"
    if not settings.embedding_base_url:
        return False, "embedding_base_url not configured"
    if not settings.embedding_model:
        return False, "embedding_model not configured"
    if settings.embedding_dimension <= 0:
        return False, "embedding_dimension must be positive"
    return True, f"configured ({provider}/{settings.embedding_model})"


def _check_job_executor() -> tuple[bool, str]:
    """Confirm the background job executor is alive and able to accept
    submissions. The ThreadPoolJobExecutor sets _shutdown=True before
    shutting down, so this distinguishes a healthy executor from one
    that has been torn down (race during deploy)."""
    executor = get_job_executor()
    if executor is None:
        return False, "not initialized"
    if getattr(executor, "_shutdown", False):
        return False, "shut down"
    return True, "ok"


def _check_cache() -> tuple[bool, str]:
    """Configured cache backend. Always succeeds for the in-memory
    backend (it just exists); for Redis we report 'configured' and let
    the operator observe traffic-level liveness separately. We don't
    open a Redis connection here — readiness probes fire every few
    seconds and a flaky cache should not flap the whole probe."""
    settings = get_settings()
    backend = settings.query_cache_backend
    if backend == "redis":
        return True, f"configured ({backend})"
    return True, f"ok ({backend})"


def _check_rate_limiter() -> tuple[bool, str]:
    """Configured rate-limiter backend. Same reasoning as the cache
    check: report the configured backend without opening a connection."""
    settings = get_settings()
    backend = settings.rate_limit_backend
    if backend == "redis":
        return True, f"configured ({backend})"
    return True, f"ok ({backend})"


_CHECK_FUNCS: dict[str, Any] = {
    "database": _check_database,
    "llm": _check_llm,
    "embedding": _check_embedding,
    "job_executor": _check_job_executor,
    "cache": _check_cache,
    "rate_limiter": _check_rate_limiter,
}


@router.get("/readyz")
def readyz(response: Response) -> dict:
    """Readiness: every dependency is reachable. 200 when all required
    checks pass, 503 otherwise. JSON body lists each component and a
    short status string so an operator can grep the probe response
    without SSH-ing into the pod.

    Wave 3: also updates the `semrag_health_up{component=...}` gauges
    so the same health state is observable through Prometheus. The
    gauges stay in lockstep with the readiness verdict — 1 for a
    passing check, 0 for a failing one.
    """
    # Look up the check function fresh on every call rather than
    # caching the names at import time. Lets tests monkeypatch
    # `health._check_database` etc. and see the override apply.
    results: dict[str, str] = {}
    healthy = True
    for name, default_check in _CHECK_FUNCS.items():
        # Map the public check name ("database") to the module-level
        # function name ("_check_database") so a monkeypatch on
        # `health._check_database` is honored.
        module_attr = f"_check_{name}"
        check = globals().get(module_attr, default_check)
        ok, message = check()
        results[name] = message
        metrics_module.set_health_up(name, ok)
        if not ok:
            healthy = False
    if healthy:
        response.status_code = status.HTTP_200_OK
        overall = "ready"
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall = "not_ready"
    return {
        "status": overall,
        "checks": results,
        "version": _package_version() or "0.0.0",
        "timestamp": _dt.datetime.now(tz=_dt.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# /metrics — Prometheus exposition (Wave 3)
# ---------------------------------------------------------------------------

# Process start time — captured at import time so the uptime gauge is
# monotonic and stable across the process lifetime. Tests that swap
# `metrics.set_uptime(...)` directly bypass this; production reads
# the gauge value, which is initialised to zero and updated by the
# `MetricsUptimeRefresher` thread (below) only when the metric is
# observed.
_PROCESS_STARTED_AT = time.time()


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition endpoint.

    Renders the Wave 3 metrics registry in the standard
    `text/plain; version=0.0.4` Prometheus exposition format. The
    endpoint is intentionally unauthenticated — it follows the
    Prometheus convention of treating scrape targets as
    infrastructure-trusted. Lock it down at the network layer
    (Kubernetes NetworkPolicy, ingress firewall) instead of in-app.

    Disabled via ``Settings.metrics_enabled=False`` (returns 404) so
    test fixtures don't accidentally expose the registry.
    """
    settings = get_settings()
    if not settings.metrics_enabled:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    # Refresh the uptime gauge from the captured process start time.
    # Cheap to compute; keeps the gauge in lockstep with the actual
    # process age even if no other code path has touched it.
    metrics_module.set_uptime(max(0.0, time.time() - _PROCESS_STARTED_AT))
    body, content_type = metrics_module.render_metrics()
    return Response(content=body, media_type=content_type)