"""
Prometheus metrics registry and metric definitions for Wave 3.

The `prometheus_client` library gives us a process-global default
REGISTRY; this module installs a private `REGISTRY` so test isolation
is straightforward (a new registry per process/test). All metric
objects are constructed once at import time and reused — counter /
histogram construction in a hot path (per request) would itself
become a measurable cost.

Metric inventory (Wave 3 §1)
----------------------------

HTTP
    semrag_http_requests_total{method,path,status}
    semrag_http_request_duration_seconds{method,path,status}
    semrag_http_request_size_bytes{method,path}
    semrag_http_response_size_bytes{method,path}

RAG
    semrag_rag_retrieval_duration_seconds{stage}
    semrag_rag_rerank_duration_seconds{}
    semrag_rag_embedding_duration_seconds{provider}
    semrag_rag_llm_duration_seconds{provider,model}
    semrag_rag_tool_duration_seconds{tool}

Indexing
    semrag_indexing_duration_seconds{project_id}
    semrag_indexing_files_processed_total{project_id}
    semrag_indexing_chunks_generated_total{project_id}
    semrag_indexing_chunks_embedded_total{project_id}
    semrag_indexing_jobs_queued_total{project_id}
    semrag_indexing_jobs_completed_total{project_id}
    semrag_indexing_jobs_failed_total{project_id}

Database
    semrag_db_pool_connections{state}        # gauge (active/idle)
    semrag_db_query_duration_seconds{op}
    semrag_db_transactions_total{op}

Cache
    semrag_cache_hits_total{backend}
    semrag_cache_misses_total{backend}
    semrag_cache_evictions_total{backend}

Rate limiter
    semrag_rate_limit_accepted_total{}
    semrag_rate_limit_rejected_total{}

Uptime
    semrag_uptime_seconds

Health (gauges)
    semrag_health_up{component}

We deliberately keep cardinality bounded:

- HTTP `path` is the FastAPI route template (`/v1/projects/{project_id}/query`)
  not the rendered URL — never label on a project_id, which would
  explode the series count with every new project.
- Database / query `op` is a small enum (`select`, `insert`, `update`,
  `delete`, `commit`, `rollback`) — not a raw SQL string.
- Cache `backend` is `memory` or `redis` — two values.
- Tool names are bounded to the ToolRegistry's surface (today:
  read_file / list_files / search_codebase).

Notes for test usage
--------------------

`reset_registry()` tears down the process-global REGISTRY and rebuilds
it. Tests call it in a fixture so a registry polluted by one test
never leaks into the next. Production never sees this path.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


def _new_registry() -> CollectorRegistry:
    """Create a fresh registry. Module-level singleton uses this at
    import time; tests call ``reset_registry`` to swap it out."""
    return CollectorRegistry()


# Module-level REGISTRY — one per process. `reset_registry` swaps this
# for tests; production never calls `reset_registry`.
REGISTRY: CollectorRegistry = _new_registry()


# ---------------------------------------------------------------------------
# Buckets — chosen to cover the latency spectrum the pipeline produces.
# LLM calls cap at 90s (Settings.llm_timeout_seconds); embedding calls
# at 300s. The buckets below cover 5ms .. 100s; anything beyond lands
# in +Inf without affecting the histogram count.
# ---------------------------------------------------------------------------

# Default Prometheus client buckets are tuned for web requests
# (~5ms .. 10s). For LLM / embedding we need the long tail too.
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)
# Smaller buckets for fast in-process stages (vector search, fusion).
_FAST_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
)
# Bytes: 100B .. 5MB (covers request bodies up to Settings.max_body_size).
_SIZE_BUCKETS: tuple[float, ...] = (
    100, 1_000, 10_000, 100_000, 1_000_000, 5_000_000,
)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "semrag_http_requests_total",
    "HTTP requests served, labelled by method, route template, and status code.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "semrag_http_request_duration_seconds",
    "End-to-end HTTP request latency in seconds.",
    ["method", "path", "status"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
HTTP_REQUEST_SIZE = Histogram(
    "semrag_http_request_size_bytes",
    "HTTP request body size in bytes.",
    ["method", "path"],
    buckets=_SIZE_BUCKETS,
    registry=REGISTRY,
)
HTTP_RESPONSE_SIZE = Histogram(
    "semrag_http_response_size_bytes",
    "HTTP response body size in bytes.",
    ["method", "path"],
    buckets=_SIZE_BUCKETS,
    registry=REGISTRY,
)
HTTP_EXCEPTIONS = Counter(
    "semrag_http_exceptions_total",
    "Exceptions raised while serving an HTTP request.",
    ["method", "path", "exception"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

RAG_RETRIEVAL_DURATION = Histogram(
    "semrag_rag_retrieval_duration_seconds",
    "Retrieval stage latency. stage=vector|lexical|fusion|graph_expansion|rerank.",
    ["stage"],
    buckets=_FAST_BUCKETS,
    registry=REGISTRY,
)
RAG_RERANK_DURATION = Histogram(
    "semrag_rag_rerank_duration_seconds",
    "Reranker latency (cross-encoder or lexical overlap).",
    [],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
RAG_EMBEDDING_DURATION = Histogram(
    "semrag_rag_embedding_duration_seconds",
    "Embedding provider call latency.",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
RAG_LLM_DURATION = Histogram(
    "semrag_rag_llm_duration_seconds",
    "LLM provider call latency (non-streaming completion).",
    ["provider", "model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
RAG_TOOL_DURATION = Histogram(
    "semrag_rag_tool_duration_seconds",
    "Tool execution latency in the agent loop.",
    ["tool"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

INDEXING_DURATION = Histogram(
    "semrag_indexing_duration_seconds",
    "Indexing job wall-clock duration.",
    ["project_id"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
INDEXING_FILES_PROCESSED = Counter(
    "semrag_indexing_files_processed_total",
    "Files processed during indexing (added + modified + deleted).",
    ["project_id"],
    registry=REGISTRY,
)
INDEXING_CHUNKS_GENERATED = Counter(
    "semrag_indexing_chunks_generated_total",
    "Chunks produced by the chunker during indexing.",
    ["project_id"],
    registry=REGISTRY,
)
INDEXING_CHUNKS_EMBEDDED = Counter(
    "semrag_indexing_chunks_embedded_total",
    "Chunks embedded by the embedding provider during indexing.",
    ["project_id"],
    registry=REGISTRY,
)
INDEXING_JOBS_QUEUED = Counter(
    "semrag_indexing_jobs_queued_total",
    "Indexing jobs enqueued.",
    ["project_id"],
    registry=REGISTRY,
)
INDEXING_JOBS_COMPLETED = Counter(
    "semrag_indexing_jobs_completed_total",
    "Indexing jobs that finished successfully.",
    ["project_id"],
    registry=REGISTRY,
)
INDEXING_JOBS_FAILED = Counter(
    "semrag_indexing_jobs_failed_total",
    "Indexing jobs that ended in the FAILED state.",
    ["project_id"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

# Gauge keyed by `state` = active | idle | overflow. We don't try to
# keep this in lockstep with SQLAlchemy's pool internals (no public
# API for that on every version); the value is updated by the
# metrics middleware on each request so it tracks observed pool
# behaviour, not pool internals.
DB_POOL_CONNECTIONS = Gauge(
    "semrag_db_pool_connections",
    "Database connection pool state — observed active and idle connections.",
    ["state"],
    registry=REGISTRY,
)
DB_QUERY_DURATION = Histogram(
    "semrag_db_query_duration_seconds",
    "Database query latency.",
    ["op"],
    buckets=_FAST_BUCKETS,
    registry=REGISTRY,
)
DB_TRANSACTIONS = Counter(
    "semrag_db_transactions_total",
    "Database transactions completed (commit + rollback).",
    ["op"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_HITS = Counter(
    "semrag_cache_hits_total",
    "Query cache hits.",
    ["backend"],
    registry=REGISTRY,
)
CACHE_MISSES = Counter(
    "semrag_cache_misses_total",
    "Query cache misses.",
    ["backend"],
    registry=REGISTRY,
)
CACHE_EVICTIONS = Counter(
    "semrag_cache_evictions_total",
    "Query cache entries evicted (TTL expiry or invalidation).",
    ["backend"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

RATE_LIMIT_ACCEPTED = Counter(
    "semrag_rate_limit_accepted_total",
    "Requests admitted by the rate limiter.",
    [],
    registry=REGISTRY,
)
RATE_LIMIT_REJECTED = Counter(
    "semrag_rate_limit_rejected_total",
    "Requests rejected by the rate limiter (HTTP 429).",
    [],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Health & uptime
# ---------------------------------------------------------------------------

UPTIME = Gauge(
    "semrag_uptime_seconds",
    "Seconds since process start.",
    registry=REGISTRY,
)
HEALTH_UP = Gauge(
    "semrag_health_up",
    "Component health gauge (1 = healthy, 0 = unhealthy).",
    ["component"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def render_metrics() -> tuple[bytes, str]:
    """Return the (body, content_type) pair for an HTTP response.

    The body is the Prometheus exposition format; the content type is
    the registered Prometheus text format version (`text/plain;
    version=0.0.4`). Hand the pair straight to FastAPI's
    ``Response(content=..., media_type=...)``.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def reset_registry() -> None:
    """Replace the module-level registry with a fresh one.

    Tests call this in a fixture so a registry polluted by one test
    doesn't leak into the next. Production never sees this path —
    the registry lives for the lifetime of the process.
    """
    global REGISTRY
    REGISTRY = _new_registry()
    # Re-bind every metric on the new registry. `prometheus_client`
    # counters/histograms/gauges hold an immutable reference to the
    # registry they were constructed with, so a `REGISTRY = ...`
    # swap alone would leave the old objects still tied to the old
    # registry. Recreate them so subsequent ``observe()`` /
    # ``inc()`` calls land in the new registry.
    _rebind_metrics()


def _rebind_metrics() -> None:
    """Rebind every module-level metric onto the current REGISTRY.

    Called from ``reset_registry``. Idempotent — repeated calls
    replace the bindings again. Metrics retain their `name` and
    `documentation` strings; this only swaps the registry.
    """
    global HTTP_REQUESTS, HTTP_REQUEST_DURATION, HTTP_REQUEST_SIZE, HTTP_RESPONSE_SIZE, HTTP_EXCEPTIONS
    global RAG_RETRIEVAL_DURATION, RAG_RERANK_DURATION, RAG_EMBEDDING_DURATION, RAG_LLM_DURATION, RAG_TOOL_DURATION
    global INDEXING_DURATION, INDEXING_FILES_PROCESSED, INDEXING_CHUNKS_GENERATED, INDEXING_CHUNKS_EMBEDDED
    global INDEXING_JOBS_QUEUED, INDEXING_JOBS_COMPLETED, INDEXING_JOBS_FAILED
    global DB_POOL_CONNECTIONS, DB_QUERY_DURATION, DB_TRANSACTIONS
    global CACHE_HITS, CACHE_MISSES, CACHE_EVICTIONS
    global RATE_LIMIT_ACCEPTED, RATE_LIMIT_REJECTED
    global UPTIME, HEALTH_UP

    HTTP_REQUESTS = Counter(
        "semrag_http_requests_total",
        "HTTP requests served, labelled by method, route template, and status code.",
        ["method", "path", "status"],
        registry=REGISTRY,
    )
    HTTP_REQUEST_DURATION = Histogram(
        "semrag_http_request_duration_seconds",
        "End-to-end HTTP request latency in seconds.",
        ["method", "path", "status"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    HTTP_REQUEST_SIZE = Histogram(
        "semrag_http_request_size_bytes",
        "HTTP request body size in bytes.",
        ["method", "path"],
        buckets=_SIZE_BUCKETS,
        registry=REGISTRY,
    )
    HTTP_RESPONSE_SIZE = Histogram(
        "semrag_http_response_size_bytes",
        "HTTP response body size in bytes.",
        ["method", "path"],
        buckets=_SIZE_BUCKETS,
        registry=REGISTRY,
    )
    HTTP_EXCEPTIONS = Counter(
        "semrag_http_exceptions_total",
        "Exceptions raised while serving an HTTP request.",
        ["method", "path", "exception"],
        registry=REGISTRY,
    )
    RAG_RETRIEVAL_DURATION = Histogram(
        "semrag_rag_retrieval_duration_seconds",
        "Retrieval stage latency.",
        ["stage"],
        buckets=_FAST_BUCKETS,
        registry=REGISTRY,
    )
    RAG_RERANK_DURATION = Histogram(
        "semrag_rag_rerank_duration_seconds",
        "Reranker latency.",
        [],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    RAG_EMBEDDING_DURATION = Histogram(
        "semrag_rag_embedding_duration_seconds",
        "Embedding provider call latency.",
        ["provider"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    RAG_LLM_DURATION = Histogram(
        "semrag_rag_llm_duration_seconds",
        "LLM provider call latency.",
        ["provider", "model"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    RAG_TOOL_DURATION = Histogram(
        "semrag_rag_tool_duration_seconds",
        "Tool execution latency.",
        ["tool"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    INDEXING_DURATION = Histogram(
        "semrag_indexing_duration_seconds",
        "Indexing job wall-clock duration.",
        ["project_id"],
        buckets=_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    INDEXING_FILES_PROCESSED = Counter(
        "semrag_indexing_files_processed_total",
        "Files processed during indexing.",
        ["project_id"],
        registry=REGISTRY,
    )
    INDEXING_CHUNKS_GENERATED = Counter(
        "semrag_indexing_chunks_generated_total",
        "Chunks produced by the chunker during indexing.",
        ["project_id"],
        registry=REGISTRY,
    )
    INDEXING_CHUNKS_EMBEDDED = Counter(
        "semrag_indexing_chunks_embedded_total",
        "Chunks embedded during indexing.",
        ["project_id"],
        registry=REGISTRY,
    )
    INDEXING_JOBS_QUEUED = Counter(
        "semrag_indexing_jobs_queued_total",
        "Indexing jobs enqueued.",
        ["project_id"],
        registry=REGISTRY,
    )
    INDEXING_JOBS_COMPLETED = Counter(
        "semrag_indexing_jobs_completed_total",
        "Indexing jobs that finished successfully.",
        ["project_id"],
        registry=REGISTRY,
    )
    INDEXING_JOBS_FAILED = Counter(
        "semrag_indexing_jobs_failed_total",
        "Indexing jobs that ended in the FAILED state.",
        ["project_id"],
        registry=REGISTRY,
    )
    DB_POOL_CONNECTIONS = Gauge(
        "semrag_db_pool_connections",
        "Database connection pool state.",
        ["state"],
        registry=REGISTRY,
    )
    DB_QUERY_DURATION = Histogram(
        "semrag_db_query_duration_seconds",
        "Database query latency.",
        ["op"],
        buckets=_FAST_BUCKETS,
        registry=REGISTRY,
    )
    DB_TRANSACTIONS = Counter(
        "semrag_db_transactions_total",
        "Database transactions completed.",
        ["op"],
        registry=REGISTRY,
    )
    CACHE_HITS = Counter(
        "semrag_cache_hits_total",
        "Query cache hits.",
        ["backend"],
        registry=REGISTRY,
    )
    CACHE_MISSES = Counter(
        "semrag_cache_misses_total",
        "Query cache misses.",
        ["backend"],
        registry=REGISTRY,
    )
    CACHE_EVICTIONS = Counter(
        "semrag_cache_evictions_total",
        "Query cache entries evicted.",
        ["backend"],
        registry=REGISTRY,
    )
    RATE_LIMIT_ACCEPTED = Counter(
        "semrag_rate_limit_accepted_total",
        "Requests admitted by the rate limiter.",
        [],
        registry=REGISTRY,
    )
    RATE_LIMIT_REJECTED = Counter(
        "semrag_rate_limit_rejected_total",
        "Requests rejected by the rate limiter.",
        [],
        registry=REGISTRY,
    )
    UPTIME = Gauge(
        "semrag_uptime_seconds",
        "Seconds since process start.",
        registry=REGISTRY,
    )
    HEALTH_UP = Gauge(
        "semrag_health_up",
        "Component health gauge.",
        ["component"],
        registry=REGISTRY,
    )


# ---------------------------------------------------------------------------
# Convenience helpers — short names for the most common operations.
# ---------------------------------------------------------------------------

def observe_http(method: str, path: str, status: int, duration_s: float, request_bytes: int, response_bytes: int) -> None:
    """One-call recording for a completed HTTP request."""
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, path=path, status=str(status)).observe(duration_s)
    if request_bytes >= 0:
        HTTP_REQUEST_SIZE.labels(method=method, path=path).observe(request_bytes)
    if response_bytes >= 0:
        HTTP_RESPONSE_SIZE.labels(method=method, path=path).observe(response_bytes)


def inc_http_exception(method: str, path: str, exc_name: str) -> None:
    HTTP_EXCEPTIONS.labels(method=method, path=path, exception=exc_name).inc()


def observe_retrieval(stage: str, duration_s: float) -> None:
    RAG_RETRIEVAL_DURATION.labels(stage=stage).observe(duration_s)


def observe_rerank(duration_s: float) -> None:
    RAG_RERANK_DURATION.observe(duration_s)


def observe_embedding(provider: str, duration_s: float) -> None:
    RAG_EMBEDDING_DURATION.labels(provider=provider).observe(duration_s)


def observe_llm(provider: str, model: str, duration_s: float) -> None:
    RAG_LLM_DURATION.labels(provider=provider, model=model).observe(duration_s)


def observe_tool(tool: str, duration_s: float) -> None:
    RAG_TOOL_DURATION.labels(tool=tool).observe(duration_s)


def observe_db_query(op: str, duration_s: float) -> None:
    DB_QUERY_DURATION.labels(op=op).observe(duration_s)


def inc_db_transaction(op: str) -> None:
    DB_TRANSACTIONS.labels(op=op).inc()


def set_db_pool(state: str, value: float) -> None:
    DB_POOL_CONNECTIONS.labels(state=state).set(value)


def inc_cache_hit(backend: str) -> None:
    CACHE_HITS.labels(backend=backend).inc()


def inc_cache_miss(backend: str) -> None:
    CACHE_MISSES.labels(backend=backend).inc()


def inc_cache_eviction(backend: str) -> None:
    CACHE_EVICTIONS.labels(backend=backend).inc()


def inc_rate_limit_accepted() -> None:
    RATE_LIMIT_ACCEPTED.inc()


def inc_rate_limit_rejected() -> None:
    RATE_LIMIT_REJECTED.inc()


def set_uptime(seconds: float) -> None:
    UPTIME.set(seconds)


def set_health_up(component: str, healthy: bool) -> None:
    HEALTH_UP.labels(component=component).set(1 if healthy else 0)