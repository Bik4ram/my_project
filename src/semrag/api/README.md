# `src/semrag/api` — HTTP API Package

> **The SemRAG HTTP surface.** This README is the engineering map for the
> FastAPI application: why the package exists, what each module owns, how
> a request actually flows through the middleware stack and dependency
> graph, and how to extend it safely. It is `api`-specific; the
> end-to-end stage walkthroughs, dependency-injection philosophy, and
> cross-package boundaries live in
> [developer_guide.md](../../../docs/developer_guide.md). For "where is X?"
> use [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md). For the launch
> recipe, see [RUN_API.md](../../../RUN_API.md).

Intended audience: **Developers** and **Maintainers** who touch the HTTP
layer.

---

## 1. Purpose

`semrag.api` is everything a client talks to. It exposes indexing, file
introspection, prompt-construction, and answer generation over HTTP while
keeping three responsibilities in one place:

1. **The FastAPI application assembly** — `app.py` builds the app,
   wires middleware, registers routers, and runs the process lifespan
   (startup validation, logging/tracing bootstrap, in-process worker).
2. **Per-request cross-cutting concerns** — authentication, per-project
   authorization, rate limiting, query caching, request IDs, structured
   access logging, metrics, tracing, and the global error envelope.
3. **The translation seam** — routers convert HTTP payloads (Pydantic
   schemas in `schemas.py`) into calls on `semrag.services`; they never
   implement business logic themselves.

The API follows the package's architectural rules: routes delegate to
`services` (never to `ingestion`/`retrieval` primitives directly), every
pluggable seam is a Protocol/ABC with a factory, and backends fail loudly
rather than silently falling back. See
[Boundaries](../README.md#4-dependency-relationships) and
[developer_guide.md §15](../../../docs/developer_guide.md#15-common-architectural-rules).

> **Not a spec.** The OpenAPI schema served at `/docs` is the machine
> contract; this document is the *why* and the *where to change*.

---

## 2. Package structure

```
src/semrag/api/
├── __init__.py                empty; package marker
├── app.py                     app factory, lifespan, router registration
├── routers/
│   ├── __init__.py            empty; package marker
│   ├── projects.py            /index, /files, /jobs — indexing & listing
│   ├── query.py               /query (sync) and /query/stream (SSE)
│   └── answer.py              /answer and /answer/stream (agent loop)
├── auth.py                    require_api_key — Bearer validation
├── api_key.py                 APIKey model, APIKeyStore ABC, hashing helpers
├── api_key_store.py           PostgresAPIKeyStore + seed_initial_keys
├── project_auth.py            get_project_authorizer — per-project gate
├── config.py                  API-side env-var readers
├── dependencies.py            shared FastAPI dependencies + singletons
├── exception_handler.py       global exception → standard error envelope
├── health.py                  /livez, /readyz, /version, /metrics
├── middleware.py              request ID, tracing, metrics, logging, security, body cap
├── query_cache.py             QueryCache ABC + TTL + Redis implementations
├── rate_limit.py              RateLimiter ABC + Memory + Redis implementations
├── redis_client.py            sync/async Redis client factories
├── schemas.py                 Pydantic request/response models
└── startup_validation.py      fail-fast runtime config validation + key seeding
```

Dependencies: `core` (settings, observability, metrics, tracing),
`services`, `retrieval`, `reasoning`, `jobs`. Routes import from
`services`, not from lower primitives.

---

## 3. Module responsibilities

### `app.py` — application assembly and lifespan

The only file that constructs the `FastAPI` app. `create_app()`:

1. Sets the OpenAPI title/version/description.
2. Attaches the lifespan.
3. Calls `setup_middleware(app)` (see §7).
4. Calls `add_exception_handlers(app)` (see §9).
5. Registers the four routers: `projects`, `query`, `answer`, `health`.
6. Defines the backwards-compatible **`GET /health`** alias
   (`{"status": "ok"}`) kept for older docker `HEALTHCHECK` directives.

The **lifespan** (`_lifespan`) is where the process boots and tears down:

```text
startup:
  validate_runtime_config()        fail-fast on misconfigured deploy
  configure_logging(settings)
  configure_tracing(settings)
  if settings.api_runs_worker:     start in-process WorkerService thread
shutdown:
  worker.stop()
  worker_thread.join(timeout=5)
  shutdown_resources()             drain job executor + dispose DB engine
```

Production multi-replica deployments set `SEMRAG_API_RUNS_WORKER=false`
and run a dedicated `semrag-worker` process instead.

### `routers/` — the four route groups

All business routers share a `prefix="/v1/projects"`. The `health`
router has no prefix.

| Router    | Endpoint and method            | Responsibility                                                      |
| --------- | ------------------------------ | ------------------------------------------------------------------- |
| projects  | `POST /v1/projects/{project_id}/index`  | Validate root path, enqueue an indexing **job**, return job meta. Returns **202** if a new job was created, **200** if an identical job already exists. |
| projects  | `GET  /v1/projects/{project_id}/jobs/{job_id}` | Poll job state. 404 unknown job, 403 job belongs to another project. |
| projects  | `DELETE /v1/projects/{project_id}/jobs/{job_id}` | Request job cancellation (only `QUEUED`/`RUNNING`).                 |
| projects  | `GET  /v1/projects/{project_id}/files` | List indexed files (relative path, category, size) from Postgres.    |
| query     | `POST /v1/projects/{project_id}/query` | Build an LLM-ready prompt (no LLM call). Cache-aware.               |
| query     | `POST /v1/projects/{project_id}/query/stream` | Same pipeline, SSE `stage`/`result` events.                         |
| answer    | `POST /v1/projects/{project_id}/answer` | Run the full agent loop via `QueryService.ask` and return the final answer + citations. |
| answer    | `POST /v1/projects/{project_id}/answer/stream` | SSE events: `agent_start`, `agent_end`, final `answer` (or `error`). |
| health    | `GET /livez /readyz /version /metrics` | See §10. Not behind auth or rate limiting.                          |

Router files own exactly one concern each: `projects.py` is indexing and
file listing, `query.py` is prompt construction, `answer.py` is the
agent-driven answer path.

### `auth.py`, `api_key.py`, `api_key_store.py`, `project_auth.py` — identity

- `api_key.py` — the `APIKey` domain model (`name`, `status`,
  `allowed_projects`, `expires_at`, `last_used_at`), the `APIKeyStore`
  ABC, the `InMemoryAPIKeyStore`, and the two crypto helpers
  `hash_api_key` (blake3 over a hardcoded domain separator) and
  `verify_api_key` (constant-time `hmac.compare_digest`). Both stores
  persist the **digest**, never the plaintext bearer.
- `api_key_store.py` — `PostgresAPIKeyStore` (table `api_keys`, PK is the
  key hash) plus `seed_initial_keys`, which idempotently inserts the
  `SEMRAG_API_KEYS` configured at boot.
- `auth.py` — the `require_api_key` FastAPI dependency: reads the
  `Authorization: Bearer …` header, looks the key up by digest,
  re-verifies the digest in constant time, checks validity (revoked /
  expired), and touches `last_used_at`.
- `project_auth.py` — `get_project_authorizer()` returns a dependency
  that enforces per-project authorization via `is_allowed_for_project`.

### `dependencies.py` — the dependency container

Creates the process-wide singletons and exposes them to routes (see §5).

### `config.py` — API-side env vars

A lightweight `os.environ` reader for a handful of API values. Most
configuration lives in `core/settings.py`; this module fills the gap for
settings that pre-date the `Settings` model and are read directly by
routers (`SEMRAG_API_KEYS`, `SEMRAG_RATE_LIMIT_PER_MINUTE`,
`SEMRAG_QUERY_CACHE_TTL_SECONDS`, `SEMRAG_EMBEDDING_PROVIDER_DIR`).

### `query_cache.py` and `rate_limit.py` — correctness buffers

`QueryCache` and `RateLimiter` ABCs with in-memory and Redis
implementations (see §5 and the caching/rate-limit notes in
[developer_guide.md](../../../docs/developer_guide.md#64-caches-and-rate-limiter-api)).
Both feed `semrag_cache_*` / `semrag_rate_limit_*` Prometheus counters.

### `redis_client.py` — Redis factory chokepoint

The single place that turns `SEMRAG_REDIS_URL` into a client. Exposes
sync (`require_redis_client`) and async (`require_async_redis_client`)
factories; the `require_*` forms raise `RuntimeError` when Redis is
selected but no URL is configured — the no-silent-fallback rule.

### `schemas.py` — wire models

All request/response models (`IndexProjectRequest/Response`,
`ListFilesResponse`, `QueryRequest/Response`, `AnswerResponse`, …).
No business logic, only pydantic `Field` validation.

### `startup_validation.py` — refuse to boot misconfigured

`validate_runtime_config()` aggregates cross-cutting checks the pydantic
fields can't express (database URL scheme, embedding/LLM config
consistency, and the production rules: `database` API-key store, `database`
job store, `SEMRAG_API_KEYS` set). It also seeds `SEMRAG_API_KEYS` into the
active store on boot. Raises `RuntimeConfigurationError` listing every
problem at once.

---

## 4. Request lifecycle

A request on the protected surface travels through six layers, in this
order (inbound middleware order as verified in §7):

```text
client → MaxBodySize → SecurityHeaders → Tracing → Logging → Metrics
       → RequestID → GZip → CORS → TrustedHost → router → dependencies
       → service → response   (back out through the stack in reverse)
```

1. **Middleware** — the request first crosses the stack described in §7:
   body-size enforcement, security headers, tracing, logging, metrics,
   request-ID assignment, then compression, CORS, and Host allow-listing.
   On the way out the stack runs in reverse: access-log + audit events,
   trace context injection, and the `X-Request-ID` echo.
2. **Router + path match** — FastAPI resolves the route and runs the
   request body through the Pydantic schema. `RequestValidationError` is
   the normal result of a malformed body (422).
3. **Dependencies** — each route's `Depends(...)` graph is resolved:
   the DB session, the project authorizer (which itself depends on
   `require_api_key`), the rate limiter, and the service factories.
   See §5.
4. **Handler** — the route function performs its actual job. Query
   routes check the cache first, then run retrieval + prompt assembly;
   index routes call `JobManager.create_job`; answer routes call
   `QueryService.ask`.
5. **Response** — the handler returns a pydantic model; FastAPI
   serializes it. Any unhandled `HTTPException` or exception is converted
   to the standard envelope (§9) by the exception middleware.
6. **Outbound middleware** — on the way out, the stack records metrics,
   emits the structured access-log line + audit events, applies security
   headers, injects trace context, and echoes `X-Request-ID`.

There are two streaming paths (`/query/stream`, `/answer/stream`):
instead of returning one JSON document they yield Server-Sent Events from
a generator. See
[developer_guide.md §3.2](../../../docs/developer_guide.md#32-streaming).

---

## 5. Dependency injection

FastAPI `Depends(...)` is the only injection mechanism. The wiring lives
in `dependencies.py`; routes declare their needs as `Depends(...)`
arguments and never construct their own collaborators.

**Module-scope singletons** (created at import time):

| Singleton            | Built by                            | Reset for tests                       |
| -------------------- | ----------------------------------- | ------------------------------------- |
| SQLAlchemy engine    | `build_engine(settings)`            | (disposed in `shutdown_resources`)    |
| Session factory      | `build_session_factory(...)`        | —                                     |
| Job store            | `InMemoryJobStore` / `PostgresJobStore` | `reset_job_store_for_tests()`     |
| Job manager          | `JobManager(store)`                 | `reset_job_store_for_tests()`         |
| Job executor         | `ExecutorFactory.create_executor`   | lazily rebuilt after shutdown         |
| Query cache          | `TTLQueryCache` / `RedisQueryCache` | `reset_shared_cache()`                |
| API key store        | `InMemoryAPIKeyStore` / `Postgres`  | `reset_api_key_store_for_tests()`     |

**Per-request dependencies:**

- `get_db_session` — a generator: **yield → commit on success → rollback
  on exception → always close**. This is the only correct way to hold a
  session for the lifetime of a request.
- `get_query_service` — builds a `QueryService(db, embedding_provider,
  llm, settings)` per request so setting-dependent providers are read
  fresh; the embedding/LLM providers come from `build_embedding_provider`
  and `build_llm`.
- `require_api_key` (from `auth.py`) — returns the validated `APIKey`.
- `get_project_authorizer()` (a **factory**) — returns a closure
  depending on `require_api_key` and enforcing the project allowlist.
  Routes call `Depends(get_project_authorizer())`.
- `enforce_rate_limit` — depends on `require_api_key` + the rate limiter
  singleton; raises 429 when the key is over its window.
- `get_job_manager`, `get_job_executor`, `get_session_factory`,
  `get_settings` — plain accessors for routes.

**Testing seam.** Every singleton has a documented `reset_*_for_tests()`
helper or a lazily re-created path, because tests monkeypatch settings
between cases and need the next dependency resolution to rebuild. In
production each reset path is never exercised.

The full DI + singleton rationale (including *why* singletons live at
module scope and *why* factories are the only `Settings` readers) is in
[developer_guide.md §5](../../../docs/developer_guide.md#5-dependency-injection).

---

## 6. Authentication & authorization

### Authentication — `require_api_key`

Every protected route depends, directly or via `get_project_authorizer()`,
on `require_api_key`. It expects `Authorization: Bearer <token>`:

1. Missing or non-`Bearer` header → **401** `HTTP_401`.
2. Look up the store **by digest**: the bearer is hashed once with
   `hash_api_key` (blake3 over domain separator), then the store matches
   the digest (in-memory dict key, or SQL `key_hash =`).
3. Defense-in-depth: `verify_api_key` re-compares the digest with
   `hmac.compare_digest` — constant-time regardless of where the digest
   diverges.
4. `APIKey.is_valid()` rejects revoked or expired keys → **401**.
5. Updates `last_used_at`.

**Key storage.** Both stores keep only the blake3 digest (a leaked DB
dump yields no usable tokens). `SEMRAG_API_KEY_STORE` selects
`memory` (single-replica, per-process) or `database` (persistent,
production-safe). Production **refuses to boot** with the memory store
(see `startup_validation.py`). Boot-time seeding inserts the
`SEMRAG_API_KEYS` values into the active store, idempotently.

### Authorization — `get_project_authorizer()`

This *factory* returns a dependency that reads `project_id` from the path
and calls `APIKey.is_allowed_for_project(project_id)`:

- `allowed_projects` empty → access to all projects.
- Otherwise the key is restricted to the listed projects; any other
  project gets **403** `HTTP_403`.

Both **401** (authentication failure) and **403** (authorization failure)
feed audit-log events via `LoggingMiddleware` (`semrag.audit.log_audit_event`).

### Rate limiting — `enforce_rate_limit`

Runs after auth, keyed by the API key digest. `MemoryRateLimiter` is a
sliding window per process; `RedisRateLimiter` is a sliding window backed
by a Redis sorted set plus a Lua script for an atomic check-and-record.
Over the window → **429**. Every accept/reject increments the
`semrag_rate_limit_*` counters. See
[developer_guide.md §8.3](../../../docs/developer_guide.md#83-rate-limiting).

---

## 7. Middleware pipeline

`setup_middleware(app)` in `middleware.py` registers everything. Starlette
composes middleware LIFO in FastAPI/Starlette 1.x: **the last added runs
first on the way in**. Configuration-gated pieces (`TrustedHost`, `CORS`,
`GZip`, `Metrics`) are added only when the corresponding setting is set.

> **Verified runtime order.** The `setup_middleware` docstring describes
> an outermost-first list that does **not** match observed behavior in the
> installed Starlette 1.3.1 / FastAPI 0.141.1. The actual inbound order
> (verified with a probe route) is shown below and in §4; trust it, not
> the comment, when reasoning about ordering dependencies.

Inbound (outermost → innermost), **in the order they actually run**:

```text
MaxBodySize → SecurityHeaders → Tracing → Logging → Metrics → RequestID
→ GZip → CORS → TrustedHost → (Starlette Exception/AsyncExitStack) → app
```

Consequences that follow from this order:

- **`MaxBodySize` is the outermost application middleware** — a request
  with an oversized `Content-Length` is rejected with **413** before any
  security header, trace, or log work happens.
- **`RequestID` runs deep** — it generates/echoes `X-Request-ID`, so every
  inner and outer layer can reference `request.state.request_id`.
- **`Tracing` wraps `Logging` and `Metrics`** — a single span covers both
  the metric recording and the structured log emission; `Metrics` measures
  end-to-end handler latency, and `Logging` appends the access-log line and
  fires audit events based on the final status code.

| Middleware class        | Responsibility                                                            | Gate             |
| ----------------------- | ------------------------------------------------------------------------- | ---------------- |
| `MaxBodySizeMiddleware` | Reject bodies > `SEMRAG_MAX_BODY_SIZE` (413).                              | always           |
| `SecurityHeadersMiddleware` | Applies `SEMRAG_SECURITY_HEADERS` to every response.                    | always           |
| `TracingMiddleware`     | Opens one OTel span per request; propagates `traceparent` downstream; no-op when tracing off. | always |
| `LoggingMiddleware`     | Structured access log (request id, method, path, status, latency) + audit events. | always     |
| `MetricsMiddleware`     | Records `semrag_http_*` latency/size/status/exceptions on the route template. | `metrics_enabled` |
| `RequestIDMiddleware`   | Generates or reuses an `X-Request-ID`, echoes it, exposes on `request.state`. | always           |
| `GZipMiddleware`        | Compresses responses ≥ 1000 bytes.                                        | `enable_gzip`    |
| `CORSMiddleware`        | CORS from `SEMRAG_CORS_ORIGINS`.                                          | `cors_origins`   |
| `TrustedHostMiddleware` | Allow-lists `Host` headers from `SEMRAG_TRUSTED_HOSTS`.                   | `trusted_hosts`  |

Labeling note (`_route_template`): metrics/tracing always use the **route
template** (`/v1/projects/{project_id}/query`), never the raw URL path —
otherwise project IDs explode the Prometheus cardinality.

Settings that affect the pipeline: `trusted_hosts`, `cors_origins`,
`enable_gzip`, `max_body_size`, `security_headers`, `metrics_enabled`,
`tracing_enabled`, `trace_sample_ratio`. Definitions live in
`core/settings.py`.

---

## 8. Caching, rate limiting, and observability

These three cross-cutting concerns are wired as dependencies, not as
handler logic. The rules that keep them honest:

1. **Always route through the dependency**, never construct a limiter or
   cache inline in a handler. `enforce_rate_limit` and `get_query_cache`
   are the only seams.
2. **Cache before heavy work.** Query routes hash the request into a
   `cache_key` and short-circuit retrieval on a hit
   (`QueryResponse.from_cache=True` is set for the hit response). TTL is
   `SEMRAG_QUERY_CACHE_TTL_SECONDS` (default 300s) — short on purpose:
   re-indexing can change the vector space for the tfidf provider.
3. **Invalidate precisely.** `QueryCache.invalidate_project(project_id)`
   clears only that project's entries (via a per-project key index),
   and the index route invalidates **at enqueue time** so concurrent
   queries miss the cache while a re-index is in flight.
4. **Expose decisions.** Cache hits/misses/evictions and rate-limit
   accept/rejects all increment Prometheus counters so a Grafana panel
   can graph cache-hit rate and limiter saturation.

These are the honest-scope caveats from the module docstrings: the
in-memory cache and limiter are correct **per process**, not across
replicas; the Redis variants are what you deploy when you need the
distributed property, and they fail fast rather than silently falling
back.

---

## 9. Error handling

`exception_handler.py` registers three handlers; every error body uses one
envelope so clients can parse defensively:

```json
{
  "error": {
    "code": "HTTP_404",
    "message": "…",
    "request_id": "<uuid or null>",
    "details": {}
  }
}
```

| Source                      | Status | `code`             | `details`                 |
| --------------------------- | ------ | ------------------ | ------------------------- |
| `HTTPException` (any router / dependency) | the exception's status | `HTTP_<status>` | `{}` |
| `RequestValidationError`    | 422    | `VALIDATION_ERROR` | `{"errors": [pydantic errors…]}` |
| any other `Exception`       | 500    | `INTERNAL_SERVER_ERROR` | `{}` (generic message, no stack trace ever leaks) |

Design points:

- `request_id` is threaded from `request.state.request_id` so a client can
  correlate an error with the access-log line.
- The generic 500 handler returns a fixed message in every environment;
  the real traceback goes to the logs via the logging configuration.
- The **rule** (from
  [developer_guide.md §11](../../../docs/developer_guide.md#11-error-handling)):
  catch only what you can recover from. Let everything else reach the
  generic handler so the envelope stays consistent and the log gets the
  full traceback.

---

## 10. Health / metrics endpoints

All defined in `health.py` (plus the `/health` alias in `app.py`), all
**unauthenticated** — probes and scrapers are infrastructure-trusted and
must not need a key:

| Endpoint   | Contract | Body (when available) |
| ---------- | -------- | --------------------- |
| `GET /livez`  | Process liveness only — **no** DB/LLM/embedding I/O. Always 200 if Python runs. | `{"status": "alive"}` |
| `GET /version` | Build metadata, cached at import, no I/O. | app version (package metadata, else `0.0.0`), git commit, build timestamp, python version, environment |
| `GET /readyz`  | Readiness: every dependency check passes → 200; any failure → 503. Also flips `semrag_health_up{component}` gauges. | `{"status": "ready"/"not_ready", "checks": {…}, "version", "timestamp"}` |
| `GET /metrics` | Prometheus text exposition (`text/plain; version=0.0.4`); 404 when `SEMRAG_METRICS_ENABLED=false`. | rendered registry |

`/readyz` runs one check per component (`_CHECK_FUNCS`), re-read live so
tests can monkeypatch them:

| Check           | Approves when                                                        |
| --------------- | -------------------------------------------------------------------- |
| `database`      | `SELECT 1` **and** the `vector` pgvector extension is installed, via the shared engine. |
| `llm`           | Provider is `mock`, or base URL + model are configured (no live probe — probes fire constantly). |
| `embedding`     | Provider is `tfidf`, or base URL + model + positive dimension are configured. |
| `job_executor`  | Executor exists and has not been shut down (`_shutdown` unset).      |
| `cache`         | Reports configured backend (`redis` → "configured"; no connection opened). |
| `rate_limiter`  | Reports configured backend (same reasoning as cache).                |

Probe semantics to remember (also documented for operators in
[../operations/README.md](../../../docs/operations/README.md)): use `/livez`
for kubelet liveness, `/readyz` to decide traffic, `/metrics` for
Prometheus, and keep in mind that `/metrics` is intentionally
unauthenticated — lock it down at the network layer.

---

## 11. Security model

Layered, defence-in-depth:

1. **Transport/host hardening** — `TrustedHostMiddleware` (Host
   allow-list), CORS allow-list, security headers on every response
   (`SEMRAG_SECURITY_HEADERS`), `MaxBodySizeMiddleware` (413 on oversized
   bodies).
2. **Authentication** — Bearer API keys stored as blake3 digests only,
   constant-time verification, expiry/revocation via `APIKey.is_valid`.
3. **Authorization** — per-project allowlist forced by
   `get_project_authorizer()` (403 when a key is not allowed for the
   project).
4. **Production defaults that refuse to boot unsafe** —
   `startup_validation.py` blocks production + `SEMRAG_API_KEY_STORE`
   unset/in-memory, production + `SEMRAG_JOB_STORE_BACKEND=memory`, and
   production without `SEMRAG_API_KEYS`. `dev-local-key` is rejected in
   production by the `Settings` validator.
5. **Rate limiting** — per-key sliding window before heavy work.
6. **Secrets hygiene** — API keys are hashed with a domain separator; the
   redis clients are factory-created from `SEMRAG_REDIS_URL`.

Not enforced (by design): query-cache/rate-limiter network isolation
(they are in-process vectors only; Redis variants are not TLS-forced
here — terminate TLS at the edge), per-route rate limits, and a full
audit trail beyond the access-log events. See
[developer_guide.md §8.6](../../../docs/developer_guide.md#86-what-semrag-does-not-enforce).

---

## 12. Configuration

Settings for this package live in two places. Prefer `core/settings.py`
for anything new; `api/config.py` remains for the API-specific legacy
readers.

| Setting (env var)                | Where   | Default                       | Used by                                |
| -------------------------------- | ------- | ----------------------------- | -------------------------------------- |
| `SEMRAG_API_KEYS`                | `config.py`/`Settings` | `dev-local-key` (rejected in prod) | key seeding + auth                 |
| `SEMRAG_API_KEY_STORE`           | `Settings` | `database`                   | store selection (`api_key.py`)          |
| `SEMRAG_RATE_LIMIT_PER_MINUTE`   | `config.py`/`Settings` | `60`                    | `rate_limit.py`                         |
| `SEMRAG_RATE_LIMIT_BACKEND`      | `Settings` | `memory`                   | limiter selection                       |
| `SEMRAG_QUERY_CACHE_TTL_SECONDS` | `config.py`/`Settings` | `300`                | `query_cache.py`                        |
| `SEMRAG_QUERY_CACHE_BACKEND`     | `Settings` | `memory`                   | cache selection                         |
| `SEMRAG_REDIS_URL`               | `Settings` | `""`                     | `redis_client.py` (empty → raise with redis backend) |
| `SEMRAG_EMBEDDING_PROVIDER_DIR`  | `config.py`/`Settings` | `/tmp/semrag_embedding_providers` | tfidf fit directory                |
| `SEMRAG_TRUSTED_HOSTS`, `SEMRAG_CORS_ORIGINS`, `SEMRAG_MAX_BODY_SIZE`, `SEMRAG_ENABLE_GZIP`, `SEMRAG_SECURITY_HEADERS` | `Settings` | restrictive defaults | `middleware.py` |
| `SEMRAG_METRICS_ENABLED`, `SEMRAG_TRACING_ENABLED`, `SEMRAG_TRACE_SAMPLE_RATIO`, `SEMRAG_LOG_LEVEL`, `SEMRAG_LOG_FORMAT` | `Settings` | … | metrics/tracing/logging, lifespan     |
| `SEMRAG_API_RUNS_WORKER`         | `Settings` | `true`                    | `app.py` lifespan                      |
| `SEMRAG_JOB_STORE_BACKEND`       | `Settings` | `database`                | `dependencies.py` job store            |

> Keep new API knobs in `core/settings.py` and read them via
> `get_settings()`. `api/config.py` is legacy; don't grow it.

---

## 13. How to add

### 13.1 A new endpoint

1. Pick the router by concern: indexing → `routers/projects.py`, prompt
   construction → `routers/query.py`, answer-generation → `routers/answer.py`.
   A genuinely new resource gets a new file under `routers/`.
2. Add/update the request + response models in `schemas.py` (a response
   model is what enables `docs`-compatible OpenAPI output).
3. Attach the dependency chain: `Depends(get_project_authorizer())`,
   `Depends(enforce_rate_limit)`, and a DB session
   (`Depends(get_db_session)`) if you read/write Postgres.
4. Do the real work in `services`, map the result back to the response
   model, never raise a domain error (`HTTPException` only). Keep the
   handler a thin translation.
5. Register/keep the router in `app.py`.
6. Add tests under `tests/unit/api/` using `TestClient` + the existing
   conftest fixtures.

### 13.2 Middleware

1. Subclass `BaseHTTPMiddleware` (or write a pure ASGI class) in
   `middleware.py`, mirroring the existing classes.
2. Register it in `setup_middleware`. Remember the **LIFO composition**:
   add it so your middleware lands where it must run in the inbound
   order (late in the add sequence → early inbound; early add → inner).
3. If it's configurable, gate it on a `Settings` field and document the
   env var (§12).

### 13.3 A dependency

1. Define the provider function in `dependencies.py` (or a dedicated
   module) — with `Depends(...)` parameters as needed.
2. Export it and use it in routes: `Depends(your_dependency)`.
3. If it owns mutable state, add a `reset_your_dependency_for_tests()`
   hook and document why tests need it.
4. A provider that needs `Settings` must read it inside the function (or
   via `get_settings()`), not at import, unless you want a singleton.

### 13.4 An authentication method

The current model is one header format (Bearer). To add another:

1. Implement a dependency that returns an authenticated identity
   (mirror `require_api_key`).
2. Keep the contract: raise `HTTPException(401)` for missing/invalid
   credentials, authorize per-project with a `403`-raising gate.
3. Prefer composing `require_api_key` inside a wrapper over replacing the
   whole chain, so rate limiting and auditing still see the credential.
4. If you add a credential store, implement the `APIKeyStore` ABC and
   register it in `get_api_key_store` (e.g. add the Redis store that the
   current code warns about).

### 13.5 A response model

1. Define a pydantic `BaseModel` subclass in `schemas.py`.
2. Use it as the route's `response_model=…`, and mirror it for the
   streaming variant's final event payload if the path streams.
3. Keep `schemas.py` free of logic — validation only.

---

## 14. Common debugging paths

| Symptom                                                  | Start here                                                            |
| -------------------------------------------------------- | --------------------------------------------------------------------- |
| **401 on every request**                                 | `SEMRAG_API_KEYS` not seeded into the active store. Check store backend (`SEMRAG_API_KEY_STORE`), see `seed_initial_keys`, restart. |
| **403 for a valid key**                                  | The key's `allowed_projects` allowlist; check `project_auth.py` + the `api_keys` row. |
| **429 unexpectedly**                                     | `SEMRAG_RATE_LIMIT_PER_MINUTE` / limiter backend; check `rate_limit.py` and metrics. |
| **413 on a legitimate request**                          | `SEMRAG_MAX_BODY_SIZE` (5 MB default) — raise it or stream.          |
| **Correlating a user error with logs**                   | Take `X-Request-ID` from the response, grep the access log for `request_id=…`. |
| **`/readyz` returns 503**                                | Read the `checks` object; the failing component is named (`database`, `llm`, `embedding`, `job_executor`, `cache`, `rate_limiter`). |
| **Startup refuses to boot**                              | `RuntimeConfigurationError` message lists every problem; fix all before restarting. |
| **Middleware behavior surprises you**                    | Remember the inbound order in §7 — notably that `MaxBodySize` and `SecurityHeaders` run outside `RequestID`/tracing. |
| **Cache serving stale results after re-index**           | `invalidate_project` runs at enqueue. If you suspect drift, check `query_cache.py` and the project-key index. |
| **Streaming endpoint returns HTTP 200 but no events**    | Check the generator yields and the client handles `text/event-stream`; see `routers/query.py` and `routers/answer.py`. |
| **Missing metrics on `/metrics`**                        | Ensure `SEMRAG_METRICS_ENABLED=true`; metrics labels use route templates, so look for the template string not the URL. |

For the query/agent internals beyond the HTTP seam, see
[developer_guide.md §14](../../../docs/developer_guide.md#14-debugging-workflow).

---

## 15. Common mistakes

- **Trusting the `setup_middleware` docstring's ordering.** The comment
  describes an outermost-first list that does not match the observed
  LIFO runtime order in this dependency set (Starlette 1.3.1 / FastAPI
  0.141.1). Reason from §7's verified order.
- **Constructing a QueryService/embedding provider inside a handler.**
  Use `Depends(get_query_service)` / factories; handlers must be
  translation only.
- **Holding a session beyond the request.** Never stash
  `get_db_session`'s session on a singleton; commit/rollback/close are
  handled by the generator.
- **Swallowing exceptions with a generic `try/except` in a router.** The
  centralized handlers already map to the standard envelope; a broad
  catch hides the traceback from `general_exception_handler`.
- **Returning domain objects as a response without a `response_model`.**
  Define the schema so OpenAPI, validation, and error envelopes stay
  consistent.
- **Growing `api/config.py`.** New settings belong in `core/settings.py`
  (`SEMRAG_*` env vars) and are read via `get_settings()`.
- **Forgetting the `redis` no-silent-fallback rule.** Selecting a Redis
  backend without `SEMRAG_REDIS_URL` must raise (as the `require_*`
  factories do), never silently downgrade to memory.
- **Expecting `/metrics` or probes to require auth.** They are
  intentionally unauthenticated; secure at the network layer.
- **Adding auth/rate-limit only to *some* routes.** If the new endpoint
  is on the protected surface, include the full dependency chain in the
  same PR.
- **Ignoring cardinality on metrics.** Never label on raw project IDs —
  use the route template (see `_route_template`).

---

## 16. Related documentation

- [src/semrag README](../README.md) — package boundaries, dependency
  rules, extension points at the whole-package level.
- [developer_guide.md](../../../docs/developer_guide.md) — end-to-end
  lifecycle, DI philosophy, security model, error handling, debugging.
- [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md) — file-by-file
  ownership and "where do I change X?" tables.
- [Operations runbooks](../../../docs/operations/README.md) — on-call
  material keyed to `/readyz` checks and metrics.
- [RUN_API.md](../../../RUN_API.md) — local launch recipe and curl examples.
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — long-form architecture and
  data-flow diagrams.
- [docs/README.md](../../../docs/README.md) — the documentation index.