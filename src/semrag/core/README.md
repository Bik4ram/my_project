# `src/semrag/core` — Cross-Cutting Primitives

> **The single package every other subpackage depends on, and the
> single package that depends on nothing else in this codebase.**
> This README explains why `core` exists, what belongs in it, what
> must never be added to it, and how each module is meant to be used.
> It is `core`-specific; the end-to-end observability walkthrough lives
> in [developer_guide.md §9](../../../docs/developer_guide.md#9-observability)
> and the package map is in
> [src/semrag/README.md](../README.md).

Intended audience: **Developers** and **Maintainers** who touch
anything that crosses subpackage boundaries — settings, logging,
metrics, tracing, secret hygiene, debug artifacts.

---

## 1. Purpose of the package

`core` exists to hold **cross-cutting primitives** — utilities used by
more than one subpackage that are not domain-specific. Without this
package, every subpackage would re-implement `Settings`, every module
would write its own logger, and every metric label would diverge
silently.

The single rule that defines the package:

> **`core` imports from nothing in `semrag`. Every other subpackage
> imports from `core`.**

This rule is what makes the package composable. `ingestion/`,
`retrieval/`, `reasoning/`, `storage/`, `services/`, `jobs/`, and
`api/` all import `Settings`, `get_logger`, `Metrics.timer`,
`redact_payload`, `tracer(...)` — and `core` does not import any of
them. The dependency arrow points one way.

What goes in `core`:

- `Settings` and `get_settings()` — the only configuration schema.
- `configure_logging`, `get_logger`, `Metrics`, `stage(...)`,
  `redact_payload` — the only observability surface.
- `DebugArtifactWriter` — the only per-run artifact store.
- `tracing.configure_tracing`, `tracer`, `inject_context`,
  `extract_context` — the only OpenTelemetry seam.
- `QueryTrace` — the only per-query trace accumulator.
- `performance.bulk_throughput` / `ingestion_stage_timer` — a thin
  veneer over `Metrics` for ingestion-stage reporting.

What must never be added to `core`:

- Domain types from `ingestion/`, `retrieval/`, `reasoning/`,
  `storage/`, or `services/`.
- HTTP / FastAPI primitives — those belong in `api/`.
- Anything that imports from another SemRAG subpackage.
- Configuration reading that bypasses `Settings` (the legacy
  `env_flag()` helper in `settings.py` is the documented exception,
  used only by bootstrap code; everything else goes through
  `Settings`).
- Module-level singletons that hold runtime state (other than the
  `Settings` lru-cache singleton and the Prometheus `REGISTRY`,
  which are both process-wide fixtures).

If a utility feels like it belongs in `core` but is only used by one
subpackage, it stays in that subpackage. The criterion is **used by
≥2 subpackages**. See [Boundaries](#13-architectural-boundaries).

---

## 2. Package structure

```
src/semrag/core/
├── __init__.py                  one-line module docstring
├── settings.py                  Settings (pydantic-settings),
│                                 get_settings(), reset_settings_cache(),
│                                 env_flag() bootstrap helper
├── observability.py             configure_logging(), get_logger(),
│                                 Metrics + count/observe/timer, stage(),
│                                 StageStats, TRACE level, _JsonFormatter
│                                 and _HumanFormatter
├── metrics.py                   Prometheus REGISTRY + the project's
│                                 semrag_* Counter/Histogram/Gauge
│                                 inventory + render_metrics(),
│                                 reset_registry(), and the per-helper
│                                 convenience functions (observe_http,
│                                 inc_cache_hit, …)
├── tracing.py                   configure_tracing(settings), tracer(),
│                                 current_trace_id_hex(), current_span_id_hex(),
│                                 inject_context() / extract_context(),
│                                 no-op stand-ins, reset_tracing_state()
├── redaction.py                 redact_value(), redact_payload(),
│                                 the _SENSITIVE_KEY_MARKERS / inline
│                                 patterns
├── debug_artifacts.py           DebugArtifactWriter, new_run_id(),
│                                 per-run directory layout
├── query_trace.py               QueryTrace + QueryTraceStage pydantic
│                                 models (22-stage shape)
└── performance.py               bulk_throughput(),
│                                 ingestion_stage_timer()
```

Eight files, each one a single concern. Every public function is
documented in its module; this README describes how they compose and
which is responsible for what.

---

## 3. Responsibilities of every module

### `settings.py` — the configuration schema

`Settings` is a `pydantic-settings.BaseSettings` with the
`SEMRAG_` env-var prefix, `.env` support, and a `model_validator`
that enforces combinations the field types can't express (notably,
`dev-local-key` is rejected when `environment == "production"`).

`Settings` is intentionally a `BaseSettings` (mutable singleton)
rather than `@dataclass(frozen=True)`. Some fields are derived at
construction (e.g. `embedding_dimension` could default from the
provider) and tests need `settings.replace(...)` for hot-swap. The
canonical access pattern is `get_settings()`; the canonical reset
for tests is `reset_settings_cache()` (clears the `lru_cache` and
forces a fresh `Settings()`).

`env_flag(name, default=False)` is the single bootstrap-only helper
that reads `os.environ` directly — it exists for the small handful of
modules that have not migrated to `Settings`. The module docstring is
explicit: **"preferred path is always to add the field to Settings"**.

Field inventory and semantics live in the source (see `settings.py`
for the full list); the developer guide enumerates the three buckets
in §10 (Database / Retrieval / Operations).

### `observability.py` — logging + per-stage metrics + redaction seam

Three concerns in one module, deliberately:

1. **Stdlib logging configuration.** `configure_logging(settings)`
   installs the formatter + handler exactly once. Two formatters —
   `_JsonFormatter` (one-line stable-shape JSON per record) and
   `_HumanFormatter` (colored, time-stamped) — selected by
   `Settings.log_format`. The `TRACE` level (`5`) is registered with
   stdlib logging so `Settings.log_level = "TRACE"` and
   `logger.log(TRACE_LEVEL, ...)` both work without per-call integer
   arithmetic.

2. **`Metrics` and friends.** Three primitives:

   - `Metrics.count(name, value=1, **tags)` — increment a counter.
   - `Metrics.observe(name, value, **tags)` — observe a histogram
     sample.
   - `Metrics.timer(name, **tags)` (context manager) — record elapsed
     milliseconds on exit. Yields a mutable dict the caller can
     populate with extra tags (e.g. `hit_count`).

   Each call emits one INFO-level log record on the
   `semrag.metrics` logger with a stable prefix (`metric=... value=...
   tags=...`) so a log shipper can split metric events from ordinary
   log lines without parsing free-form text. The convenience aliases
   `count(...)`, `observe(...)`, `timer(...)` are short-name wrappers
   for the same methods.

3. **Per-stage structured logging.** `stage(name, *, logger, level=...,
   **tags)` is the context manager every pipeline uses. It emits a
   `stage=... event=start` line on entry, a `stage=... event=finish
   elapsed_ms=...` line on success, and `stage=... event=finish
   status=error error=...` on exception. Tags propagate to both
   events so a log pipeline can filter on `project_id` without
   joining two records. `StageStats` is the mutable bag the caller
   populates inside the block (`stats["chunk_count"] = 12`); the
   finish line emits them as structured fields.

`get_logger(name)` is the standard idiom — `logger = get_logger(__name__)`.
Configuration happens at startup; the logger itself is always usable.

`_JsonFormatter` automatically runs every payload through
`redact_payload` (see `redaction.py`) so secrets never leak into a
JSON log line. It also stamps every record with the active trace id
and span id (`trace_id`, `span_id`); the call uses
`current_trace_id_hex()` / `current_span_id_hex()` from `tracing.py`,
which fall back to zeroed W3C-format sentinels when no span is
active.

### `metrics.py` — Prometheus registry + per-subsystem inventory

A process-global `CollectorRegistry` is constructed once at import
time (`REGISTRY = _new_registry()`). All metric objects (counters,
histograms, gauges) are also constructed at import time and reused —
constructing a `Counter` per request is a measurable cost in a hot
path.

Inventory, verified against `metrics.py`:

| Subsystem   | Metric                                                                  | Labels                          |
| ----------- | ----------------------------------------------------------------------- | ------------------------------- |
| HTTP        | `semrag_http_requests_total` (Counter)                                  | method, path, status            |
| HTTP        | `semrag_http_request_duration_seconds` (Histogram)                      | method, path, status            |
| HTTP        | `semrag_http_request_size_bytes` (Histogram)                           | method, path                    |
| HTTP        | `semrag_http_response_size_bytes` (Histogram)                          | method, path                    |
| HTTP        | `semrag_http_exceptions_total` (Counter)                                | method, path, exception         |
| RAG         | `semrag_rag_retrieval_duration_seconds` (Histogram)                     | stage                           |
| RAG         | `semrag_rag_rerank_duration_seconds` (Histogram)                        | (none)                          |
| RAG         | `semrag_rag_embedding_duration_seconds` (Histogram)                     | provider                        |
| RAG         | `semrag_rag_llm_duration_seconds` (Histogram)                           | provider, model                 |
| RAG         | `semrag_rag_tool_duration_seconds` (Histogram)                          | tool                            |
| Indexing    | `semrag_indexing_duration_seconds` (Histogram)                          | project_id                      |
| Indexing    | `semrag_indexing_files_processed_total` (Counter)                       | project_id                      |
| Indexing    | `semrag_indexing_chunks_generated_total` (Counter)                      | project_id                      |
| Indexing    | `semrag_indexing_chunks_embedded_total` (Counter)                       | project_id                      |
| Indexing    | `semrag_indexing_jobs_queued_total` (Counter)                           | project_id                      |
| Indexing    | `semrag_indexing_jobs_completed_total` (Counter)                        | project_id                      |
| Indexing    | `semrag_indexing_jobs_failed_total` (Counter)                           | project_id                      |
| Database    | `semrag_db_pool_connections` (Gauge)                                    | state (active/idle/overflow)    |
| Database    | `semrag_db_query_duration_seconds` (Histogram)                          | op                              |
| Database    | `semrag_db_transactions_total` (Counter)                                | op                              |
| Cache       | `semrag_cache_hits_total` (Counter)                                     | backend                         |
| Cache       | `semrag_cache_misses_total` (Counter)                                   | backend                         |
| Cache       | `semrag_cache_evictions_total` (Counter)                                | backend                         |
| Rate-limit  | `semrag_rate_limit_accepted_total` (Counter)                            | (none)                          |
| Rate-limit  | `semrag_rate_limit_rejected_total` (Counter)                            | (none)                          |
| Uptime      | `semrag_uptime_seconds` (Gauge)                                         | (none)                          |
| Health      | `semrag_health_up` (Gauge)                                              | component                       |

Histogram buckets are deliberate: `_LATENCY_BUCKETS` covers 5 ms to
120 s (LLM calls cap at 90 s; embeddings at 300 s); `_FAST_BUCKETS`
covers 1 ms to 5 s (in-process stages like vector search, fusion);
`_SIZE_BUCKETS` covers 100 B to 5 MB (`Settings.max_body_size`).

Cardinality is bounded by design — `path` is the FastAPI route
template (`/v1/projects/{project_id}/query`), not the rendered URL;
`backend` is two values; `tool` is the `ToolRegistry`'s surface
(read_file / list_files / search_codebase).

Two public helpers close the loop:

- `render_metrics() -> (bytes, content_type)` — pair-ready for the
  `/metrics` endpoint. Hands straight to `Response(content=...,
  media_type=...)`.
- `reset_registry()` — **tests only**. Tears down `REGISTRY`, rebuilds
  it, and rebinds every metric via `_rebind_metrics()` (counters /
  histograms hold an immutable reference to the registry they were
  constructed with, so a bare `REGISTRY = ...` swap would leave the
  old objects pointing at the old registry). Production never calls
  this — the registry lives for the lifetime of the process.

### `tracing.py` — OpenTelemetry facade

A thin wrapper around the OTel SDK so the application code is
decoupled from the OTel SDK version, and so tracing can be disabled
(tests, offline dev) without every import failing.

Public surface:

- `configure_tracing(settings)` — called once at startup. Idempotent
  (the `_configured` guard short-circuits subsequent calls). When
  `Settings.tracing_enabled` is `False` or the SDK is not installed,
  falls back to a no-op tracer. The exporter family is picked from
  the `OTLP_ENDPOINT` URL scheme: `http://` or `https://` → HTTP /
  protobuf (port 4318); `grpc://` or no scheme → gRPC (port 4317).
  Sampler is built from `Settings.trace_sample_ratio` —
  `0.0 → ALWAYS_OFF`, `1.0 → ALWAYS_ON`, anything in between
  → `TraceIdRatioBased`.
- `tracer(name)` — returns a tracer whose
  `start_as_current_span(name, **attrs)` is a context manager. The
  returned object is a real OTel `Tracer` when configured, otherwise
  `_NoopTracer`. Call sites do not branch on the flag.
- `current_trace_id_hex()` / `current_span_id_hex()` — W3C-format
  hex strings (32 / 16 chars respectively) used by `_JsonFormatter`
  so every log record carries the active trace id. Returns zeroed
  sentinels when no span is active or tracing is disabled.
- `inject_context(headers)` / `extract_context(headers)` — propagate
  W3C trace context across process boundaries (HTTP, future DB
  driver instrumentation).
- `reset_tracing_state()` — **tests only**. Clears `_configured` so
  the next `configure_tracing(...)` runs again.

OpenTelemetry imports are deferred to `configure_tracing` so tests
that don't enable tracing don't pull grpc / protobuf.

### `redaction.py` — the secret chokepoint

Every redaction in the codebase goes through `redact_value(key,
value)` or `redact_payload(payload)`. Two rules:

1. **Sensitive key matching** — case-insensitive substring against a
   list of markers (`api_key`, `apikey`, `authorization`, `bearer`,
   `token`, `password`, `private_key`, `secret`, `connection_string`,
   `session_token`, `db_password`, …). On match, the value is replaced
   with the literal string `"***"` regardless of type.
2. **Inline secret scrubbing** — `Bearer` tokens, AWS access-key
   ids (`AKIA…`), JWTs (`eyJ…`), and password components of
   `user:password@host` URLs are replaced inside string values
   regardless of the key.

Deliberately **not** redacted: `request_id`, `trace_id`, `project_id`,
`job_id`, numeric fields, URLs without credentials. The redactor is
conservative — when in doubt, it redacts.

`_payload` is iterated shallowly: nested dicts are inspected by key
but not walked recursively. This is intentional — the
`LogRecord.__dict__` pipeline flattens before `redact_payload` is
called, and a deeply recursive redactor is hard to audit.

### `debug_artifacts.py` — per-run artifact bundle

`DebugArtifactWriter` is the single per-run debug-bundle writer.
Activated only when `Settings.log_level` is `DEBUG` or `TRACE`; at
INFO+ every method is a no-op (`is_active` returns `False` and the
writers return immediately). This is what makes the bundle free at
production log levels.

API:

- `start_run(run_id)` — allocates the per-run subdirectory tree
  (`scan/`, `parse/`, `ir/`, `chunks/`, `embeddings/`,
  `knowledge_graph/`, `retrieval/`, `prompt/`, `llm/`, `timings/`,
  `query_traces/`). Idempotent for the same `run_id`.
- `write_json(category, name, payload)` — generic JSON writer.
- `write_text(category, name, content)` — raw text writer.
- `write_parse(rel_path, payload)` — one-per-source-file under
  `parse/` (the relative path is flattened — `src/foo.py` becomes
  `src_foo.py.parse.json`).
- `write_query_trace(payload, compress=False)` — full
  `QueryTrace.model_dump()` under `query_traces/`. With
  `compress=True` writes `query_trace.json.gz`.
- `run_completed()` — retention prune. Sort run directories by name
  (run ids are sortable timestamps), delete all but the last
  `Settings.debug_artifact_retention`. `retention=0` skips pruning
  (operator-managed).

Run ids are supplied by the caller (typically `IngestionService.index
(...)` and `QueryService.ask(...)`). The writer never invents one.
`new_run_id()` is the canonical generator: a millisecond timestamp
plus a 3-byte hex suffix, e.g. `1700000000000-a3b7`. The format is
sortable for retention and unique enough for daily run volumes.

The full per-stage file layout is documented in
[services/README.md §9](../services/README.md#9-debugging) and in
[developer_guide.md §9.4](../../../docs/developer_guide.md#94-debug-artifacts).

### `query_trace.py` — the 22-stage accumulator

`QueryTrace` is a pydantic model with 11 `QueryTraceStage` fields,
one per major stage (preprocessing, intent_classification,
query_expansion, embedding, retrieval, context_assembly,
prompt_construction, token_estimation, llm_stage,
citation_extraction, final_answer). Built incrementally by
`QueryService.ask()` when debug artifacts are active, serialized via
`model_dump(mode="json")`, stored under `query_traces/` by the
`DebugArtifactWriter`.

The module docstring calls it "no-op when the log level is not
DEBUG or TRACE" — in practice the `QueryTrace` itself is cheap to
construct (a pydantic model with default fields), but the *write* to
disk is gated by the writer's `is_active`. Construction is the
caller's decision; the bundle write is the writer's gate.

### `performance.py` — thin metrics veneers

Two helpers:

- `bulk_throughput(unit_count, elapsed_seconds, *, name, **tags)` —
  records `<name>.items` and `<name>.throughput_per_sec` via
  `Metrics.count(...)` / `Metrics.observe(...)`. Negative or zero
  `elapsed_seconds` reports 0 throughput (no division-by-zero noise).
- `ingestion_stage_timer(stage, **tags)` — context manager that
  pre-tags with `stage=<stage>` so dashboards can filter without
  re-typing the tag set at every call site. Internally uses
  `Metrics.timer("ingestion.stage.<stage>.duration_ms", **merged)`.

These don't replace `Metrics` — they thicken it with two conveniences
for ingestion-time reporting.

---

## 4. Configuration system (high level only)

This is **not** an env-var reference (see [developer_guide.md §10]
(../../docs/developer_guide.md#10-configuration) for the cross-cutting
configuration material). It is a description of the machinery.

`Settings` is the **only** source of truth for runtime configuration.
Three rules govern how it is read:

1. **Construct via `get_settings()`**, not `Settings()` directly. The
   `lru_cache(maxsize=1)` on `get_settings()` means every reader
   sees the same instance; tests call `reset_settings_cache()` to
   force a fresh load after monkeypatching env vars.

2. **Pass `settings` to constructors, do not import module globals.**
   `IngestionService`, `QueryService`, and the embedding / LLM /
   cache factories all take a `Settings` argument. They read inside
   their constructor (or per-call inside a method), not at import
   time. This is what lets a test build an `IngestionService` against
   a tempdir with a different `debug_artifacts_dir` than production.

3. **Bootstrap code uses `env_flag()`; nothing else does.** A handful
   of legacy readers (config bootstrap, the `core` module's own
   redaction toggles in old branches) read `os.environ` via
   `env_flag()` while a migration is in flight. The setting's module
   docstring is explicit: "preferred path is always to add the field
   to Settings".

The `_check_dev_keys_after` `model_validator` enforces one
combination the field types can't express: `dev-local-key` is
rejected when `environment == "production"` and
`reject_dev_keys_in_production` is `True` (the default). The
exception is `RuntimeConfigurationError`-shaped
`pydantic.ValidationError` at startup with an actionable message.

---

## 5. Logging architecture

### Levels

Five built-in levels plus one project-local:

| Level     | Numeric | Use                                                        |
| --------- | ------- | ---------------------------------------------------------- |
| TRACE     | 5       | Per-chunk, per-embedding, per-message verbose tracing.      |
| DEBUG     | 10      | Stage-by-stage detail. Activates the debug-artifact writer. |
| INFO      | 20      | Default production level. Stage boundaries + decisions.     |
| WARNING   | 30      | Recoverable anomalies (exporter init failed, etc.).        |
| ERROR     | 40      | Recoverable errors. Stack traces per `log_exceptions`.      |
| CRITICAL  | 50      | Process about to exit.                                      |

`TRACE` is registered with `logging.addLevelName(5, "TRACE")` so
`Settings.log_level = "TRACE"` and `logger.log(TRACE_LEVEL, ...)`
both work without per-call arithmetic.

### Formatters

Two formatters, swapped via `Settings.log_format`:

- `_HumanFormatter` — `asctime LEVEL  logger | message`, used in
  development.
- `_JsonFormatter` — one JSON line per record with stable shape:

  ```json
  {
    "ts": "2024-01-01T12:00:00Z",
    "level": "INFO",
    "logger": "semrag.services.services",
    "message": "...",
    "trace_id": "0...0",
    "span_id": "0...0",
    "<extra fields>": "..."
  }
  ```

  Every key in `LogRecord.__dict__` is serialised via
  `json.dumps(value)`; non-serialisable values fall back to `repr()`
  rather than dropping the entire record. Sensitive values are
  redacted via `redact_payload`. The schema is downstream-stable;
  keys don't change without a major version bump.

### Stages

`stage(name, *, logger, level="INFO", **tags)` is the context
manager every pipeline uses. Tags propagate to start + finish
records so a log pipeline can filter on `project_id` without
joining two records. `StageStats` is the mutable bag the caller
populates inside the block.

Stage naming is stable across the codebase — see
[services/README.md §5](../services/README.md#5-service-orchestration)
for the canonical ingest / query stage names.

---

## 6. Metrics

Two metric systems coexist, deliberately:

1. **`Metrics` (in `observability.py`)** — every call emits one
   `INFO`-level log record on the `semrag.metrics` logger with the
   stable prefix `metric=<name> value=<v> tags=...`. Library-free;
   ship the metrics logger to whatever backend the operator runs
   (Promtail, Vector, Fluent Bit). The call sites stay the same
   forever; the format is the deployment seam.

2. **Prometheus `REGISTRY` (in `metrics.py`)** — process-global
   `CollectorRegistry` constructed at import time. Counters,
   histograms, and gauges are constructed once and reused. The
   `/metrics` endpoint (`api/health.py`) renders the registry in the
   Prometheus text exposition format via `render_metrics()`.

Why both? The `Metrics` log-line path means the system is deployable
without a separate metrics backend while still being parseable into
Prometheus / OpenTelemetry later by any log pipeline. When a
Prometheus exporter is added, the only change is swapping the metric
handler — the call sites stay the same.

Use `Metrics` from application code (`Metrics.timer(...)`,
`Metrics.count(...)`) for per-call timing and counting. Use the
Prometheus inventory (`observe_http`, `inc_cache_hit`,
`observe_llm`, …) for stable, dashboard-facing series. The helper
function names are deliberately verbose so dashboards reference them
without ambiguity.

---

## 7. Tracing

`tracing.py` is the **only** OpenTelemetry seam. Application code
calls `tracer(name).start_as_current_span(...)` and never branches
on the `tracing_enabled` flag — when disabled, the no-op tracer
yields a `_NoopSpan` that returns itself from every method.

Configuration:

| Setting                 | Source                                       | Default                  |
| ----------------------- | -------------------------------------------- | ------------------------ |
| `tracing_enabled`       | `Settings.tracing_enabled`                   | `False`                  |
| `otlp_endpoint`         | `Settings.otlp_endpoint`                     | `http://localhost:4317`  |
| `otel_service_name`     | `Settings.otel_service_name`                 | `semrag`                 |
| `trace_sample_ratio`    | `Settings.trace_sample_ratio`                | `1.0`                    |

URL scheme → exporter:

- `http://` / `https://` → OTLPSpanExporter (HTTP/protobuf, port 4318).
- `grpc://` or no scheme → OTLPSpanExporter (gRPC, port 4317).

Sampling:

- `0.0` → `ALWAYS_OFF` (sampling is the toggle; feature flag is
  orthogonal — both are needed for "no spans at all").
- `1.0` → `ALWAYS_ON`.
- Anything in between → `TraceIdRatioBased`.

Exporter init failures are caught and logged at WARNING; the
provider is added without a processor. The application continues
with spans but no export. This is the "fail loud, don't crash"
behaviour.

---

## 8. Shared utilities

`core` exposes four shared utilities used by more than one
subpackage:

- **`get_logger(name)`** — the standard idiom. Every module does
  `logger = get_logger(__name__)` at module scope.
- **`Metrics.count / observe / timer`** — per-call timing and
  counting. Used by retrieval, embedding, the agent loop, the
  embedding pipeline, and the CLI.
- **`stage(...)`** — per-stage structured logging. Used by the
  service layer's `ingest.*` / `query.*` stages.
- **`redact_payload(payload)`** — applied automatically by
  `_JsonFormatter`. Not called directly by application code in
  production; tests that build log records by hand may use it.

`Settings` is also a shared utility in this sense — every
subpackage imports it, and every factory reads it.

---

## 9. Dependency rules

> **`core` imports from nothing in `semrag`. Every other subpackage
> imports from `core`.**

Verified against the source (`grep -rn "from semrag\."
src/semrag/core/`): every import inside `core` is either `from
semrag.core.X import ...` (intra-package) or a third-party library.
No module inside `core` imports from `ingestion`, `retrieval`,
`reasoning`, `storage`, `services`, `jobs`, or `api`.

The reverse is also true: `audit.py`, `worker.py`, `cli.py`, every
embedding provider, every LLM provider, the agent loop, the
retrieval pipeline, and the services layer all import from
`semrag.core.settings` or `semrag.core.observability` (often both).

This is what makes the package composable. If a future change adds
an import from `semrag.ingestion` inside `core`, the boundary has
slipped — and the slip will surface in code review because every
boundary table in the codebase ([src/semrag/README.md §5]
(../README.md#4-dependency-relationships), this README, and
[developer_guide.md §5](../../../docs/developer_guide.md#5-dependency-injection))
restates the rule.

Within `core`, the intra-package imports are:

```
observability → settings, redaction, tracing (lazy, for trace ids)
metrics       → (no intra-package imports)
tracing       → settings
redaction     → (no intra-package imports)
debug_artifacts → observability
query_trace   → (no intra-package imports)
performance   → observability
settings      → debug_artifacts (lazy, inside debug_writer())
```

The lazy imports (`settings.py → debug_artifacts`,
`observability.py → tracing`) are deliberate — they avoid importing
OTel SDK / artifact writers when only one half is used (e.g. an
embedding provider reading `Settings` does not need the artifact
writer loaded).

---

## 10. Extension points

`core` exposes a small, well-defined surface for extension.

| Need                                            | Where to change                                                                 | Notes                                                         |
| ----------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Add a new `SEMRAG_*` env var                    | `core/settings.py` — new `Settings` field, then read via `get_settings()`      | Add to one of the existing groups (Database / Retrieval / Operations / Security / Observability). Never `os.environ.get(...)` outside bootstrap. |
| Add a new log level or formatter                | `core/observability.py` — register the level with `logging.addLevelName(...)`   | Project-local levels must register with stdlib `logging` before being used in `Settings`. |
| Add a new Prometheus metric                     | `core/metrics.py` — add the metric near related helpers, add a `*_helper` function | Use `metrics.py:reset_registry()` to verify the rebind path. Never import the registry from outside `core`. |
| Add a new `Metrics` event type                  | `core/observability.py` — extend `Metrics.count / observe / timer`              | Keep the log-record format stable; downstream log pipelines depend on the schema. |
| Add a new sensitive key                         | `core/redaction.py` — add to `_SENSITIVE_KEY_MARKERS`                           | Match by case-insensitive substring; favour false positives over false negatives. |
| Add a new inline secret pattern                 | `core/redaction.py` — add to `_INLINE_SECRET_PATTERNS`                          | One line; do not introduce a recursive scrubber. |
| Add a new debug-artifact stage                  | `core/debug_artifacts.py` — add a `write_*` method and update the `start_run` directory list | Methods must short-circuit on `is_active` so the production hot path costs nothing. |
| Add a new query-trace stage                     | `core/query_trace.py` — add a `QueryTraceStage` field on `QueryTrace`            | Caller is responsible for populating it; the bundle write is gated by the artifact writer. |
| Add a new tracing context propagation entry     | `core/tracing.py` — extend `inject_context` / `extract_context`                  | Stick to W3C format. |

What is **not** an extension point:

- **The dependency rule.** `core` does not import from other
  SemRAG subpackages. There is no "soft" exception.
- **Module-level singletons.** Only the `Settings` lru-cache and the
  Prometheus `REGISTRY` are allowed; everything else is constructed
  per-call (e.g. `DebugArtifactWriter` is constructed by callers,
  not held as a module global).
- **Side-effect imports.** `core` modules must be safe to import
  individually. `settings.py` does not import `pydantic_settings`
  eagerly across the entire package — it imports only what it needs.

---

## 11. Common debugging paths

| Symptom                                                   | Where to look                                                                          |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Misconfigured deploy fails to boot                         | `pydantic.ValidationError` at the field that failed. The `_check_dev_keys_after` validator surfaces the production API-key rejection explicitly. |
| JSON logs are missing trace context                       | `_JsonFormatter` calls `current_trace_id_hex()` / `current_span_id_hex()` from `tracing.py`. Both return zeroed W3C-format sentinels when no span is active; confirm `Settings.tracing_enabled=true` *and* a span is active on the context. |
| Secrets leaking into logs                                 | `core/redaction.py` is the chokepoint. If a key slips through, extend `_SENSITIVE_KEY_MARKERS` (case-insensitive substring match) or `_INLINE_SECRET_PATTERNS`. |
| Prometheus `/metrics` endpoint empty / 404                 | `Settings.metrics_enabled` controls the `/metrics` route. When `False`, the route returns 404. When `True`, `core/metrics.py:render_metrics()` returns the registry in the Prometheus text format. |
| Metrics registry polluted between tests                    | `reset_registry()` — rebuilds `REGISTRY` and rebinds every metric via `_rebind_metrics()`. Production never calls this. |
| OTel exporter init failed at startup                      | `core/tracing.py` catches the exception, logs WARNING, continues without a processor. Spans are still created but never exported. Check the OTLP endpoint URL scheme (`http+` vs `grpc://`). |
| Debug-artifacts bundle is missing despite `log_level=DEBUG` | `Settings.debug_artifacts_dir` must resolve to a writable path. The writer is silent when it cannot write — verify the directory exists and the process can write to it. `DebugArtifactWriter.is_active` returns `True` only when `Settings.log_level` is `DEBUG` or `TRACE`. |
| `get_settings()` returns stale values in a test            | `reset_settings_cache()` (clears the `lru_cache`). For hot-swap inside a test, use `Settings().replace(...)` and pass the result to constructors explicitly. |
| `Settings` validator raises "Default development API key"  | `SEMRAG_API_KEYS` still contains `dev-local-key`. The validator rejects it when `environment == "production"` (and `reject_dev_keys_in_production == True`, the default). |
| A new metric label explodes Prometheus cardinality         | Check `core/metrics.py` for the documented cardinality rules. `path` must be the FastAPI route template, never the rendered URL; `backend` is two values; `tool` is the `ToolRegistry` surface. |
| `QueryTrace` is missing a stage                            | `QueryService.ask()` populates each `QueryTraceStage` field by hand. Check [services/README.md §5](../services/README.md#5-service-orchestration) for which stages are populated where. |
| Stdlib loggers (urllib3, httpx, sqlalchemy.engine) spam output | `configure_logging` sets them to `WARNING`. If they re-appear, a third-party library may have configured its own logger; add it to the `noisy` list in `configure_logging`. |

For more debugging material, see
[developer_guide.md §14](../../../docs/developer_guide.md#14-debugging-workflow).

---

## 12. Common mistakes

- **Reading `os.environ` directly.** The project rule is "add a
  field to `Settings`". `env_flag()` is bootstrap-only; new code
  does not use it. The exception that justifies the helper (legacy
  readers mid-migration) is documented in the helper's docstring.
- **Constructing `Settings()` directly.** Use `get_settings()`. The
  `lru_cache` keeps a single instance so every reader sees the
  same value; `Settings()` directly bypasses that.
- **Holding `Settings` as a module-level singleton.** Pass it to
  constructors explicitly. `IngestionService` and `QueryService`
  take `settings` as a constructor argument; they read inside
  `__init__` and inside their methods, not at import time.
- **Forgetting to call `reset_settings_cache()` after monkeypatching
  env vars in tests.** Without it, the `lru_cache` keeps the old
  instance and the new env vars never take effect. The canonical
  pattern is `monkeypatch.setenv(...)` → `reset_settings_cache()` →
  rebuild the service.
- **Calling `configure_logging` twice and stacking handlers.** The
  `_configured` guard makes it idempotent, but the second call is a
  no-op for a reason — there is no operator-driven reason to
  re-configure at runtime. Tests that need a fresh config call
  `reset_*()` helpers, not `configure_logging`.
- **Bypassing `redact_payload` for "obvious" secrets.** The
  redactor is the single chokepoint; ad-hoc scrubbing in a router
  or service is what causes leaks. When in doubt, redactor it.
- **Importing `from semrag.ingestion...` inside `core`.** This
  violates the dependency rule. There is no exception; the rule
  is what makes the package composable. If a `core` module needs
  to do something domain-specific, the helper belongs in the
  domain subpackage and `core` consumes it via the same rule.
- **Using `print()` instead of `get_logger`.** Stdlib logging is
  configured once and consistently — format, level, handlers,
  redaction. `print()` bypasses all of it and is the primary
  reason a structured log line goes missing in a production
  investigation.
- **Mutating `Settings` fields directly.** The model is mutable
  (the design choice documented in `settings.py` is deliberate
  for hot-swap), but production code never assigns
  `settings.foo = ...`. Use `settings.replace(foo=...)` so the
  change is auditable.
- **Constructing counters/histograms at request time.** All
  Prometheus metrics are constructed once at import time in
  `core/metrics.py` and reused. Constructing a `Counter` per
  request is a measurable cost; the registry would also fill
  with duplicate registrations.

---

## 13. Architectural boundaries

Four boundaries keep `core` composable.

### 1. `core` is the only package with no siblings in its import graph

The dependency graph in [src/semrag/README.md §4]
(../README.md#4-dependency-relationships) shows `core` floating at
the bottom. No other subpackage sits at the same level — every
other subpackage depends on `core` and is depended on by `api` /
`services` / `jobs` in turn. This is what makes `core` a
"universal dependency" without becoming a god module: it has
nothing *but* cross-cutting primitives.

### 2. `core` does not know about the domain subpackages

A new domain type (a `Chunk`, a `RetrievalResult`, an `Embedding`)
does not live in `core`. The criterion is **used by ≥2
subpackages**; if the answer is one, the type stays in its owning
subpackage. The developer guide restates this in [§5]
(../../docs/developer_guide.md#5-dependency-injection) and
[CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md) uses it as the
default location rule.

### 3. `core` does not own runtime state

Module-level singletons are forbidden except for two documented
exceptions:

- `get_settings()` is `@lru_cache(maxsize=1)` — the same
  `Settings` instance is returned every call. This is the
  configuration singleton and is reset by
  `reset_settings_cache()` in tests.
- `REGISTRY` in `core/metrics.py` is a process-global
  `CollectorRegistry`. This is the Prometheus fixture and is
  reset by `reset_registry()` in tests.

Everything else (loggers, metrics counters, the artifact writer,
the tracer) is constructed per-call. The artifact writer's
`start_run` is idempotent for the same `run_id`, which is what
lets the same writer instance be reused across a run.

### 4. `core` is not the place for HTTP, FastAPI, or request-shape primitives

Anything that imports `fastapi`, `starlette`, or `pydantic` for
request-body validation belongs in `api/`. The `Settings` model
itself uses pydantic, but that is configuration schema, not
request schema — the boundary is on *what the schema is for*.

---

## 14. Related documentation

- [src/semrag README](../README.md) — package map and the dependency
  graph showing `core` at the bottom.
- [src/semrag/services/README.md](../services/README.md) — how the
  service layer uses `core`'s primitives (`stage(...)`,
  `DebugArtifactWriter`, `QueryTrace`, `tracer(...)`).
- [src/semrag/api/README.md](../api/README.md) — how the API wires
  `Settings`, `configure_logging`, `configure_tracing`, and the
  Prometheus helpers (`observe_http`, `inc_rate_limit_*`,
  `inc_cache_*`).
- [developer_guide.md](../../../docs/developer_guide.md) — end-to-end
  observability (§9), dependency injection philosophy (§5), testing
  patterns (§12), and debugging workflow (§14).
- [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md) — file-by-file
  ownership and "where do I change X?" tables.
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — long-form architecture
  including the placement of `core` relative to the other
  subpackages.
- [docs/operations/README.md](../../../docs/operations/README.md) —
  on-call runbooks; `high-latency.md` and `database-outage.md` are
  the ones that lean most heavily on the metrics exported by
  `core/metrics.py`.