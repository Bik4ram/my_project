# Changelog

All notable changes to SemRAG are documented in this file. Releases are
dated; entries grouped by release.

---

## Unreleased

### Wave 3 — Observability & Enterprise Operations

- **Prometheus metrics** — `semrag.core.metrics` registers a private
  `CollectorRegistry` exposing the full Wave 3 metric set:
  `semrag_http_requests_total`, `semrag_http_request_duration_seconds`,
  `semrag_http_request_size_bytes`, `semrag_http_response_size_bytes`,
  `semrag_http_exceptions_total`, `semrag_rag_retrieval_duration_seconds`
  (stage label), `semrag_rag_rerank_duration_seconds`,
  `semrag_rag_embedding_duration_seconds` (provider label),
  `semrag_rag_llm_duration_seconds` (provider+model labels),
  `semrag_rag_tool_duration_seconds` (tool label),
  `semrag_indexing_duration_seconds` (project_id label),
  `semrag_indexing_files_processed_total`,
  `semrag_indexing_chunks_generated_total`,
  `semrag_indexing_chunks_embedded_total`,
  `semrag_indexing_jobs_{queued,completed,failed}_total` (project_id
  label), `semrag_db_pool_connections`, `semrag_db_query_duration_seconds`,
  `semrag_db_transactions_total`,
  `semrag_cache_{hits,misses,evictions}_total`,
  `semrag_rate_limit_{accepted,rejected}_total`, `semrag_uptime_seconds`,
  `semrag_health_up{component=...}`. Cardinality is bounded: HTTP paths
  use the route template (`/v1/projects/{project_id}/query`), never the
  raw URL. `reset_registry()` keeps tests isolated.
- **OpenTelemetry tracing** — `semrag.core.tracing` configures a
  `TracerProvider` from settings (`tracing_enabled`, `otlp_endpoint`,
  `otel_service_name`, `trace_sample_ratio`) and exports via OTLP gRPC
  (default port 4317) or HTTP/protobuf (port 4318) depending on the URL
  scheme. `tracer()` returns a no-op when disabled or when the SDK is
  not installed, so the rest of the app stays decoupled from OTel.
  `current_trace_id_hex()` / `current_span_id_hex()` stamp every log
  record with the active W3C ids. `inject_context()` /
  `extract_context()` propagate the trace across HTTP and DB
  boundaries.
- **Secret redaction in structured logs** —
  `semrag.core.redaction.redact_payload()` runs in the JSON formatter
  before any payload is written. Sensitive keys (`api_key`, `token`,
  `password`, `authorization`, `secret`, `private_key`,
  `connection_string`, …) are replaced with `***`; free-form string
  values are scrubbed for Bearer tokens, AWS access key ids, JWTs, and
  inline credentials in Postgres / MySQL URLs.
- **HTTP middleware** — `MetricsMiddleware` records request count,
  latency, request/response sizes, and per-exception counters; route
  templates are extracted from the FastAPI route to keep label
  cardinality bounded. `TracingMiddleware` opens a server span per
  request, injects W3C context into outbound calls, and stamps
  `trace_id` / `span_id` into the request's structured log fields.
- **Health metrics** — `/readyz` updates `semrag_health_up{component}`
  gauges to 1.0 / 0.0 after every probe, so dashboards and alerts can
  reason about which dependency is down without scraping `/readyz`.
  `/metrics` is exposed only when `SEMRAG_METRICS_ENABLED=true` and
  refreshes `semrag_uptime_seconds` on each scrape.
- **Monitoring stack** (`deploy/monitoring/`) — Prometheus
  scrape config + recording rules, Alertmanager routing tree with
  PagerDuty / Slack receivers, Grafana datasources + provider, and
  eight dashboards (API, retrieval, LLM, indexing, jobs, database,
  cache, rate-limit) covering the full Wave 3 metric set.
- **Alert rules** (`deploy/monitoring/prometheus/rules/semrag.yml`) —
  six alert groups (Availability, Latency, Indexing, Database, LLM /
  Embedding, RateLimiter) with `for:` windows tuned to the SLOs.
- **Operational runbooks** (`docs/operations/`) — README plus
  database-outage, llm-outage, embedding-failure, indexing-failure,
  job-backlog, high-latency, memory-exhaustion, rollback, and
  escalation guides. Each runbook lists symptoms, dashboard links,
  first-line checks, escalation criteria, and rollback steps.
- **Load testing** — `tests/load/locustfile.py` covers query, answer,
  indexing, and health traffic; `tests/load/run_load_test.py` is a
  dependency-free smoke test that runs without Locust installed.
- **Performance regression suite** —
  `tests/perf/run_perf_suite.py` measures p50/p95/p99 latency on the
  in-process TestClient and optionally against a live URL; reports a
  pass/fail line per endpoint against configurable thresholds.
- **Tests** — 28 new unit tests cover the redaction logic, the
  Prometheus exposition format, the OTel facade (including no-op
  paths), and the `/metrics` + `/readyz` wiring. Full unit suite: 513
  passing, 80 pre-existing skips, 0 regressions.

### Wave 2 — Deployment & Runtime

- **`GET /livez`, `GET /readyz`, `GET /version`** — production-grade
  health endpoints. `/livez` is process liveness with no external
  dependency checks. `/readyz` checks Postgres + pgvector, LLM and
  embedding configuration, the JobExecutor, cache, and rate-limiter
  backends; returns 503 with a structured body when any check fails.
  `/version` returns app version, git commit, build timestamp, and
  Python version from package metadata. Reuses the existing engine
  and JobExecutor singletons — no parallel startup path.
- **Runtime configuration validation** —
  `semrag.api.startup_validation.validate_runtime_config()` runs in
  the FastAPI lifespan at process start. Aggregates every misconfig
  into one actionable error so a bad deploy fails fast.
- **Graceful shutdown** — FastAPI lifespan `shutdown_resources()`
  drains the JobExecutor (`wait=True`), disposes the SQLAlchemy engine
  pool, and is idempotent across multiple invocations.
- **docker-compose.yml + override** — production-grade Compose with
  pgvector/pgvector:pg16, profile-gated Ollama and Redis, named
  networks, persistent volumes, healthchecks, resource limits, and
  `stop_grace_period` for clean SIGTERM handling. The override binds
  ports to loopback, mounts the source tree, and runs uvicorn with
  `--reload` for local dev.
- **Kubernetes manifests** (`deploy/k8s/`) — Namespace, ConfigMap,
  Secret template, PVC, Deployment, Service, Ingress. Liveness uses
  `/livez`, readiness uses `/readyz`; startup probe gives the first
  LLM probe 90s to succeed. `preStop` sleep + 30s
  `terminationGracePeriodSeconds` drains in-flight requests.
- **Helm chart** (`deploy/helm/semrag/`) — Chart.yaml, values.yaml,
  templates (Deployment, Service, Ingress, ConfigMap, Secret, PVC,
  ServiceAccount, HPA, NOTES.txt), helpers. Parameterized for
  database, Redis, Ollama, replicas, resources, image, ingress.
- **Deployment documentation** (`DEPLOYMENT.md`) — every supported
  target (local, Docker, Compose, K8s, Helm), env-var reference,
  production checklist, troubleshooting.

- **Bug fix** — `LoggingMiddleware` and `log_audit_event` were
  passing user fields as plain kwargs to `logger.info()`, which the
  stdlib rejects with `TypeError: Logger._log() got an unexpected
  keyword argument`. Routed through `extra=` instead so the existing
  JSON formatter still picks them up via `LogRecord.__dict__`.
- **Bug fix** — `get_job_executor()` now lazily re-creates the
  executor after shutdown. Lets a process that receives SIGTERM and
  then a second SIGUSR (or a test fixture that runs the lifespan
  twice) come back with a healthy executor instead of a permanently
  shut-down singleton.

### Retrieval Improvement — Source Authority (RI-02)

- **Source Authority model** (`retrieval/source_authority.py`) — the
  document-aware ranking adjustment is now a single configurable model
  that separates (1) **document classification** (path + `chunk_type`
  rules map a chunk to a semantic category like `rtl`, `specification`,
  `simulation_log`, `uvm`, `coverage_report`), (2) **authority
  assignment** (every category maps to a weight in
  `SourceAuthorityConfig.authorities` — the one place operators tune),
  and (3) **score adjustment** (`apply_source_authority` applies the
  multiplicative nudge bounded by `max_boost` and re-sorts). Adding a
  new document type (waveforms, datasheets, verification plans, timing
  reports) is a config change, not a code change.
- **Backward compatibility** — `doc_type_prior.py` is now a thin shim
  keeping `apply_doc_type_prior` and `DEFAULT_MAX_BOOST`; the default
  authority table is the verbatim translation of the RI-01 weights, so
  default ranking is unchanged. The adjustment is recorded under both
  the canonical `signal_scores["source_authority"]` key and the legacy
  `signal_scores["doc_type_prior"]` key, and `ChunkTrace` gains an
  additive `source_authority` field — so the RB-02 failure analyzer and
  RB-03 analytics keep working without modification.
- **Configurable from one place** — `SourceAuthorityConfig` (plus
  `Settings.source_authority_overrides` / `source_authority_max_boost`
  wired through `QueryService`). `HybridRetriever` and
  `apply_source_ranking` accept an optional `source_authority_config`
  (default preserves existing behavior). No document type is filtered —
  logs remain valid retrieval candidates, only ordering changes.

### Retrieval Improvement — Retrieval Diversity (RI-03)

- **Diversity-aware selection** (`retrieval/ranking/diversity.py`) — a
  new ranking stage (6.5) that runs after every ranking score is final
  (RRF, graph expansion, intent boost, source authority, reranker) and
  immediately before the stable top-k slice. It selects the final
  top-k so the LLM context holds complementary sources — multiple
  document categories and files — instead of many near-identical chunks
  from a single file. The selection is a deterministic greedy pass in
  score order: the highest-scored chunk is always kept, and a later
  candidate is skipped only when it is redundant given the chunks
  already chosen (per-file cap, per-category cap, and suppression of
  adjacent windows of the same file). Independent of graph, intent, and
  source authority; relevance is preserved because diversity only
  influences later selections.
- **Configurable from one place** — `DiversityConfig` (enabled,
  `max_per_file`, `max_per_category`, `neighbour_window`,
  `pool_multiplier`) wired through `Settings.retrieval_diversity_*` and
  `QueryService`. When a reranker is configured, the pipeline asks it
  for `top_k * pool_multiplier` candidates so the selection has a larger
  pool to diversify from (the reranker itself is untouched). Disabling
  the stage reproduces the pre-diversity ranking byte-for-byte.
- **Explainable** — every evaluated candidate records
  `signal_scores["diversity"]` (1.0 selected / 0.0 rejected) and a
  reason (`selected`, `file_limit`, `category_limit`, `neighbour`) in the
  additive `ChunkTrace.diversity` / `ChunkTrace.diversity_reason`
  columns. RB-02 reconstructs a DIVERSITY stage (present only when the
  trace recorded diversity, so legacy traces reconstruct the exact
  pre-RI-03 orderings) and attributes chunks dropped by the final
  selection to the new `FailureCategory.DIVERSITY`; RB-03 maps it to a
  `diversity` responsibility bucket.

### Retrieval Improvement — Better Log Chunking (RI-04)

- **Event-aware log chunker** (`ingestion/chunking/log_chunker.py`,
  registered for `FileCategory.LOG` in `chunking/pipeline.py`) —
  replaces generic 60-line window slicing for logs so semantically
  meaningful multi-line events stay together: a UVM check message plus
  its packet table plus its `[ADDR_PASS]`/`[X_FAIL]` verdict, or an
  `ERROR`/`FATAL` line plus its continuation/stack trace, become one
  chunk. Boundaries are derived from the existing regression-log parser
  vocabulary and the repository's real QuestaSim UVM corpus
  (`sim/log/write1_test_log.log`): UVM severities anywhere in the line,
  generic severity/result markers at line start; `PASS`/`FAIL`/`INFO`
  alone are not boundaries. UVM per-macro occurrence tails
  (`UVM_ERROR at time ... : N occurrences`) are continuations, not
  boundaries, so a FATAL stack trace isn't split by its own summary line.
- **Noise handled, nothing deleted** — consecutive single-line events
  (the corpus's ~41% repeated `[STATE_COV_ITEM]` messages) group into
  bounded 60-line windows instead of one chunk per line, and the UVM
  Report Summary swallows the file to EOF so severity-count lines don't
  re-split. Every line still lands in exactly one chunk (verified
  no-gap/no-overlap over the full 4003-line corpus). Oversized events
  and marker-less logs reuse the existing 60/10 overlapping-window
  convention, so marker-less logs chunk identically to
  `GenericWindowChunker`.
- **Metadata contract unchanged** — `chunk_type="log"` (the reserved
  vocabulary `services._NOISE_CHUNK_TYPES` and
  `evaluation.retrieval_analytics.categorize_document` already consume),
  `source_node_id=None`, deterministic content-derived `chunk_id` with
  sequential disambiguator, 1-indexed inclusive line ranges. No change to
  vector/lexical search, source authority, diversity/intent ranking,
  reranker, or graph ranking.

### Retrieval Improvement — Deterministic Query Expansion (RI-05)

- **Deterministic, lexical-side query expansion**
  (`retrieval/query_expansion.py`) — when a question uses different
  wording than the repository (a user asks "why did the test fail?" but
  the corpus says `UVM_ERROR ... mismatch`), the lexical leg silently
  misses chunks the vector leg already tolerates via paraphrase. The
  expansion generates up to `max_expansions` additional search
  formulations, each = the original query + a bounded group of the
  intent's repository terminology (reusing `intent.py`'s existing
  `_INTENT_KEYWORDS` vocabulary, exposed via the new public
  `intent_keywords()`), searched by the existing `PostgresLexicalSearch`
  and fused into the same RRF under distinct signal names
  (`lexical_exp1`, `lexical_exp2`, ...). Zero-network and pure — no LLM,
  no extra embedding calls, no added latency beyond a bounded set of FTS
  queries — so it stays deterministic and fail-open.
- **Original query always searched** — `expand_query` never replaces the
  query; the pipeline searches the original text first and the
  formulations are additive signals. No trivial expansions: terms already
  present are not re-appended, and GENERAL/structural/blank queries (and
  queries already covering every domain term) produce no formulations.
- **Bounded + backward-compatible** — output capped by
  `max_expansions` / `max_expansion_length`; pool bounded at
  `(1 + max_expansions) * lexical_top_k + vector_top_k` candidates before
  RRF dedups by `chunk_id`. `HybridRetriever(query_expansion_config=None)`
  is byte-identical to pre-RI-05 behavior; expansion is enabled only when
  explicitly configured (service + API wiring).
- **Explainable trace (additive)** — `RankingTrace` gains `expansions`,
  `expansion_latency_ms`, `expansion_failed`, `expansion_enabled`;
  `ChunkTrace` gains `retrieved_by_original`,
  `retrieved_by_expansion`, `expansion_queries`; `QueryService` copies
  the metadata onto the existing `QueryTrace.query_expansion` stage.
- **Config** — `SEMRAG_QUERY_EXPANSION_ENABLED` (default `True`),
  `SEMRAG_QUERY_EXPANSION_MAX_EXPANSIONS` (3),
  `SEMRAG_QUERY_EXPANSION_MAX_EXPANSION_LENGTH` (80),
  `SEMRAG_QUERY_EXPANSION_TIMEOUT_MS` (`None`; reserved for a future
  external/LLM mechanism, unused by the deterministic implementation).
  No ranking stage changed: source authority, diversity, intent ranking,
  graph ranking, and the reranker are untouched. `failure_analyzer.py`
  reports expansion-only chunks as `_retrieved_stage="unknown"` (an
  additive-fields-only gap, safe by construction; tracked as follow-up).

### Retrieval Improvement — Better Retrieval Metadata (RI-06)

- **Additive chunk metadata, factual and single-sourced** — `Chunk`
  (and its storage/retrieval mirrors) gains `language`, `category`,
  `parser_name`, `chunker_name`, `symbol_name`, plus a derived
  `is_generated` property. Every field has one canonical source and is
  never re-derived downstream: `language` from the scanner's canonical
  extension mapping (`scanner/classification.language_for_extension`),
  `category` from the existing Source Authority
  `classify_document(chunk_type, relative_path)` (reused, not a second
  classifier), `parser_name` from the `ParseResult` that actually ran,
  `chunker_name` stamped by `ChunkerRegistry` (the one place that knows
  which chunker really produced the chunk, fallback included),
  `symbol_name` set by chunkers only where the parser genuinely knows a
  module/class/function name (never heuristically invented — generic
  text/log/markdown chunks are `None`), and `is_generated` derived from
  `category == "generated"` rather than a stored duplicate flag.
- **Enriched once, centrally** — `chunk_project` fills the
  file-level fields (`language`, `category`, `parser_name`) for every
  chunk in one pass, so no downstream stage recalculates them. Embedding
  text is untouched (metadata and embedding stay separate concerns);
  `embed_chunks` passes the fields through to `VectorChunk` unchanged.
- **Full propagation path** — metadata survives
  `Chunk → VectorChunk → chunks table → VectorSearchResult →
  RetrievalResult → ChunkTrace`, including the lexical and graph
  expansion paths that construct `RetrievalResult` directly from SQL.
  The ingestion debug artifact (`chunks/chunk_metadata`) and the
  ranking trace now expose the fields additively.
- **Migration `a1c06e93d2f4`** — five nullable columns
  (`language`, `category`, `parser_name`, `chunker_name`, `symbol_name`)
  on `chunks`; additive and safe for existing rows (they read back with
  empty/`None` values until re-ingested). Defaults on the models keep
  pre-RI-06 construction sites and persisted rows compatible.
- **Metadata enablement only** — no ranking behavior changed. Source
  authority weights, diversity, graph/intent ranking, the reranker,
  vector similarity, lexical scoring, and query expansion are untouched;
  the metadata is available for future retrieval/evaluation stages.

### Retrieval Improvement — Metadata-Aware Reranking (RI-07)

- **Bounded metadata tie-break in the lexical-overlap reranker** —
  `LexicalOverlapReranker` keeps its hybrid-aware multiplicative Jaccard
  boost unchanged and adds a deterministic, bounded tie-breaker: within
  a near-tie (post-Jaccard scores within `near_tie_epsilon`, default
  `1e-4`), a candidate whose `symbol_name` matches a query term may be
  promoted ahead of its near-tied peers — but ONLY when the existing
  QueryIntent / structural-query logic identifies the query as
  code-navigation (ARCHITECTURE intent or a structural query). It is a
  tie-break, never a score: a gap larger than the epsilon is never
  displaced, and for non-navigation queries, missing metadata, or a
  disabled tie-break the ordering is exactly the previous deterministic
  ordering. No cross-encoder, no LLM reranker, no new retrieval stage,
  no new multiplicative score, no Source Authority / Intent / Diversity
  changes; provenance (generated vs authored) is deliberately inert in
  the tie-break.
- **Fail-open reranker contract** — a configured reranker that raises no
  longer aborts retrieval: the pipeline catches the exception, falls
  back to the pre-reranker ranking, logs it, and records
  `trace.rerank_failed = True` on the existing `RankingTrace` (additive
  field). The same guard protects the streaming query endpoint. The
  `HTTPRerankerClient` itself is untouched (still unwired, tested via
  mocks).
- **Config** — two additive settings:
  `retrieval_rerank_metadata_tiebreak` (default `true`) and
  `retrieval_rerank_near_tie_epsilon` (default `1e-4`), wired through
  `QueryService._build_retriever` and both query endpoints.
- **Explainability** — the decision is recorded under
  `signal_scores["rerank_metadata_prior"] = 1.0` only on candidates the
  tie-break actually promoted; the existing trace architecture is
  reused with a single additive `rerank_failed` field.

### Evaluation — rtl_uvm Retrieval Ground Truth (RB-04)

- **Curated golden query set for the production I2C UVM corpus**
  (`evaluation/benchmarks_rtl_uvm.py`) — 50 `GoldenQuery` entries for
  project `rtl_uvm` (the repo at `~/projects/RTL_UVM`: RTL, UVM
  testbench, QuestaSim logs, coverage reports) with exact
  `relevant_chunk_ids`, `primary_files`, intent and category labels.
  Category mix (guidelines, not quotas): code navigation 8, design
  explanation 8, debugging 8, assertion/verification 5, bus/parameter
  ("register") 6, UVM/testbench 6, log investigation 4, terminology
  mismatch 5. Delivered as an additive module (`BENCHMARKS_RTL_UVM` +
  `get_rtl_uvm_benchmarks()`) following the README-documented
  per-project pattern; the existing semrag `benchmarks.py` set and its
  tests are untouched.
- **Ground truth sourced offline, never by the retriever** — every
  `relevant_chunk_id` was taken verbatim from the deterministic
  chunk-ID space produced by a fresh offline scan -> parse -> IR ->
  chunk run of the corpus with the current code (467 chunks, no
  embeddings, no DB writes) and judged against the actual chunk content
  in that dump. Debugging queries are grounded in code-level evidence
  (scoreboard `*_FAIL` branches, uncovered FSM states/transitions in
  `code_coverage.txt`) because the corpus logs are pass-only (0
  UVM_ERROR/UVM_FATAL). Log queries target the RI-04 event-aware `log`
  chunks (e.g. `sim/log/write1_test_log.log::log::24` ADDR_MATCH packet,
  `::205` UVM Report Summary).
- **`repo_area=GENERAL` throughout** — `RepoArea` describes the semrag
  platform's own modules, which do not apply to the external hardware
  corpus; `expected_intent` carries the meaningful label (`RTL`,
  `VERIFICATION`, `DEBUG`, `CONFIG`, `GENERAL`).
- **Validation tests** (`tests/unit/evaluation/test_benchmarks_rtl_uvm.py`)
  — query count in range, required fields, unique ids, project
  consistency, well-formed `rtl_uvm::path::chunk_kind::disambiguator`
  chunk IDs belonging to the project, no duplicate IDs within a query,
  no semrag-repo path leakage, category distribution bounds, intent
  coverage (RTL + DEBUG + VERIFICATION), `GoldenQuery` serialization
  round-trip, and primary-file membership in the corpus. The full
  cross-check of every referenced chunk ID against the authoritative
  467-chunk set passed during curation.

### Phase 3 — Production hardening

- **Centralized settings** — pydantic-settings with `SEMRAG_` env prefix.
  Single source of truth replacing scattered `os.environ.get(...)` reads.
- **Structured logging & metrics** — `core.observability` ships JSON or
  human output, plus a `Metrics` facade that emits `metric=` events on
  the `semrag.metrics` logger for retrieval latency, embedding cache
  hit/miss, LLM token usage, tool execution, agent iterations, and
  query timing.
- **LLM provider abstraction** — `LLMProvider` Protocol with concrete
  `OllamaLLM` (native `/api/chat` for Ollama 0.3x), `OpenAILLM`,
  `OpenAICompatibleLLM` (covers vLLM and any OpenAI-compatible shim),
  `MockLLM`. Adding a provider is a one-line `elif` in
  `reasoning/llm/factory.py`.
- **Embedding provider abstraction** — same pattern, factory at
  `ingestion/embedding/factory.py`.
- **Service layer** — `IngestionService.index()` and
  `QueryService.ask()` orchestrate the full pipeline. CLI and the FastAPI
  routers depend on services only.
- **Bug fix** — LLM HTTP client was assembling URLs as
  `base_url + endpoint` where both already contained the host:port,
  producing `http://xhttp://x/v1/chat/completions`. `_endpoint()` now
  returns a path; the shared helper joins.
- **Bug fix** — `Metrics.timer()` / `count()` collided with a tag named
  `name` because their first positional argument is `name`. Replaced
  with `tool=` in agent metrics.
- **Bug fix (Phase 1)** — NUL bytes from PDF file reads no longer crash
  Postgres binding; binary files are skipped during chunking, and
  defensive NUL stripping in `PgvectorStore`.
- **Bug fix (Phase 1)** — Ollama 0.32.x returns 404 on
  `/v1/chat/completions`. Routed through `/api/chat` instead.

### Phase 3.5 — Production observability

- **TRACE log level** — between DEBUG and the floor. Operators opt in via
  `SEMRAG_LOG_LEVEL=TRACE` for per-request detail.
- **Debug artifact writer** — `.semrag/debug/<run-id>/` written only at
  DEBUG/TRACE; categories: `scan/`, `parse/`, `ir/`, `chunks/`,
  `embeddings/`, `knowledge_graph/`, `retrieval/`, `prompt/`, `llm/`,
  `timings/`.
- **Configurable retention** — default 5 runs; configurable via
  `SEMRAG_DEBUG_ARTIFACT_RETENTION`. Older runs are pruned automatically
  on each run completion.
- **Per-stage instrumentation** — `stage()` context manager emits
  start/finish log lines with elapsed_ms, status, and stats. Major
  pipeline stages in `IngestionService.index()` and `QueryService.ask()`
  are wrapped.
- **Token usage capture** — `CompletionResult` reports
  `prompt_tokens`/`completion_tokens`/`total_tokens`. OpenAI-compat
  providers read `usage.*`; Ollama reads `prompt_eval_count`/`eval_count`.
  Counters: `llm.tokens.total`, `llm.tokens.prompt`,
  `llm.tokens.completion`.

### Documentation

- `README.md` — quickstart, env-var table, observability summary.
- `ARCHITECTURE.md` — module layout, data-flow diagrams, provider
  abstractions, testing strategy.
- `CHANGELOG.md` — this file.
- `AGENTS.md` — repository guidelines unchanged at the policy level,
  cross-links to `ARCHITECTURE.md` and `README.md`.

---

## Pre-Phase 3

Pre-history — initial scanner, parsers, IR builder, chunkers, embeddings,
retrieval, agent, API, and CLI were all in place and indexed/query
end-to-end against a working Postgres + Ollama before the Phase 3
refactor began.
