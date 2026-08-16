# Developer Guide

> **Engineering handbook for SemRAG.** This document teaches how the
> system actually works, why each subsystem is shaped the way it is, and
> how to safely extend or debug it. It does not duplicate the end-to-end
> data-flow diagrams — see [ARCHITECTURE.md](../ARCHITECTURE.md) for
> those. It does not catalog every module — see
> [CODEBASE_GUIDE.md](./CODEBASE_GUIDE.md) for that.

Intended audience: **Developers** and **Maintainers** working on the
codebase.

---

## Table of contents

1. [Introduction](#1-introduction)
2. [High-level architecture](#2-high-level-architecture)
3. [End-to-end request lifecycle](#3-end-to-end-request-lifecycle)
4. [End-to-end indexing lifecycle](#4-end-to-end-indexing-lifecycle)
5. [Dependency injection](#5-dependency-injection)
6. [Storage layer](#6-storage-layer)
7. [Background jobs](#7-background-jobs)
8. [Security model](#8-security-model)
9. [Observability](#9-observability)
10. [Configuration](#10-configuration)
11. [Error handling](#11-error-handling)
12. [Testing strategy](#12-testing-strategy)
13. [Extension points](#13-extension-points)
14. [Debugging workflow](#14-debugging-workflow)
15. [Common architectural rules](#15-common-architectural-rules)

---

## 1. Introduction

### 1.1 Goals

SemRAG is a production RAG platform for **semiconductor design
repositories** — RTL, UVM, SystemVerilog, plus PDFs, Excel, and
Markdown. Its goal is to give verification and RTL engineers trustworthy,
grounded, cited answers about large codebases, without ever inventing a
file path or line range that is not in the retrieved context.

Concretely:

- A **hybrid retriever** (vector + lexical + graph expansion, fused by
  Reciprocal Rank Fusion) returns the most relevant chunks for a query.
- A **tool-calling ReAct agent** produces a final answer in natural
  language with `[Source: path, lines X-Y]` citations.
- A **citation validator** confirms that every cited path corresponds to
  a real retrieved chunk; fabricated citations are surfaced as a metric
  rather than silently passing.
- The whole pipeline is **observability-first**: a per-stage trace, a
  per-stage log line, a per-stage Prometheus metric, and an optional
  per-run debug artifact bundle.

### 1.2 Design philosophy

Five ideas shape every PR that lands in this codebase.

1. **Be opaque only where truth matters.** Retrieval ranking, citation
   mapping, and persistence are fully observable; the LLM call itself is
   treated as untrusted input whose output is checked for citations.
2. **One concern per package; one Protocol per pluggable seam.** Each
   `Protocol`/`ABC` is the only seam a new backend needs to fill. The
   rest of the code is unaware of which backend is loaded.
3. **No silent fallback.** A `backend=redis` setting with a missing
   `SEMRAG_REDIS_URL` raises `RuntimeError` at first use; it does not
   downgrade to memory. See `redis_client._require_url()` in
   `src/semrag/api/redis_client.py`.
4. **Pragmatic over pure.** Settings are a single mutable pydantic
   singleton because hot-swap during tests is worth the loss of
   `@dataclass(frozen=True)`.  See `core/settings.py`.
5. **Tests are production.** Pytest fixtures target the same singletons
   the API does, so a green test exercises the same wiring a request
   does.

### 1.3 Core principles

- **Deterministic IDs.** IR node IDs and chunk IDs are content-derived
  (blake3 over path + offset + content) so two processes indexing the
  same content produce the same IDs. The IR builder's docstring is the
  authoritative explanation.
- **Streaming where it matters, batching where it doesn't.** Embedding
  is batched (`embedding/pipeline.py`); the agent loop is iterative
  (one LLM call per turn). Don't invert either.
- **Boring, well-named code wins.** A 200-line function with a clear
  `index_scan_chunk_embed_persist` name beats a 50-line abstraction
  hiding the same flow behind five small functions.

---

## 2. High-level architecture

The system has nine production packages. Each owns one concern and is
the only place that concern lives. [CODEBASE_GUIDE.md](./CODEBASE_GUIDE.md)
catalogs every directory; this chapter describes how the packages
*interact*.

```
┌────────────────────────────────────────────────────────────────────┐
│                              CLI / HTTP API                        │
│   semrag ask / semrag index        /query  /answer  /projects      │
└───────────────┬───────────────────────────┬─────────────────────────┘
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌──────────────────┐
        │  Ingestion    │           │  Query           │
        │  Service      │           │  Service         │
        └───────┬───────┘           └────────┬─────────┘
                │                            │
                ▼                            ▼
    ┌────────────────────────────────────────────────────────┐
    │  Ingestion package     Retrieval package   Reasoning   │
    │  (scanner, parsers,    (vector + lexical   package      │
    │   ir, chunking,        + graph expansion   (llm,       │
    │   embedding)            + reranking)       agents,     │
    │                                              prompting)│
    └────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Storage package         │
                    │  (postgres, vector,      │
                    │   graph)                 │
                    └─────────────────────────┘
                                ▲
                                │
                    ┌───────────┴───────────┐
                    │  Jobs package          │
                    │  (worker pulls from    │
                    │   the Postgres queue)  │
                    └─────────────────────────┘
```

Three characteristics of this shape that are non-obvious and matter:

- **Ingestion is upstream of retrieval.** Ingestion writes IR nodes,
  edges, and embedded chunks to Postgres. Retrieval reads from there.
  There is no second source of truth.
- **Reasoning depends on retrieval, not the other way around.** The
  agent loop calls `search_codebase`, which routes through the same
  `HybridRetriever` the synchronous query path does. LLM behavior is
  bounded by what retrieval can find.
- **Jobs depend on ingestion, not the reverse.** The `jobs` package
  exists so the HTTP API can return immediately on `POST /projects/{id}/index`
  while a worker runs the actual pipeline. The CLI bypasses jobs and
  calls `IngestionService.index()` synchronously.

For the data-flow diagrams and the related tasks API, see the
"End-to-end …" chapters below.

---

## 3. End-to-end request lifecycle

A request takes one of two paths:

- **CLI**: `python run_local_qa.py ask ...` → `semrag.cli.main()` →
  builds an `IngestionService` or `QueryService` directly → synchronous
  result.
- **HTTP**: `POST /projects/{project_id}/query` or `/answer` →
  `api/routers/query.py` or `answer.py` → `QueryService.ask()` →
  result → `QueryResponse` / `AnswerResponse` Pydantic model.

Both paths share the same `QueryService.ask()` method, so behavior is
identical.

### 3.1 The seven stages of a query

The diagrams below show what happens inside `QueryService.ask()`
(`src/semrag/services/services.py`). For a graphical view, see the
"End-to-end query data flow" diagram in [ARCHITECTURE.md](../ARCHITECTURE.md).

1. **Structural detection.** `is_structural_query(question)` decides
   whether the question is "summarize this repo / list its files"
   (structural) or a normal query. Structural queries are seeded with
   the project's FILE INDEX before any retrieval, so the LLM can
   ground structural claims in real paths.

2. **Query embedding.** `embedding_provider.embed_texts([question])`
   produces one vector. The provider is selected by
   `embedding/factory.py:build_embedding_provider()` from
   `SEMRAG_EMBEDDING_PROVIDER`.

3. **Retrieval.** A `HybridRetriever` runs, in this order:

   a. Vector search via `PgvectorStore` for the top
      `SEMRAG_VECTOR_TOP_K` chunks.
   b. Lexical search via `PostgresLexicalSearch` for the top
      `SEMRAG_LEXICAL_TOP_K` chunks.
   c. Query expansion (RI-05, `retrieval/query_expansion.py`): up to
      `SEMRAG_QUERY_EXPANSION_MAX_EXPANSIONS` additional lexical
      searches (original query + intent-vocabulary terms), folded into
      the same RRF as new `lexical_expN` signals. Deterministic,
      zero-network, fail-open.
   d. Reciprocal Rank Fusion (`retrieval/fusion.py`) merges all the
      lists (vector + lexical + expansions) into a single fused
      ranking.
   e. Noisy-source penalty — chunks whose `relative_path` matches
      `SEMRAG_NOISY_SOURCE_PATH_TERMS` (default `/log/`) get multiplied
      by `SEMRAG_NOISY_SOURCE_PENALTY`. Simulation logs systematically
      over-match natural-language queries otherwise.
   f. Graph expansion over the IR (when
      `SEMRAG_ENABLE_GRAPH_EXPANSION` is true) appends structurally
      adjacent nodes; their score is penalized and depth-decayed by
      `ranking/graph.py`.
   g. Intent boost (when `SEMRAG_ENABLE_INTENT_BOOST` is true):
      `apply_intent_ranking` (`retrieval/ranking/intent.py`) multiplies
      each chunk's fused score by an intent-derived factor.
   h. Source authority (RI-02, `retrieval/source_authority.py`):
      document-aware nudge by `(chunk_type, relative_path)`, applied
      after intent boost and before the reranker.
   i. Reranking via `Reranker` (`retrieval/reranking.py`) when
      `SEMRAG_RERANKER` is `lexical_overlap` or `http`. `none` skips.
      Fail-open (RI-07): a reranker that raises falls back to the
      pre-reranker ranking.
   j. Diversity-aware selection (RI-03, `retrieval/ranking/diversity.py`):
      picks a compositionally diverse top-k by selection, bounded by
      `SEMRAG_RETRIEVAL_DIVERSITY_MAX_PER_FILE` /
      `_MAX_PER_CATEGORY` / `_NEIGHBOUR_WINDOW` / `_POOL_MULTIPLIER`.
   k. Stable top-k sort (deterministic tie-break on `chunk_id`),
      truncated to `SEMRAG_FINAL_TOP_K`.
   l. Hybrid top-k reservation (`HYBRID_RESERVATION_SLOTS = 2` in
      `retrieval/pipeline.py`) preserves a slot for chunks that won
      one channel by a large margin but not the other.

4. **Prompt assembly.** `PromptBuilder.build()`
   (`reasoning/prompting/pipeline.py`) allocates token budget in a
   specific order — see the docstring there:

   - Total budget = `LLM_CONTEXT_WINDOW - LLM_RESERVED_FOR_RESPONSE`
   - System prompt + history are subtracted **first** (non-negotiable).
   - Remaining budget goes to retrieved context, dropping the lowest-
     scored chunks first when tight.
   - History itself is bounded by `MAX_HISTORY_TOKENS`; oldest turns are
     dropped first (sliding window).

5. **Agent loop.** `AgentLoop.run()`
   (`reasoning/agents/agent_loop.py`) drives a ReAct-style loop with a
   hard iteration cap (`DEFAULT_MAX_ITERATIONS = 8`):

   a. Call `LLMProvider.complete(messages, tools=tool_schemas)`.
   b. If the result has no tool calls → it's a final answer; return it.
   c. Otherwise, execute each tool via `ToolRegistry.execute(name,
      arguments)` (default tools: `search_codebase`, `list_files`,
      optionally `read_file` if `project_root` is configured).
   d. Append tool results as `tool`-role messages; loop.
   e. If the cap is hit, return `AgentResult(hit_iteration_limit=True,
      final_answer=None)`; the caller decides how to surface that.

   Tool-call messages with non-empty prose from the model are
   *intentionally* stripped of content — small models often emit
   garbage prose alongside a valid tool call, and replaying that prose
   back into the next turn's context derails the next completion. The
   prose-before-tool-call is noise; the tool call is the intent.

6. **Citation validation.** After the agent returns, the response is
   checked against retrieved chunks. `validate_citations(response,
   included_chunks)` (`evaluation/citation_validator.py`) parses every
   `[Source: path, lines X-Y]` reference and confirms the path was in
   the context. The result exposes
   `CitationValidationResult.has_fabricated_citations` and
   `.fabrication_rate` so it can be surfaced as a metric and inspected
   in the evaluation harness.

7. **Response shaping.** Final `QueryResponse` / `AnswerResponse`
   carries the answer text, included chunks (with `relative_path`,
   `lines`, scores), and a `DebugSummary` populated from
   `QueryService._summarize_*` helpers.

### 3.2 Streaming

The `/query/stream` and `/answer/stream` routes emit the same result
incrementally via Server-Sent Events. The streaming path is *not* a
separate agent; it is the same `AgentLoop.run()` whose iterations are
yielded. There is exactly one source-of-truth loop in the codebase.

### 3.3 Caching

`QueryResponse` is cached against a content hash
(`api/query_cache.py:cache_key`) when a `QueryCache` is configured.
Cache hits short-circuit stages 1–7 and return the cached response with
`from_cache=True`. The cache is per-project; `invalidate_project()`
clears it precisely rather than nuking the whole cache, so a re-index
of one project does not invalidate another project's cached queries.

The rate limiter (`api/rate_limit.py`) is independent: it accepts or
rejects the request *before* the cache lookup.

---

## 4. End-to-end indexing lifecycle

Indexing is one-way and stage-bounded. See the "End-to-end indexing
data flow" diagram in [ARCHITECTURE.md](../ARCHITECTURE.md) for the
graphical view; this chapter describes each stage in execution order.

`IngestionService.index()`
(`src/semrag/services/services.py`):

1. **Stage: scan** (`ingest.scan`).
   - `Scanner.scan(project_id, root_path)`
     (`ingestion/scanner/orchestrator.py`).
   - Walks the filesystem, hashes each file (thread-pool-bounded by
     `max_hash_workers` — see the orchestrator docstring for the
     NFS-friendly reasoning), collects git metadata, extracts
     dependency references.
   - Compares against the previous manifest via
     `change_detector.detect_changes()` and returns a `ChangeSet`
     (`added` / `modified` / `deleted_paths`).
2. **Stage: parse** (`ingest.parse`).
   - For each file in the manifest, the parser is chosen by the
     `ParserRegistry` (`ingestion/parsers/pipeline.py`). The registry
     dispatches by language / extension to the right `Parser`
     implementation: SystemVerilog, VHDL, Python/Tcl, Excel, PDF,
     Markdown, regression logs, structured data.
   - Each parser returns a parse-result object normalized to a common
     shape, which the IR builder can consume.
3. **Stage: IR** (`ingest.ir`).
   - `IRBuilder().build(project_id, manifest.files, parse_results)`
     (`ingestion/ir/builder.py`) runs a two-pass build producing
     nodes (`Module`, `Class`, `Function`, `Assertion`, …) and edges
     (`calls`, `instantiates`, `imports`, …).
   - Node IDs are deterministic (blake3 over content) so re-indexing
     produces stable IDs.
4. **Stage: chunk** (`ingest.chunk`).
   - `chunk_project(...)` (`ingestion/chunking/pipeline.py`) selects a
     chunker per language via `Chunker` Protocol — Python uses an
     AST-aware chunker, SystemVerilog uses an HDL-aware chunker,
     Markdown gets a heading-aware chunker, log files get an
     event-aware chunker (RI-04, `chunking/log_chunker.py`), and
     everything else falls back to `GenericWindowChunker`.
5. **Stage: embed** (`ingest.embed`).
   - `embed_chunks(chunks, provider, cache)`
     (`ingestion/embedding/pipeline.py`) batches chunks through
     `EmbeddingProvider.embed_texts(texts)` with cache lookups on
     both sides of the call. Cache hits are short-circuited.
   - The cache backend is selected by `build_embedding_cache()` in
     `ingestion/embedding/factory.py`. Memory / JSON-file / Redis.
6. **Stage: persist** (`ingest.persist`).
   - `IRGraphRepository(session).save(ir_graph)` writes nodes + edges
     to Postgres (SQL defined by `alembic/versions/*`).
   - `PgvectorStore(session).upsert_chunks(vector_chunks)` writes
     embedded chunks to the pgvector-backed table.
   - Both happen inside a single SQLAlchemy session so a failure
     halfway rolls back the whole index run.
7. **Stage: cleanup**.
   - `DebugArtifactWriter.run_completed()` closes the per-run
     artifact bundle under `.semrag/debug/<run_id>/`.

When called from the CLI (`semrag index ...`), the same `index()`
returns an `IndexResult` once all stages finish. When called from the
HTTP API, the index stage enqueues a job and the pipeline runs in the
worker process — see the next chapter.

### 4.1 The job system

Indexing through the HTTP API is asynchronous:

- `POST /projects/{project_id}/index` creates a `Job` in `JobState.QUEUED`
  via `JobManager.create_job(...)` and returns a `job_id`.
- A separate `worker` process (the `semrag-worker` entrypoint, see
  `src/semrag/worker.py` and `src/semrag/jobs/worker.py`) polls for
  `QUEUED` jobs, atomically transitions them to `RUNNING`, and calls
  `IngestionService.index()` inside the job.
- The client polls `GET /projects/{project_id}/jobs/{job_id}` for
  progress (`progress`, `current_stage`, `statistics`).

When `SEMRAG_JOB_STORE_BACKEND=database` (production default), the queue
is durable across restarts. When `=memory`, it is single-replica —
appropriate for dev/test only; explicitly called out in
`docs/operations/` and `ARCHITECTURE.md`.

---

## 5. Dependency injection

There are two distinct styles in this codebase, deliberately:

- **Service classes** (`IngestionService`, `QueryService`) take every
  collaborator as a constructor argument. They are constructed at the
  edge — `cli.py` for the CLI, the dependency providers in
  `api/dependencies.py` for the API — and passed their session, LLM,
  embedding provider, settings, and so on. Tests construct them with
  in-memory fakes; production constructs them with the real backed
  singletons.
- **FastAPI dependencies** are wired via `Depends(...)` and resolved
  per request. The cross-cutting concerns (DB session, rate limiter,
  query cache, job manager, executor, embedding provider) are
  singletons imported once in `api/dependencies.py` and reused.

Three concrete consequences:

1. **Singletons live at module scope.** `_engine`, `_session_factory`,
   `_job_store`, `_job_manager`, `_job_executor`, the shared cache,
   and the rate limiter are all created at import time. This is why
   `dependencies.py` has `reset_job_store_for_tests()` and why
   `query_cache.py` has `reset_shared_cache()`: tests need to swap
   settings between cases and the module-scope singletons must be
   re-bound.
2. **Factories are the only place that reads `Settings` for
   construction.** `build_embedding_provider`, `build_llm`,
   `build_session_factory`, `require_redis_client` are the chokepoints.
   When you change which backend is selected, you change it via
   `Settings` + factory; never instantiate a backend directly inside a
   route handler.
3. **Sessions are short-lived.** `get_db_session()` is a per-request
   generator: yield → commit on success → rollback on exception →
   close. Anything held across requests should live on the application
   state (`app.state`), not on a session.

### 5.1 Settings

`Settings` (`src/semrag/core/settings.py`) is a mutable pydantic
singleton, loaded once via `get_settings()` (lru-cached). Tests call
`reset_settings_cache()` after monkeypatching env vars so the next read
reflects them. Every pydantic field has a `SEMRAG_*` env-var prefix and
`.env` support baked in.

The model validator at the bottom of `settings.py` enforces a few
combinations the field types can't express on their own — most
importantly, **`dev-local-key` is rejected when `SEMRAG_ENVIRONMENT=production`**.

### 5.2 Factories

Three factories are the only entry points for backend selection:

| Factory                                                              | What it builds                               |
| -------------------------------------------------------------------- | -------------------------------------------- |
| `build_embedding_provider(settings)`                                 | `EmbeddingProvider` implementation           |
| `build_embedding_cache(settings)`                                    | `EmbeddingCache` backend                      |
| `build_llm(settings)`                                                | `LLMProvider` implementation                 |
| `require_redis_client()` / `require_async_redis_client()`            | sync / async Redis clients; fail-fast        |
| `build_engine(settings)` / `build_session_factory(engine, settings)` | SQLAlchemy engine and sessionmaker           |

There is no dynamic registry, no plugins directory, no entry-point
discovery. Each factory is a plain function you read top-to-bottom.

---

## 6. Storage layer

Three persistence planes, each behind a Protocol so the rest of the
codebase does not bind to a concrete backend.

### 6.1 Postgres (`storage/postgres/`)

- `session.py` — engine + sessionmaker. Pool sizing is controlled by
  `SEMRAG_DB_POOL_SIZE`, `SEMRAG_DB_MAX_OVERFLOW`,
  `SEMRAG_DB_POOL_RECYCLE`, and `SEMRAG_DB_POOL_TIMEOUT`. Migrations
  live in `alembic/versions/` (one file per revision).
- `ir_repository.py` — IR node + edge CRUD.
- `manifest_store.py` — `PostgresManifestStore`, the durable scan-
  manifest backend used by the scanner.
- `diagnostics_repository.py` — diagnostics / run-history queries.
- `graph_queries.py` — graph-shaped SQL queries over the IR tables;
  used by retrieval's graph-expansion path when
  `SEMRAG_ENABLE_GRAPH_EXPANSION=true`.

### 6.2 Vector store (`storage/vector/`)

- `interfaces.py` — `VectorStore` Protocol.
- `pgvector_store.py` + `pgvector_models.py` — pgvector implementation;
  embed-and-upsert a chunk row with its embedding in one call. The
  search-time entry point is `vector_search_adapter.to_retrieval_results(...)`
  which converts vector rows to the retrieval-stage `RetrievalResult`.

The Protocol lets a future `QdrantStore` (or any equivalent) drop in
without changes elsewhere; the `Document`/`parse`/`chunk` planes only
ever see `VectorStore`.

### 6.3 Graph store (`storage/graph/`)

- `interfaces.py` — `GraphStore` Protocol.
- `neo4j_store.py` — Neo4j adapter. **Disabled by default**;
  experimental. Not wired into retrieval — `graph_expansion.py` reads
  directly from the in-process `PostgresGraphQueries` (see the
  "Skipped" note in the retrieval README for why).

### 6.4 Caches and rate limiter (`api/`)

- `query_cache.py` — `QueryCache` ABC with `TTLQueryCache` (per-process,
  in-memory) and `RedisQueryCache` (distributed). Backend selection via
  `SEMRAG_QUERY_CACHE_BACKEND`.
- `rate_limit.py` — `RateLimiter` ABC with `MemoryRateLimiter` (sliding
  window per process) and `RedisRateLimiter` (sliding window backed by
  a Redis sorted set + Lua script for atomicity). Backend selection via
  `SEMRAG_RATE_LIMIT_BACKEND`.

### 6.5 Redis integration (`api/redis_client.py`)

The `redis_client.py` module deliberately exposes two flavors:

- `get_redis_client()` / `require_redis_client()` — sync. Used by the
  query cache (sync ABC) and the embedding cache (sync Protocol).
- `get_async_redis_client()` / `require_async_redis_client()` — async
  (`redis.asyncio.Redis`). Used by the rate limiter whose methods are
  declared `async def`.

The `require_*` variants raise `RuntimeError` if `SEMRAG_REDIS_URL` is
empty — the **no-silent-fallback rule**. An operator who wants
best-effort should pick `backend=memory`; selecting `redis` means "I
require the distributed property, do not fall back."

### 6.6 The cache + DB transaction model

The query cache lives *outside* any DB transaction and is best-effort.
A DB rollback does not invalidate the cache; that is intentional —
`invalidate_project()` is wired to re-index flows so an explicit
invalidation runs at the right moment. Do not invent a
`cache.invalidate_on_rollback`; the architecture doesn't need it.

---

## 7. Background jobs

`semrag.jobs` exists so the HTTP API can return immediately on indexing
requests. Five files, each with one responsibility:

- `models.py` — `Job`, `JobState` (`QUEUED` / `RUNNING` / `DONE` /
  `FAILED` / `CANCELLED`), `IndexingJobRow`.
- `store.py` — `JobStore` ABC with `InMemoryJobStore` (single-replica
  convenience) and `PostgresJobStore` (durable, default).
- `postgres_store.py` — durable implementation.
- `manager.py` — `JobManager` exposes create / update / fetch /
  list jobs.
- `executor.py` — `JobExecutor` ABC, `ThreadPoolJobExecutor`
  implementation, `ExecutorFactory.create_executor(settings)`.
- `pipeline.py` — `PipelineStage` and `ProgressCalculator` for
  reporting per-stage progress back to the store.
- `worker.py` — `indexing_worker(...)`; the actual entry point run by
  the `semrag-worker` process.

### 7.1 State machine

```
QUEUED ──claim──> RUNNING ──ok──> DONE
                      │
                      └─error──> FAILED
                      └─cancel──> CANCELLED
```

The transition `QUEUED → RUNNING` is atomic in `PostgresJobStore` (one
SQL `UPDATE ... RETURNING`). Two workers polling at the same instant
will see at most one of them succeed. This is the
claim-in-a-single-roundtrip pattern; do not duplicate the transition
across worker code.

### 7.2 Cancellation

A job can be cancelled while `QUEUED` or `RUNNING` via
`POST .../jobs/{id}` (DELETE in the API). The check is a polling flag
read by the worker between stages; mid-stage cancellation is **not**
implemented — the worker completes the current stage, observes the
cancelled flag, and exits before starting the next. Don't pretend a
mid-stage cancel is supported; it would break invariants in
chunking / embedding batches.

---

## 8. Security model

### 8.1 Authentication

`require_api_key` (`api/auth.py`) is the single dependency that
validates the `Authorization: Bearer …` header and returns the
`APIKey` object. The bearer is hashed once on entry via
`hash_api_key` (blake3 over a domain separator; fixed-cost) and the
store lookup is constant-time by construction. `verify_api_key` is a
second constant-time check after fetch — defense-in-depth against any
future store implementation that might regress to a substring match.

`APIKey.is_valid()` checks expiry and status (revoked / suspended);
`is_allowed_for_project(project_id)` is the per-project authorization
gate enforced by `project_auth.get_project_authorizer()`.

### 8.2 Authorization

Per-project authorization is enforced via the
`get_project_authorizer()` dependency factory (see
`api/project_auth.py`). Each protected route in
`api/routers/projects.py` and the query routers depends on it. A key
that is allowed for `projA` is rejected with `403` when accessing
`projB`. The first user-facing fix when a 403 appears is: confirm the
key's project allowlist.

### 8.3 Rate limiting

`enforce_rate_limit` (`api/rate_limit.py`) is a `Depends` that returns
or raises. The limit comes from `SEMRAG_RATE_LIMIT_PER_MINUTE`. Every
decision increments a Prometheus counter
(`semrag_rate_limit_accepted_total` /
`semrag_rate_limit_rejected_total`).

### 8.4 Transport-level hardening

`api/middleware.py`:

- `RequestIDMiddleware` — assigns an `X-Request-ID` to every request
  and exposes it on `request.state.request_id`. All log records and
  error responses carry it.
- `TracingMiddleware` — extracts W3C trace context, starts OTel spans.
- `MetricsMiddleware` — `observe_http(...)` for request metrics.
- `LoggingMiddleware` — structured access log.
- `SecurityHeadersMiddleware` — adds the headers from
  `SEMRAG_SECURITY_HEADERS`.
- `MaxBodySizeMiddleware` — enforces `SEMRAG_MAX_BODY_SIZE`.

### 8.5 Production defaults

- `SEMRAG_ENVIRONMENT=production` rejects the default `dev-local-key`
  in `Settings._check_dev_keys_after()`. The operator must set
  `SEMRAG_API_KEYS` for production.
- `TRUSTED_HOSTS` and `CORS_ORIGINS` must be explicitly populated;
  empty defaults are safe but restrictive.
- The default `JobStore` is Postgres. `dev` mode keeps an in-memory
  store available but requires the operator to opt in
  (`SEMRAG_JOB_STORE_BACKEND=memory`); a production deploy that boots
  into `memory` silently loses queued jobs on restart — the class of
  bug the durable store shipped to fix.

### 8.6 What SemRAG does not enforce

- **No request-body parsing for sandboxing.** Body-size limits are
  enforced at the middleware; otherwise bodies are trusted to the
  Pydantic schema in the router.
- **No fine-grained rate limiting per route.** One global limit per
  key. Adding per-route or per-tool limits is a future seam.
- **No built-in audit log of who asked what.** Per-request logging is
  the access log only. If you need an audit trail, the
  `QueryTrace` accumulator (`core/query_trace.py`) is the right shape.

---

## 9. Observability

Three orthogonal observability systems exist; treat them as one.

### 9.1 Logging (`core/observability.py`)

`configure_logging(settings)` is called once at startup. Two formatters
(`human`, `json`) selected by `SEMRAG_LOG_FORMAT`. Level controlled
by `SEMRAG_LOG_LEVEL` (`TRACE` / `DEBUG` / `INFO` / `WARNING` /
`ERROR` / `CRITICAL`). Stages in the indexing and query paths use the
`stage(name, logger=..., level=...)` context manager; every stage
emits a `start` and `finish` log record with elapsed time.

`redaction.py` masks API keys and other secrets in any log line or
payload. The exception class in `exception_handler.py` runs through
redaction before serializing.

### 9.2 Metrics (`core/metrics.py`)

A process-global `CollectorRegistry` is created at module import.
Helpers in `metrics.py`:

- `observe_http`, `inc_http_exception`
- `observe_retrieval(stage, duration_s)`
- `observe_rerank(duration_s)`
- `observe_embedding(provider, duration_s)`
- `observe_llm(provider, model, duration_s)`
- `observe_tool(tool, duration_s)`
- `observe_db_query(op, duration_s)`, `inc_db_transaction(op)`,
  `set_db_pool(state, value)`
- `inc_cache_hit(backend)`, `inc_cache_miss(backend)`,
  `inc_cache_eviction(backend)`
- `inc_rate_limit_accepted()`, `inc_rate_limit_rejected()`
- `set_uptime(seconds)`, `set_health_up(component, healthy)`

Every helper uses labels so a single counter (`inc_cache_*`) fans out
by backend (`memory`, `redis`).

The default registry is rebuilt only via `reset_registry()`, called
from tests. Production never calls it. The `/metrics` endpoint
(`/metrics_path`, default `/metrics`) returns the rendered registry in
the Prometheus text exposition format.

### 9.3 Tracing (`core/tracing.py`)

When `SEMRAG_TRACING_ENABLED=true`, the OpenTelemetry tracer is
initialized and an OTLP exporter is created (gRPC by default; HTTP
when the URL starts with `http+`). A no-op span + no-op span context
take its place when tracing is off — call sites do not branch on the
tracing flag.

Sampling ratio from `SEMRAG_TRACE_SAMPLE_RATIO`. 1.0 = every span, 0.1
= 10% (production-realistic), 0 = no spans even when enabled (kept
distinct so the feature toggle and the sampling ratio are independent).

### 9.4 Debug artifacts (`core/debug_artifacts.py`)

`DebugArtifactWriter` is opt-in: only active when `SEMRAG_LOG_LEVEL`
is `DEBUG` or `TRACE`. On an active run, it writes:

```
.semrag/debug/<run-id>/
    scan/manifest.json
    parse/<source-file>.parse.json
    ir/graph.json
    chunks/chunk_metadata.json
    embeddings/embedding_stats.json
    knowledge_graph/graph_summary.json
    retrieval/ranking_trace.json
    prompt/final_prompt.txt
    llm/request.json
    llm/response.json
    timings/pipeline.json
```

Retention defaults to 5 most-recent runs (`SEMRAG_DEBUG_ARTIFACT_RETENTION`).
This is the single most useful tool for a maintainer debugging a
production query; the README has the artifact layout.

### 9.5 Query trace (`core/query_trace.py`)

Per-query accumulator (`QueryTrace`) with stage-level timing and
metadata. Populated by `QueryService.ask()` when debug artifacts are
active. Useful for offline analysis of latency regressions.

---

## 10. Configuration

`Settings` is the only source of truth for runtime configuration. Every
field has a `SEMRAG_*` env-var prefix.

Three buckets of settings, with their dominant concerns:

| Bucket      | Where to start                                                                    |
| ----------- | --------------------------------------------------------------------------------- |
| Database    | `database_url`, `db_pool_*`                                                       |
| Retrieval   | `vector_top_k`, `lexical_top_k`, `final_top_k`, `rrf_k`, `reranker`, `enable_*`   |
| Operations  | `api_keys`, `rate_limit_per_minute`, `query_cache_backend`, `embedding_cache_backend`, `redis_url`, `job_store_backend`, `environment`, `*_store` |

Cross-cutting checks (combinations the field types can't express) are
enforced by `startup_validation.run_startup_validation()` at process
boot. It is called once from `app.py` lifespan; deployment that ships
to production without that startup validation step is misconfigured.

### 10.1 When a setting changes meaning

Two settings have changed semantics historically:

- `SEMRAG_RATE_LIMIT_PER_MINUTE` defaults to 60. Lower it for shared
  keys; raise it only for service accounts you control.
- `SEMRAG_REDIS_URL` — leaving it empty with a Redis backend selected
  is an error, not a fallback. This is intentional.

---

## 11. Error handling

Three layers:

1. **Pydantic models** validate request bodies and query parameters.
   Failures raise `RequestValidationError` and are mapped to
   `422 Unprocessable Entity` by `exception_handler.validation_exception_handler`.

2. **HTTP exceptions** raised by router code (`HTTPException(...)`)
   are mapped by `exception_handler.http_exception_handler` to a
   consistent envelope:

   ```json
   {
     "error": {
       "code": "HTTP_404",
       "message": "Project 'foo' not found",
       "request_id": "<uuid>",
       "details": {}
     }
   }
   ```

3. **Unhandled exceptions** are caught by `exception_handler.general_exception_handler`
   and returned as a `500 Internal Server Error` envelope with a
   **generic message** in production. Stack traces never leak to the
   client; the underlying exception is logged with its
   `request.state.request_id` for correlation.

### 11.1 The rule

> Catch only what you can recover from. Let everything else reach the
> generic handler so the envelope is consistent and the log gets the
> full traceback.

Re-raising inside a `Depends` is fine; catching broadly inside a router
to translate to a domain message is *not* — the centralized handler is
where translation belongs.

---

## 12. Testing strategy

Three test trees; one philosophy.

### 12.1 Layout

- `tests/unit/` — pytest. Mirrors `src/semrag/<package>/`. One
  `test_<module>.py` per real module.
- `tests/load/` — Locust + a custom harness for load tests against a
  running API. Manual invocation; see `tests/load/README.md`.
- `tests/perf/` — performance regression suite; see
  `tests/perf/README.md`.

### 12.2 Conventions

- **Conftest at every package boundary.** `tests/unit/conftest.py` and
  per-package `tests/unit/<pkg>/conftest.py` (where present) provide
  shared fixtures: settings overrides, in-memory stores, temporary
  paths.
- **Real classes, fakes at the edges.** Tests instantiate the real
  service (`QueryService`, `IngestionService`) with fakes for the
  embedding provider, LLM, and persistence. They do *not* stub out
  internal helpers — that hides regressions where a helper's
  contract drifted.
- **No live network in unit tests.** Anything that touches a live
  LLM / Ollama / external API is gated by an env-var skip so
  contributors without those services can still get a green CI run.
- **Async tests use `pytest-asyncio` (or `anyio`)** with the markers
  already configured in `pyproject.toml`. The rate-limiter and
  agent-loop tests run under that.

### 12.3 What tests do NOT cover

- Live Postgres / live Redis tests are integration tests; they are not
  the default run. They live separately and are documentation-as-code
  in `scripts/`.
- Performance regression thresholds are checked by `tests/perf/`,
  not the unit suite.

---

## 13. Extension points

Every place the system is intentionally pluggable.

### 13.1 Add a parser

1. Implement the `Parser` Protocol in
   `src/semrag/ingestion/parsers/<language>/parser.py`. The Protocol
   lives in `ingestion/parsers/interfaces.py` and is the only contract
   you need.
2. Register the parser with the `ParserRegistry` in
   `ingestion/parsers/pipeline.py`. Choose the trigger: extension,
   magic bytes, or a classifier.
3. Add a `parsers/<language>/README.md` only if the parser has
   non-obvious behavior worth documenting (most don't).
4. Add tests under `tests/unit/parsers/`. Follow the existing
   per-language pattern.
5. If the language requires a special chunker, see 13.2.

### 13.2 Add a chunker

1. Implement the `Chunker` Protocol in
   `src/semrag/ingestion/chunking/<chunker>.py`.
2. Register the chunker with the chunker registry in
   `ingestion/chunking/pipeline.py`.
3. Tests under `tests/unit/chunking/`.

### 13.3 Add an embedding provider

1. Implement `EmbeddingProvider` in
   `src/semrag/ingestion/embedding/<provider>.py`. The Protocol is in
   `ingestion/embedding/interfaces.py`.
2. Add the dispatch case in
   `ingestion/embedding/factory.py:build_embedding_provider()`.
3. Add the setting (provider name, model, dimension, base URL) to
   `Settings` in `core/settings.py`.
4. Tests under `tests/unit/embedding/`.

### 13.4 Add an LLM provider

1. Implement `LLMProvider` in
   `src/semrag/reasoning/llm/<provider>.py`.
2. Add the dispatch case in `reasoning/llm/factory.py:build_llm()`.
3. Add the setting to `Settings`.
4. Tests under `tests/unit/llm/`. Use `mock_llm.py` as a template for
   a deterministic fake.

### 13.5 Add a storage backend

Two protocols to fill, in two different planes:

- **Vector store** — `VectorStore` in `storage/vector/interfaces.py`;
  implement, then update the vector-store constructor selection at
  the call sites (`retrieval/vector_search_adapter.py`,
  `services/services.py`).
- **Graph store** — `GraphStore` in `storage/graph/interfaces.py`;
  implement, then (when a production deployment actually runs Neo4j)
  wire it into `retrieval/graph_expansion.py` in place of the direct
  `PostgresGraphQueries` import.

Both planes also have a database plane (Postgres). Adding a new
relational store is the same shape: implement the relevant Repository
classes against the new driver, and add a `Backend` enum value
elsewhere.

### 13.6 Add an API endpoint

1. Pick the right router — `projects.py`, `query.py`, or `answer.py`,
   or add a new one if a new resource is involved.
2. Define a request/response Pydantic model in `api/schemas.py`.
3. Add the dependency chain: `Depends(require_api_key)` for auth,
   `Depends(get_project_authorizer())` for project scoping,
   `Depends(enforce_rate_limit)` for rate limiting.
4. Add the route + a FastAPI `tags=` value for OpenAPI grouping.
5. Tests under `tests/unit/api/`. Use `TestClient` and the existing
  `tests/unit/api/conftest.py`.

### 13.7 Add a Prometheus metric

1. Declare the metric in `core/metrics.py` near the related helpers.
   Keep the metric family names consistent (`*_total` for counters,
   `_seconds` for durations).
2. Add a helper function (`def inc_...` / `def observe_...`) that calls
   `Counter(...).labels(...).inc()` or `Histogram(...).labels(...).observe(...)`.
3. Call the helper from the relevant code path. **Never** import a
   metric directly from outside `core/metrics.py`; everything goes
   through the helper so the labels stay consistent.
4. If the metric deserves an alert rule, add it in
   `deploy/monitoring/prometheus/rules/`.

---

## 14. Debugging workflow

### 14.1 Start with the log level

```bash
SEMRAG_LOG_LEVEL=DEBUG semrag index --project-id rtl_uvm --root ~/projects/RTL_UVM
```

This turns on:
- Per-stage `start`/`finish` records with elapsed time.
- The `DebugArtifactWriter`, so a per-run bundle under
  `.semrag/debug/<run-id>/` is written.

```bash
SEMRAG_LOG_LEVEL=TRACE semrag ask --project-id rtl_uvm --question "..."
```

This additionally emits:
- Every LLM request/response as `llm/request.json` +
  `llm/response.json`.
- The final prompt as `prompt/final_prompt.txt`.
- The retrieval ranking trace as `retrieval/ranking_trace.json`.

### 14.2 Symptom → first file

| Symptom                                           | Start here                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------- |
| "Indexing didn't pick up my changes"              | `ingestion/scanner/orchestrator.py` + `change_detector.py`          |
| "Parsing fails on file type X"                    | `ingestion/parsers/pipeline.py` + the specific parser               |
| "Chunks look wrong"                               | `ingestion/chunking/pipeline.py` + the specific chunker             |
| "Embeddings are slow or missing"                  | `ingestion/embedding/pipeline.py` + `embedding/factory.py`           |
| "Retrieval returns nothing"                       | `retrieval/pipeline.py` + `vector_search_adapter.py`                |
| "Reranking is wrong"                              | `retrieval/reranking.py`                                            |
| "Agent loop never terminates"                     | `reasoning/agents/agent_loop.py`                                    |
| "LLM call fails"                                  | `reasoning/llm/factory.py` + the specific LLM implementation        |
| "Prompt is malformed"                             | `reasoning/prompting/pipeline.py` + `context_assembler.py`          |
| "Citations don't match chunks"                    | `reasoning/prompting/citation.py` + `evaluation/citation_validator.py` |
| "API returns 5xx"                                 | `api/exception_handler.py` + matching router                        |
| "Rate limit fires unexpectedly"                   | `api/rate_limit.py`                                                 |
| "Query cache returns stale results"               | `api/query_cache.py`                                                |
| "Job stuck in QUEUED"                             | `jobs/worker.py` + `jobs/executor.py`                               |

### 14.3 Promote the trace to the dashboard

When the bug is in production and you have a `request_id`, the access
log line has the timing breakdown, the response body has the
[Source: …] citations, and the per-run debug artifact (when the log
level was DEBUG/TRACE for that user) has the full
`retrieval/ranking_trace.json` + `llm/request.json`. If you regularly
hit production issues, raise log level selectively for one user / one
request via a feature flag — the codebase does not ship with one yet;
the natural place to add it is in `Settings` + a small dependency in
`api/middleware.py`.

### 14.4 Reproduce a single stage in isolation

Every retrieval and ingestion stage is independently importable. To
reproduce only the vector-search step:

```python
from semrag.storage.postgres.session import build_session_factory
from semrag.storage.vector.pgvector_store import PgvectorStore
from semrag.storage.vector.models import VectorSearchFilter

session_factory = build_session_factory()
with session_factory() as session:
    store = PgvectorStore(session)
    rows = store.search(query_embedding=[...], top_k=20, filter_=VectorSearchFilter(project_id="rtl_uvm"))
```

The same works for lexical search, rerankers, and (with a mock LLM)
the agent loop.

---

## 15. Common architectural rules

### 15.1 Hard rules

- **Never instantiate a backend inside a router.** Use the factory +
  dependency. The factories are the only place that reads `Settings`
  for construction.
- **Never swallow an exception silently.** Catch only when there is a
  specific recovery; otherwise let the centralized handler return the
  error envelope.
- **Never change a Protocol's interface without updating every
  concrete implementation.** Run `rg 'class \w+\('` and verify
  before merging.
- **Never log a path that did not appear in the retrieved context.**
  That is exactly the fabricated-citation surface the citation
  validator catches.
- **Never silently fall back.** A Redis backend with no
  `SEMRAG_REDIS_URL` must raise `RuntimeError`, not serve memory.
  Same rule for any "if X fails, use Y" code path where the
  operator explicitly opted into X.

### 15.2 Architectural boundaries

These boundaries exist for a reason; don't cross them casually.

| From                  | To                     | Allowed?                                                       |
| --------------------- | ---------------------- | -------------------------------------------------------------- |
| `core`                | anything but stdlib    | **No.** `core` is the only package that has no siblings.        |
| `ingestion`           | `retrieval`            | **No.** Ingestion is upstream of retrieval; nothing inverted.  |
| `retrieval`           | `ingestion`            | Allowed at the boundary in `services.py` only.                 |
| `reasoning`           | `retrieval`            | Allowed via `search_codebase` tool — that's the whole point.   |
| `api`                 | `services`             | Allowed; that's the role of services.                          |
| `api`                 | `ingestion` / `retrieval` directly | **No.** Routes call services, not primitives.        |
| `jobs`                | `services`             | Allowed; jobs *are* the API's way of running services async.   |
| `evaluation`          | `services`             | Allowed; eval is a query-time consumer.                        |

### 15.3 Common mistakes

- **Adding a new env var but not to `Settings`.** Always add the field
  to `Settings` and read via `get_settings()`. Ad-hoc
  `os.environ.get("SEMRAG_FOO")` calls are how schema drift starts.
- **Adding a public method to a Protocol to fix a single test.** The
  Protocol is the contract; change the call site, or refactor the
  consumer to use the data the Protocol already exposes.
- **Wrapping `redis.Redis` in an `async def`.** Use the async client
  from the start; the previous sync wrapper froze the event loop
  under load (see the comment in `api/rate_limit.py`).
- **Reading `Settings` inside `__init__` of a service class.** Pass
  settings in via the constructor; the factories do the reading once.
- **Piping debug output through `print()`.** Use `logger.*` calls so
  `configure_logging` controls the destination.
- **Returning a domain error from a router.** Raise `HTTPException`;
  let `exception_handler.*` produce the envelope.

### 15.4 Best practices

- **Make invariants explicit.** Constants like
  `HYBRID_RESERVATION_SLOTS = 2` and `GRAPH_EXPANSION_PENALTY = 0.5`
  in `retrieval/pipeline.py` are named for a reason — they are knobs
  but they are *non-obvious* knobs, and the next maintainer should
  see the constant inline rather than chasing a config flag.
- **Favor small, top-to-bottom functions over abstractions.** When
  the orchestration reads in one file from top to bottom
  (`IngestionService.index()` is the canonical example), the
  maintainer is faster than when the orchestration is hidden behind
  strategy objects.
- **Document the *why* in the docstring.** Every Protocol has a
  rationale; every backend has a "why this and not that" paragraph.
  When a comment is longer than its code, the comment wins.
- **Every PR keeps the pipeline green.** The project rule "Do not
  stop after fixing the first error" applies here — a partial fix
  that breaks a test is a regression.

---

## Cross-references

- [Architecture](../ARCHITECTURE.md) — end-to-end data flow diagrams
  (some redrawn in this guide where the chapter demands walking
  through stages; cross-check the diagrams when in doubt).
- [Codebase guide](./CODEBASE_GUIDE.md) — every directory, every file
  ownership, the lookup tables for "where do I change X?".
- [Configuration index](./README.md) — documentation entry point.
- [Repository guidelines](../AGENTS.md) — what the project values;
  contribution workflow; release notes format.