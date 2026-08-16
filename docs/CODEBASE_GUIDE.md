# Codebase Guide

> **Navigation map for the SemRAG source tree.** This guide answers
> *where is X?* and *where do I change Y?*. For *what does X do?* and the
> end-to-end data flows, see [ARCHITECTURE.md](../ARCHITECTURE.md).

Intended audience: **Developers** and **Maintainers**.

---

## How to use this guide

1. Read the [Repository tree](#repository-tree) to get the high-level
   shape of the codebase.
2. Skim the [Package ownership](#package-ownership) table to learn which
   module owns which concern.
3. Use the [Where do I change X?](#where-do-i-change-x) section when you
   have a concrete task in mind.
4. Use the [Debugging starting points](#debugging-starting-points) section
   when something is broken and you don't know where to look first.

For the *why*, the *what*, and the data-flow diagrams, follow the link
into [ARCHITECTURE.md](../ARCHITECTURE.md). This document intentionally
does not repeat them.

---

## Repository tree

A high-level view of the directories that contain source, configuration,
tests, and deployment artifacts. Every entry is real and exists in this
checkout.

```
semrag/
├── README.md                  Project pitch, quick start, env cheat sheet
├── ARCHITECTURE.md            Long-form architecture and data flow
├── DEPLOYMENT.md              Local, Docker, Compose, K8s, Helm deployment
├── AGENTS.md                  Contribution policy for AI / human agents
├── CHANGELOG.md               Release notes
├── CLAUDE.md                  Claude-Code-specific instructions
├── RUN_API.md                 Local HTTP API launch recipe
├── SETUP_POSTGRES.md          Local Postgres + pgvector recipe
│
├── run_local_qa.py            CLI entry point (also exposed as `semrag`)
├── stagewise_index.py         Stand-alone stagewise indexing runner
├── run_benchmarks.py          Evaluation benchmark driver
│
├── pyproject.toml             Package metadata + tool config (ruff, mypy, black)
├── requirements.txt           Pinned runtime dependencies
├── requirements-dev.txt       Pinned dev / test dependencies
├── requirements-dev.in        Unpinned dev dependencies (compiled by pip-compile)
├── Dockerfile                 Multi-stage production image
├── docker-compose.yml         Local Compose stack
├── docker-compose.override.yml
├── Makefile                   Common developer commands
├── alembic.ini                Alembic migration configuration
├── alembic/                   Postgres schema migrations (one file per change)
│
├── scripts/                   Operational scripts (e.g. benchmark_100_questions.py)
├── docs/                      Documentation tree (this file lives here)
│   └── operations/            On-call runbooks (one per failure mode)
│
├── src/semrag/                All production code
│   ├── core/                  Cross-cutting primitives (settings, logging, metrics)
│   ├── services/              Business orchestration (Ingestion/Query services)
│   ├── ingestion/             Indexing pipeline (one-way: scan → embed)
│   ├── retrieval/             Query pipeline (hybrid retriever)
│   ├── reasoning/             LLM agent loop
│   ├── storage/               Persistence adapters (postgres, vector, graph)
│   ├── api/                   FastAPI surface (routers, auth, rate limit)
│   ├── jobs/                  Async indexing-job runner
│   ├── evaluation/            Eval harness + citation validator
│   ├── cli.py                 Wraps the CLI entry point
│   └── worker.py              Stand-alone worker entry point
│
├── tests/                     Test suite
│   ├── unit/                  Pytest unit + integration tests
│   ├── load/                  Locust + custom load-test scripts
│   └── perf/                  Performance regression suite
│
└── deploy/                    Deployment manifests
    ├── helm/semrag/           Helm chart (templates + values.yaml)
    ├── k8s/                   Plain Kubernetes manifests (apply in numerical order)
    └── monitoring/            Prometheus + Grafana + Alertmanager configs
```

The source tree is split into nine production packages under
`src/semrag/`. Each is owned by one team-mental-model concern and is
covered in detail below.

---

## Package ownership

| Package                                     | Owns                                                                  | Entry file                |
| ------------------------------------------- | --------------------------------------------------------------------- | ------------------------- |
| `semrag.core`                               | Settings, logging, metrics, tracing, redaction, debug artifacts       | `core/settings.py`        |
| `semrag.services`                           | Top-level orchestration; composes ingestion + retrieval + reasoning   | `services/services.py`    |
| `semrag.ingestion`                          | Indexing pipeline (scan → parse → IR → chunk → embed)                 | `ingestion/scanner/orchestrator.py` |
| `semrag.retrieval`                          | Hybrid retrieval (vector + lexical + graph expansion + rerank)        | `retrieval/pipeline.py`   |
| `semrag.reasoning`                          | LLM-facing layers (LLM provider, agent loop, prompting)               | `reasoning/agents/agent_loop.py`    |
| `semrag.storage`                            | Postgres + pgvector + (optional) Neo4j adapters                       | `storage/postgres/session.py`        |
| `semrag.api`                                | FastAPI app, routers, auth, rate limit, query cache, middleware       | `api/app.py`              |
| `semrag.jobs`                               | Async job runner (model, store selection, worker loop)                | `jobs/worker.py`          |
| `semrag.evaluation`                         | Eval harness, citation validator, metrics, benchmarks                 | `evaluation/harness.py`   |

Sub-package ownership:

| Sub-package                            | Owns                                                       |
| -------------------------------------- | ---------------------------------------------------------- |
| `ingestion.scanner`                    | Walk, hash, classify, change-detect, manifest store        |
| `ingestion.parsers`                    | Language-specific file → parse-result pipeline             |
| `ingestion.ir`                         | Two-pass IR builder, deterministic node IDs, symbol tables |
| `ingestion.chunking`                   | Per-language chunking strategies                           |
| `ingestion.embedding`                  | Embedding providers, caches, batched pipeline              |
| `reasoning.llm`                        | `LLMProvider` Protocol + Ollama / OpenAI / vLLM / mock impls |
| `reasoning.agents`                     | ReAct agent loop + tool registry                            |
| `reasoning.prompting`                  | System prompt, templates, context assembler, citation      |
| `storage.postgres`                     | SQLAlchemy session, IR/manifest/diagnostics repositories  |
| `storage.vector`                       | pgvector adapter + `VectorStore` Protocol                   |
| `storage.graph`                        | Neo4j adapter + `GraphStore` Protocol                      |

---

## `src/semrag/` — package by package

### `src/semrag/core/`

Cross-cutting primitives. Imported by every other package; nothing in
`core` imports from sibling packages.

| File                  | Responsibility                                                       |
| --------------------- | -------------------------------------------------------------------- |
| `settings.py`         | `Settings` (pydantic-settings); single source of truth for env vars. |
| `observability.py`    | Stage logger, structured metrics, per-stage tracing.                |
| `metrics.py`          | Prometheus metric registration and helpers.                          |
| `tracing.py`          | OpenTelemetry tracer setup; no-op when tracing is disabled.         |
| `performance.py`      | `timer(...)` / `stage(...)` context managers used by pipelines.    |
| `query_trace.py`      | Per-query trace accumulation + dump.                                 |
| `redaction.py`        | PII / secret redaction for log lines and payloads.                  |
| `debug_artifacts.py`  | Optional per-run debug artifact writer with retention.               |

Belongs here: anything used by ≥ 2 other packages and nothing
domain-specific.

Does NOT belong here: pipeline orchestration, models, settings
subclasses for individual packages.

---

### `src/semrag/services/`

Thin orchestration layer. The two services compose the lower-level
packages into end-to-end flows.

| File           | Responsibility                                                  |
| -------------- | --------------------------------------------------------------- |
| `services.py`  | `IngestionService.index()` and `QueryService.ask()`.            |

`IngestionService` wires scanner → parsers → IR → chunking → embedding.
`QueryService` wires retrieval → reasoning (prompt assembly + agent
loop).

Belongs here: cross-package composition only.

Does NOT belong here: parsing, chunking, embedding, retrieval, or
prompting logic — those live in their respective packages.

---

### `src/semrag/ingestion/`

The one-way indexing pipeline. New code lands here when it touches the
path from source files to persisted chunks.

```
ingestion/
├── scanner/      walker, hashing, classification, manifest, change detection
├── parsers/      language-aware file → parse-result pipeline
│   ├── systemverilog/
│   ├── vhdl/
│   ├── python_tcl/
│   ├── documents/        (excel, pdf, markdown)
│   ├── structured_data/
│   └── logs/             (regression logs)
├── ir/           IR builder, models, symbol tables
├── chunking/     language-aware chunkers + pipeline
├── embedding/    providers, caches, batched embedding pipeline
└── streaming.py  streaming write paths
```

Key files:

| File                                        | Responsibility                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `scanner/orchestrator.py`                   | `IngestionOrchestrator` — entry point for the indexing pipeline.       |
| `scanner/walker.py`                         | Filesystem walker.                                                      |
| `scanner/hashing.py`                        | Content hashing utilities.                                              |
| `scanner/manifest_store.py`                 | Persisted file-manifest (in-memory / file-backed variants).             |
| `scanner/change_detector.py`                | Diff against the last manifest to decide what to re-process.           |
| `scanner/classification.py`                 | File → language classifier.                                             |
| `scanner/git_metadata.py`                   | Best-effort git log enrichment per file.                                |
| `scanner/dependency_tracker.py`             | Import / include dependency extraction across files.                   |
| `parsers/interfaces.py`                     | `Parser` Protocol that every language parser implements.               |
| `parsers/pipeline.py`                       | Selects a parser per file via the registry.                             |
| `ir/builder.py`                             | Two-pass IR builder producing nodes + edges.                           |
| `ir/models.py`                              | IR data model.                                                          |
| `ir/symbol_tables.py`                       | Per-file symbol resolution helpers.                                    |
| `chunking/interfaces.py`                    | `Chunker` Protocol.                                                     |
| `chunking/pipeline.py`                      | Selects a chunker per language.                                         |
| `chunking/generic_window_chunker.py`        | Fallback window chunker.                                               |
| `chunking/log_chunker.py`                   | Event-aware chunker for log files (RI-04).                              |
| `chunking/markdown_chunker.py`              | Heading-aware chunker for Markdown docs.                               |
| `chunking/python_chunker.py`                | AST-aware Python chunker.                                              |
| `chunking/systemverilog_chunker.py`         | HDL-aware chunker.                                                      |
| `embedding/interfaces.py`                   | `EmbeddingProvider` and `EmbeddingCache` Protocols.                    |
| `embedding/factory.py`                      | Backend selection (`build_embedding_provider`, `build_embedding_cache`). |
| `embedding/pipeline.py`                     | Batched embedding pipeline with cache lookups.                         |
| `embedding/cache.py`                        | In-memory + JSON-file caches.                                          |
| `embedding/redis_cache.py`                  | Redis-backed embedding cache.                                          |

Belongs here: anything that touches an indexed source file on its way
into Postgres + pgvector.

Does NOT belong here: query-time logic, LLM calls, prompt construction.

Dependencies: `core` (settings, observability), `storage` (IR + chunks
persistence), `retrieval` (none — ingestion is upstream of retrieval).

---

### `src/semrag/retrieval/`

The query-time retrieval pipeline. Hybrid: vector + lexical +
(optional query expansion) + graph expansion + source authority +
(optional) rerank + diversity, fused with Reciprocal Rank Fusion.

| File                          | Responsibility                                            |
| ----------------------------- | --------------------------------------------------------- |
| `pipeline.py`                 | `HybridRetriever` end-to-end entry point.                 |
| `vector_search_adapter.py`    | Adapter around `storage.vector.VectorStore`.              |
| `lexical_search.py`           | Postgres tsvector / lexical search path.                  |
| `fusion.py`                   | Reciprocal Rank Fusion of ranked lists.                   |
| `query_expansion.py`          | Deterministic intent-driven lexical expansion (RI-05).    |
| `reranking.py`                | Rerankers (`lexical_overlap`, `http`, `none`).            |
| `graph_expansion.py`          | Graph expansion over the IR (in-process, `PostgresGraphQueries`). |
| `intent.py`                   | Query-intent detection + structural-query hints.          |
| `source_authority.py`         | Document-aware ranking prior (RI-02); classification + weights. |
| `doc_type_prior.py`           | RI-01 compatibility shim over `source_authority`.         |
| `ranking/source.py`           | Noisy-source penalty + source-authority orchestration.    |
| `ranking/graph.py`            | Penalized graph-expansion scoring/merge.                  |
| `ranking/intent.py`           | Intent-boost table + `apply_intent_ranking`.              |
| `ranking/diversity.py`        | Diversity-aware top-k selection (RI-03).                  |
| `models.py`                   | Retrieval-stage data models (`RetrievalResult`, etc.).    |

Belongs here: anything that turns a user query into a ranked list of
chunk IDs.

Does NOT belong here: prompting, LLM calls, agent loop.

Dependencies: `core`, `storage`.

---

### `src/semrag/reasoning/`

LLM-facing layers. Three subpackages, each with a clear boundary.

```
reasoning/
├── llm/         LLMProvider Protocol + provider implementations + factory
├── agents/      ReAct-style agent loop + tool registry
└── prompting/   System prompt, templates, context assembly, citation
```

| File (under `reasoning/llm/`)   | Responsibility                                            |
| ------------------------------- | --------------------------------------------------------- |
| `interfaces.py`                 | `LLMProvider` Protocol.                                   |
| `factory.py`                    | Backend selection (ollama / openai / vllm / mock).        |
| `ollama_llm.py`                 | Ollama implementation.                                    |
| `openai_llm.py`                 | OpenAI implementation.                                    |
| `openai_compatible_llm.py`      | OpenAI-compatible HTTP implementation.                    |
| `_openai_compatible_impl.py`    | Shared internal helper used by the OpenAI-compatible path. |
| `mock_llm.py`                   | Deterministic mock for tests.                              |

| File (under `reasoning/agents/`) | Responsibility                                            |
| -------------------------------- | --------------------------------------------------------- |
| `agent_loop.py`                  | ReAct-style loop driving the LLM with tool calls.          |
| `tools.py`                       | Tool implementations exposed to the agent.                 |

| File (under `reasoning/prompting/`) | Responsibility                                         |
| ----------------------------------- | ------------------------------------------------------ |
| `pipeline.py`                       | Prompt assembly pipeline entry point.                  |
| `templates.py`                      | System prompt templates.                               |
| `context_assembler.py`              | Selects and formats retrieved chunks for the prompt.   |
| `models.py`                         | Prompt-stage data models.                              |
| `citation.py`                       | Citation formatting + validation.                      |
| `token_estimator.py`                | Rough token count for budget enforcement.              |

Belongs here: anything that turns retrieved chunks into a final answer
via an LLM.

Does NOT belong here: retrieval ranking, file I/O, persistence.

Dependencies: `core`, `retrieval`.

---

### `src/semrag/storage/`

Persistence adapters. Three subpackages, each behind a Protocol so the
upper layers do not bind to a concrete backend.

```
storage/
├── postgres/   SQLAlchemy session, IR/manifest/diagnostics repositories
├── vector/     pgvector adapter + models
└── graph/      Optional Neo4j adapter
```

| File (under `storage/postgres/`) | Responsibility                                          |
| -------------------------------- | ------------------------------------------------------- |
| `session.py`                     | SQLAlchemy engine / session factory.                    |
| `models.py`                      | ORM models for SemRAG tables.                           |
| `ir_repository.py`               | IR node / edge repository.                              |
| `manifest_store.py`              | Manifest persistence (alongside the in-memory variant). |
| `diagnostics_repository.py`      | Diagnostics / run-history queries.                      |
| `graph_queries.py`               | Graph-shaped SQL queries against IR tables.             |

| File (under `storage/vector/`) | Responsibility                                          |
| ------------------------------ | ------------------------------------------------------- |
| `interfaces.py`                | `VectorStore` Protocol.                                 |
| `models.py`                    | Vector-row data models.                                 |
| `pgvector_models.py`           | pgvector-specific models.                               |
| `pgvector_store.py`            | Concrete pgvector implementation.                       |

| File (under `storage/graph/`) | Responsibility                                          |
| ----------------------------- | ------------------------------------------------------- |
| `interfaces.py`               | `GraphStore` Protocol.                                  |
| `neo4j_store.py`              | Neo4j adapter (experimental, disabled by default).     |

Belongs here: anything that persists a domain object to disk / DB.

Does NOT belong here: business logic, LLM calls.

---

### `src/semrag/api/`

The HTTP surface. FastAPI app factory, routers, middleware, and the
per-request concerns (auth, rate limit, query cache).

```
api/
├── app.py                App factory; middleware wiring; startup validation
├── routers/              /projects, /query, /answer
│   ├── projects.py
│   ├── query.py
│   └── answer.py
├── auth.py               API-key authentication
├── api_key.py            APIKey model + hashing helpers
├── api_key_store.py      Postgres-backed API key store
├── project_auth.py       Project-scope authorization
├── config.py             API-specific config (Pydantic)
├── dependencies.py       Shared FastAPI dependencies
├── exception_handler.py  Global exception → HTTP response mapping
├── health.py             /livez, /readyz, /version
├── middleware.py         Custom middleware (request ID, body limits, security headers)
├── query_cache.py        QueryCache ABC + TTLQueryCache + RedisQueryCache
├── rate_limit.py         RateLimiter ABC + Memory + Redis implementations
├── redis_client.py       Redis client factories (sync + async)
├── schemas.py            Pydantic request / response schemas
└── startup_validation.py Fail-fast checks at startup
```

Belongs here: HTTP-specific glue only.

Does NOT belong here: business logic; delegate to `services`.

Dependencies: `core`, `services`, `retrieval`, `reasoning`, `jobs`.

---

### `src/semrag/jobs/`

Async indexing-job runner. Lets the API return immediately while a worker
processes the actual indexing.

| File                | Responsibility                                              |
| ------------------- | ----------------------------------------------------------- |
| `models.py`         | `Job`, `JobState`, `IndexingJobRow`.                        |
| `store.py`          | `JobStore` Protocol + `InMemoryJobStore`.                   |
| `postgres_store.py` | `PostgresJobStore` for durable job state.                   |
| `manager.py`        | `JobManager` — create / update / fetch jobs.                |
| `pipeline.py`       | The actual indexing pipeline executed inside a job.         |
| `executor.py`       | Drives a job from QUEUED → RUNNING → DONE/FAILED.           |
| `worker.py`         | The polling loop run as a stand-alone process (`semrag-worker`). |

Belongs here: anything related to the lifecycle of a long-running
indexing job.

Does NOT belong here: synchronous CLI indexing (lives in
`run_local_qa.py` + `services.py`).

---

### `src/semrag/evaluation/`

Eval harness, citation validator, and benchmarks.

| File                   | Responsibility                                          |
| ---------------------- | ------------------------------------------------------- |
| `harness.py`           | Eval harness entry point.                               |
| `evaluator.py`         | Per-question evaluator.                                 |
| `citation_validator.py | Validates that citations refer to real chunks.          |
| `models.py`            | Eval data models.                                       |
| `metrics.py`           | Eval metrics (e.g. citation precision).                  |
| `benchmarks.py`        | Canned benchmark question sets.                         |

Belongs here: anything that scores the quality of an answer.

---

### `src/semrag/cli.py` and `src/semrag/worker.py`

- `cli.py` is a thin shim invoked by `run_local_qa.py` and by the
  installed `semrag` command.
- `worker.py` is the stand-alone worker entry point, deployed as
  `semrag-worker` (also configured in the Helm chart).

---

## Tests

Tests live under `tests/`, in three trees:

```
tests/
├── unit/           Pytest unit + integration tests
│   ├── conftest.py
│   ├── agents/ api/ chunking/ core/ embedding/ evaluation/
│   ├── graph/ ir/ jobs/ llm/ parsers/ performance/ prompting/
│   ├── retrieval/ scanner/ services/ storage/ vector/
├── load/           Locust + custom load-test scripts (manual)
└── perf/           Performance regression suite (manual)
```

Unit tests are organized to mirror `src/semrag/`. The directory under
`tests/unit/<package>/` holds the tests for `src/semrag/<package>/`.

Key conventions:

- `tests/unit/conftest.py` provides shared fixtures (settings overrides,
  Postgres test database, in-memory stores).
- Load and perf tests are run manually against a deployed API; see
  [tests/load/README.md](../tests/load/README.md) and
  [tests/perf/README.md](../tests/perf/README.md).
- Integration tests that need Postgres / Ollama / a real LLM are
  skipped by default; bring the services up locally to enable them.

---

## Deployment files

```
deploy/
├── helm/semrag/                Helm chart (Chart.yaml, values.yaml, templates/)
│   └── templates/              deployment.yaml, worker-deployment.yaml, etc.
├── k8s/                        Plain Kubernetes manifests (apply in order 00–07)
│   ├── 00-namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 02-secret.yaml
│   ├── 03-pvc.yaml
│   ├── 04-deployment.yaml
│   ├── 05-service.yaml
│   ├── 06-ingress.yaml
│   └── 07-worker-deployment.yaml
└── monitoring/                 Prometheus + Grafana + Alertmanager
    ├── prometheus/             prometheus.yml + alert rules
    ├── grafana/                datasources + dashboards
    └── alertmanager/           alertmanager.yml
```

Docker:

- `Dockerfile` — multi-stage production image, non-root user.
- `docker-compose.yml` + `docker-compose.override.yml` — local Compose
  stack.

For full deployment instructions, see [DEPLOYMENT.md](../DEPLOYMENT.md).

---

## Where do I change X?

A quick lookup table for the most common maintenance tasks.

| Task                                                 | Where to change                                                |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| Add a new environment variable                       | `src/semrag/core/settings.py`                                  |
| Add a new parser for a file type                     | `src/semrag/ingestion/parsers/<language>/parser.py` + register  |
| Add a new chunker                                    | `src/semrag/ingestion/chunking/<chunker>.py` + register         |
| Add a new embedding provider                         | `src/semrag/ingestion/embedding/<provider>.py` + factory entry |
| Add a new LLM provider                               | `src/semrag/reasoning/llm/<provider>.py` + factory entry       |
| Add a new vector store backend                       | `src/semrag/storage/vector/<store>.py` implementing `VectorStore` Protocol |
| Add a new graph store backend                        | `src/semrag/storage/graph/<store>.py` implementing `GraphStore` Protocol |
| Add a new HTTP route                                 | `src/semrag/api/routers/<route>.py` + register in `api/app.py` |
| Add a new tool for the agent                         | `src/semrag/reasoning/agents/tools.py`                          |
| Change the system prompt                             | `src/semrag/reasoning/prompting/templates.py`                  |
| Add a new retrieval reranker                         | `src/semrag/retrieval/reranking.py`                            |
| Add a new Postgres schema migration                  | `alembic/versions/<rev>_<description>.py`                      |
| Add a new Prometheus metric                          | `src/semrag/core/metrics.py` + use in the relevant package      |
| Add a new on-call runbook                            | `docs/operations/<failure-mode>.md` + link from `docs/operations/README.md` |
| Change the API auth model                            | `src/semrag/api/auth.py` + `api/api_key.py`                    |
| Change the rate limit policy                         | `src/semrag/api/rate_limit.py`                                 |
| Change the query cache TTL or backend                | `src/semrag/api/query_cache.py` + `core/settings.py`           |

If the task is "make the system do X", and X already exists in another
package, change that package's Protocol/interface instead of inventing a
parallel path. The Protocols are:

- `Parser` (`ingestion/parsers/interfaces.py`)
- `Chunker` (`ingestion/chunking/interfaces.py`)
- `EmbeddingProvider`, `EmbeddingCache` (`ingestion/embedding/interfaces.py`)
- `LLMProvider` (`reasoning/llm/interfaces.py`)
- `VectorStore` (`storage/vector/interfaces.py`)
- `GraphStore` (`storage/graph/interfaces.py`)
- `QueryCache`, `RateLimiter` (`api/query_cache.py`, `api/rate_limit.py`)
- `JobStore` (`jobs/store.py`)

---

## Debugging starting points

When something is broken, jump to one of these files first based on the
symptom.

| Symptom                                              | Start here                                                       |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| "Indexing didn't pick up my changes"                 | `ingestion/scanner/change_detector.py` + manifest store          |
| "Parsing fails on file type X"                       | `ingestion/parsers/pipeline.py` + the specific parser            |
| "Chunks look wrong"                                  | `ingestion/chunking/pipeline.py` + the specific chunker          |
| "Embeddings are slow or missing"                     | `ingestion/embedding/pipeline.py` + `embedding/factory.py`        |
| "Retrieval returns nothing"                          | `retrieval/pipeline.py` + `retrieval/vector_search_adapter.py`   |
| "Reranking is wrong"                                 | `retrieval/reranking.py`                                          |
| "Agent loop never terminates"                        | `reasoning/agents/agent_loop.py`                                  |
| "LLM call fails"                                     | `reasoning/llm/factory.py` + the specific LLM implementation     |
| "Prompt is malformed"                                | `reasoning/prompting/pipeline.py` + `context_assembler.py`        |
| "Citations don't match chunks"                       | `reasoning/prompting/citation.py`                                 |
| "API returns 5xx"                                    | `api/exception_handler.py` + matching router                     |
| "Rate limit fires unexpectedly"                      | `api/rate_limit.py`                                              |
| "Query cache returns stale results"                  | `api/query_cache.py`                                              |
| "Job stuck in QUEUED"                                | `jobs/worker.py` + `jobs/executor.py`                             |
| "Postgres query is slow"                             | `storage/postgres/` (session + repository files)                  |
| "Vector search returns wrong IDs"                    | `storage/vector/pgvector_store.py`                                |
| "Settings not picked up"                             | `core/settings.py`                                                |
| "Metric missing from /metrics"                       | `core/metrics.py`                                                 |
| "Runbook needed for a new failure mode"              | `docs/operations/` + the existing failure-mode runbooks there    |

When in doubt, set `SEMRAG_LOG_LEVEL=DEBUG` (or `TRACE`) to emit the
per-stage artifacts under `.semrag/debug/<run-id>/`. The README has the
artifact layout.

---

## Cross-references

- [Architecture](../ARCHITECTURE.md) — long-form architecture and data
  flow diagrams.
- [Deployment](../DEPLOYMENT.md) — every supported deployment environment.
- [Repository Guidelines](../AGENTS.md) — contribution policy.
- [Operations runbooks](./operations/) — on-call material.
- [Tests load README](../tests/load/README.md) — load testing.
- [Tests perf README](../tests/perf/README.md) — performance regression.
- [Documentation index](./README.md) — entry point for every other
  documentation page.