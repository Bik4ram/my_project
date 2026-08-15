# `src/semrag` — Source Package

> **The SemRAG production code.** This README is the navigation map for
> the Python package itself: what each subpackage owns, how they
> depend on each other, where new code belongs. For architectural
> deep-dives, the data-flow diagrams, and the end-to-end stages, see
> [ARCHITECTURE.md](../../ARCHITECTURE.md). For "where is X?" use the
> [CODEBASE_GUIDE.md](../../docs/CODEBASE_GUIDE.md). For "why does
> this work the way it does?", see [developer_guide.md](../../docs/developer_guide.md).

Intended audience: **Developers** and **Maintainers**.

---

## 1. Purpose

`src/semrag/` is everything the platform does. Each subpackage owns a
single concern; together they form the indexing → retrieval → reasoning
pipeline that powers both the CLI (`semrag` / `run_local_qa.py`) and
the HTTP API (`semrag.api.app`).

The package exists to keep two properties true:

1. **One concern, one place.** Pipeline orchestration is in
   `services/`. Indexing is in `ingestion/`. Query-time retrieval is in
   `retrieval/`. LLM-facing layers are in `reasoning/`. Persistence is
   in `storage/`. The HTTP surface is in `api/`. Background work is
   in `jobs/`. Cross-cutting primitives (settings, logging, metrics)
   are in `core/`. Anything that belongs to one of these doesn't live
   in another.
2. **One Protocol, every backend.** Every pluggable seam — parsers,
   chunkers, embedding providers, LLM providers, vector stores, graph
   stores, query caches, rate limiters, job stores — is a Protocol or
   ABC. The rest of the codebase depends on the Protocol; adding a
   backend is a single file + a single factory entry.

---

## 2. Package organization

Nine subpackages, two top-level modules, and one log-only module. Each
entry is real and exists in this repository.

### `src/semrag/core/`

Cross-cutting primitives. **Imported by every other subpackage;
imports nothing from them.** Owns `Settings`, the logging / metrics /
tracing surface, the redaction layer, the per-query trace accumulator,
and the optional per-run debug artifact writer. If a utility is used
by more than one subpackage and is not domain-specific, it lives
here.

### `src/semrag/services/`

Thin orchestration layer. `IngestionService.index()` and
`QueryService.ask()` compose the lower-level packages into end-to-end
flows. Services are constructed at the edges (CLI, FastAPI dependency
providers, jobs) and take every collaborator via the constructor.
The two services are the only places that know the full pipeline.

### `src/semrag/ingestion/`

The one-way indexing pipeline. Has five subpackages:

- `ingestion/scanner/` — filesystem walking, content hashing, language
  classification, git metadata, dependency extraction, change
  detection, manifest storage.
- `ingestion/parsers/` — language-aware parsers producing a
  parse-result shape the IR builder can consume. One subpackage per
  language family: `systemverilog`, `vhdl`, `python_tcl`,
  `documents` (Excel / PDF / Markdown), `logs` (regression logs),
  `structured_data`.
- `ingestion/ir/` — two-pass IR builder, IR data model, deterministic
  node IDs, symbol tables. Producing nodes + edges is its only
  concern.
- `ingestion/chunking/` — language-aware chunker selection
  (`Chunker` Protocol) and the chunking pipeline. Python uses an
  AST-aware chunker; SystemVerilog uses an HDL-aware chunker; the
  default is a generic window chunker.
- `ingestion/embedding/` — `EmbeddingProvider` Protocol + per-provider
  implementations, `EmbeddingCache` Protocol + memory / JSON-file /
  Redis backends, batched embedding pipeline with cache lookups, and
  the factory functions that map `Settings` to a concrete provider +
  cache.

### `src/semrag/retrieval/`

The query-time retrieval pipeline. Owns the `HybridRetriever` that
runs vector search + lexical search, fuses them with Reciprocal Rank
Fusion, optionally applies intent boost, hybrid top-k reservation,
graph expansion, the noisy-source penalty, stable sort, and
optional reranking. Also owns the per-chunk `RankingTrace` diagnostic
that makes ranking regressions easy to localize.

### `src/semrag/reasoning/`

LLM-facing layers. Three subpackages:

- `reasoning/llm/` — `LLMProvider` Protocol; Ollama, OpenAI, OpenAI-
  compatible HTTP, vLLM, and a deterministic mock implementation.
  `build_llm()` is the only factory entry point.
- `reasoning/agents/` — the ReAct-style agent loop
  (`AgentLoop.run`) with a hard iteration cap and a `ToolRegistry`
  of agent-callable tools. The default tools are `search_codebase`
  (routes through the same `HybridRetriever`), `list_files`, and
  optionally `read_file` when a project root is configured.
- `reasoning/prompting/` — `PromptBuilder` allocates the token budget
  in a deliberate order (system + history first, retrieved context
  last), assembles context, formats citations, and estimates tokens.

### `src/semrag/storage/`

Persistence adapters. Three subpackages:

- `storage/postgres/` — SQLAlchemy session / sessionmaker, ORM models
  for the SemRAG tables, repositories for IR / manifest / diagnostics,
  and graph-shaped SQL queries for retrieval's graph-expansion path.
- `storage/vector/` — `VectorStore` Protocol; pgvector implementation
  with embed-and-upsert in one call and a search adapter that
  converts vector rows to the retrieval-stage `RetrievalResult`.
- `storage/graph/` — `GraphStore` Protocol and an experimental Neo4j
  adapter (disabled by default). Retrieval reads the graph
  in-process via `PostgresGraphQueries`; the Neo4j adapter is not
  wired into the retrieval path.

### `src/semrag/api/`

The HTTP surface. Owns the FastAPI app factory (`app.py`), the three
routers (`projects`, `query`, `answer`), the auth / rate-limit /
query-cache / Redis-client plumbing, the Pydantic request / response
schemas, the health endpoints (`/livez`, `/readyz`, `/version`), the
middleware stack (request ID, tracing, metrics, logging, security
headers, body-size cap), the global exception handler that produces
the standard error envelope, and the `dependencies.py` module that
holds the cross-cutting singletons.

### `src/semrag/jobs/`

Background indexing-job runner. Models, store Protocols and
implementations (in-memory + Postgres), the manager that creates /
updates / fetches jobs, the executor that drives a job from
QUEUED → RUNNING → DONE / FAILED / CANCELLED, the per-stage
progress calculator, and the polling worker entry point run by the
`semrag-worker` process.

### `src/semrag/evaluation/`

Eval harness, per-question evaluator, citation validator (real,
deterministic check that every `[Source: …]` reference corresponds to
a path that was in the retrieved context), eval data models, eval
metrics, and canned benchmark question sets. Eval is a query-time
consumer; it does not produce answers.

### Top-level modules

- `cli.py` — Argparse surface for the installed `semrag` command and
  the legacy `python run_local_qa.py` entry. Constructs services
  directly using the same factories the API uses.
- `worker.py` — Stand-alone worker entry point deployed as
  `semrag-worker`. Owns the worker lifecycle and the loop that pulls
  QUEUED jobs.
- `audit.py` — Security audit log helper. Single-purpose module: emit
  structured audit events for `authentication`, `authorization`,
  `project_indexing`, etc. Used by `api/auth.py` and friends.

---

## 3. Responsibility matrix

Each subpackage is owned by exactly one team-mental-model concern. The
matrix below maps the runtime pipeline to the package that owns it.

| Runtime concern                          | Owned by                    |
| ---------------------------------------- | --------------------------- |
| Pydantic-settings, log / metric / trace  | `core`                      |
| Top-level orchestration                  | `services`                  |
| Filesystem walk, hashing, change detect  | `ingestion/scanner`         |
| File → parse-result                      | `ingestion/parsers`         |
| Build IR graph (nodes + edges)           | `ingestion/ir`              |
| Chunking strategy                        | `ingestion/chunking`        |
| Embedding computation + caching          | `ingestion/embedding`       |
| Vector + lexical + RRF + rerank          | `retrieval`                 |
| LLM providers                            | `reasoning/llm`             |
| Agent loop + tool registry               | `reasoning/agents`          |
| Prompt building + token budgeting        | `reasoning/prompting`       |
| SQLAlchemy session, ORM models, repos    | `storage/postgres`          |
| pgvector embed / search                  | `storage/vector`            |
| Optional Neo4j adapter                   | `storage/graph`             |
| FastAPI app, routers, deps, middleware   | `api`                       |
| Caching, rate limit, Redis clients       | `api`                       |
| Async indexing-job lifecycle             | `jobs`                      |
| Eval harness + citation validator        | `evaluation`                |
| CLI entry + worker entry                 | `cli.py`, `worker.py`       |
| Security audit log                       | `audit.py`                  |

Two implications worth stating:

- **Adding new code means picking the row first.** If the concern
  doesn't match an existing row, you're either adding a new
  subpackage (heavy) or your code is in the wrong place (almost
  always the second one).
- **Multi-row changes are fine for services, jobs, and CLI.** These
  are exactly the packages that compose the others. Everything else
  sticks to its row.

---

## 4. Dependency relationships

The arrows below point in the direction of *imports from*. Reading the
diagram top-to-bottom: each row may import from the rows below it and
from `core` only.

```
                          ┌──────────────────────────────────────────────┐
                          │                  cli.py / worker.py          │
                          │ (read settings, build services, run worker)  │
                          └────────┬───────────────────────────┬─────────┘
                                   │                           │
                                   ▼                           ▼
                          ┌──────────────────┐       ┌────────────────────┐
                          │   services       │       │        jobs        │
                          └──────┬───────────┘       └─────┬──────────┬───┘
                                 │                        │          │
                                 ▼                        ▼          ▼
        ┌────────────┬─────────────┬──────────────┐   ┌────────────────────┐
        │ ingestion  │ retrieval   │ reasoning    │   │ storage (postgres) │
        └────┬───────┴────┬────────┴───────┬──────┘   └────────────────────┘
             │            │                │
             ▼            ▼                ▼
        ┌─────────────────────────────────────────────────────────┐
        │                       storage                           │
        │ (postgres / vector / graph)                             │
        └─────────────────────────────────────────────────────────┘
                                                                ▲
                                                                │
                                                       ┌────────┴────────┐
                                                       │      core       │
                                                       │ (settings, log, │
                                                       │  metrics, ..)   │
                                                       └─────────────────┘
```

Concretely:

- `services` imports from `ingestion`, `retrieval`, `reasoning`, and
  `storage`. It does not import from `api` or `jobs`.
- `api` imports from `services`, `retrieval`, `reasoning`, `jobs`, and
  `core`. Routes never import from `ingestion` primitives directly;
  they go through `services`.
- `jobs` imports from `services` (it *is* the API's way of running
  services asynchronously) and from `storage/postgres` (the durable
  job store).
- `core` imports from nothing in this package; it depends on stdlib
  plus `pydantic` / `pydantic-settings` / OpenTelemetry /
  Prometheus client only.
- `cli.py` and `worker.py` are at the top and may import from
  anything below.

### Architectural boundaries

These are the rules that produce the diagram above. Breaking one is a
PR-rewrite, not a quick fix.

- **`core` is the only package with no siblings in its import graph.**
  Anything else that has no subpackage-level dependencies is fine to
  live in `core`, but never in a domain subpackage.
- **`ingestion` does not import from `retrieval`.** Ingestion is
  upstream of retrieval; the dataflow is one-way.
- **`retrieval` does not import from `ingestion`.** Symmetric. The
  boundary is reached only via the persistence layer (`storage`).
- **`api` does not import from `ingestion` or `retrieval`
  primitives.** Routes call `services`, never the lower-level
  primitives directly. (The CLI is the only exception: it may
  compose them to construct a service instance.)
- **`reasoning` may route through `retrieval`** — the
  `search_codebase` agent tool is the explicit seam that lets the
  agent loop drive retrieval. That's the only reason `reasoning` is
  allowed to know about `retrieval`.
- **`evaluation` is allowed to use `services`** because eval is
  itself a query-time consumer. It does not write to the index.

For the **why** behind each boundary, see the "Common architectural
rules" chapter of the [developer guide](../../docs/developer_guide.md).

---

## 5. Source package layout

```
src/semrag/
├── __init__.py
│
├── core/                       Cross-cutting primitives (settings, logging,
│                               metrics, tracing, redaction, debug artifacts,
│                               query trace)
├── services/                   IngestionService + QueryService (composition)
│
├── ingestion/                  Indexing pipeline (one-way)
│   ├── scanner/                Walk, hash, classify, change-detect, manifest
│   ├── parsers/                Language-aware parsing
│   │   ├── systemverilog/
│   │   ├── vhdl/
│   │   ├── python_tcl/
│   │   ├── documents/          (excel, pdf, markdown)
│   │   ├── structured_data/
│   │   └── logs/               (regression logs)
│   ├── ir/                     IR builder, models, symbol tables
│   ├── chunking/               Chunkers + pipeline
│   ├── embedding/              Providers, caches, batched pipeline, factories
│   └── streaming.py            Streaming write paths
│
├── retrieval/                  Query pipeline (vector + lexical + RRF + rerank)
│
├── reasoning/                  LLM-facing layers
│   ├── llm/                    LLMProvider + implementations + factory
│   ├── agents/                 Agent loop + tool registry
│   └── prompting/              Prompt builder, context assembler, citation
│
├── storage/                    Persistence adapters
│   ├── postgres/               SQLAlchemy session / ORM / repos
│   ├── vector/                 pgvector adapter
│   └── graph/                  Neo4j adapter (experimental)
│
├── api/                        FastAPI surface
│   ├── app.py                  App factory, lifespan, router registration
│   ├── routers/                /projects, /query, /answer
│   ├── auth.py / api_key.py / project_auth.py / api_key_store.py
│   ├── query_cache.py / rate_limit.py / redis_client.py
│   ├── middleware.py / exception_handler.py / dependencies.py
│   ├── health.py / schemas.py / config.py / startup_validation.py
│
├── jobs/                       Async indexing-job runner
│
├── evaluation/                 Eval harness, citation validator, benchmarks
│
├── cli.py                      `semrag` entrypoint
├── worker.py                   `semrag-worker` entrypoint
└── audit.py                    Security audit log helper
```

For the full file-level catalog inside each directory, see the
[CODEBASE_GUIDE.md](../../docs/CODEBASE_GUIDE.md). The guide is the
authoritative "where exactly does X live" reference; this README is
the package-level mental model.

---

## 6. Development guidelines

### Where to add new functionality

| New functionality                        | Add to                                                       |
| ---------------------------------------- | ----------------------------------------------------------- |
| New `SEMRAG_*` environment variable      | `core/settings.py` + reading it via `get_settings()`        |
| New parser for a new file type           | `ingestion/parsers/<language>/parser.py` + register          |
| New chunker for a new language           | `ingestion/chunking/<chunker>.py` + register                 |
| New embedding provider                   | `ingestion/embedding/<provider>.py` + factory entry          |
| New LLM provider                         | `reasoning/llm/<provider>.py` + factory entry               |
| New vector store backend                 | `storage/vector/<store>.py` + Protocol implementer          |
| New graph store backend                  | `storage/graph/<store>.py` + Protocol implementer           |
| New HTTP route                           | `api/routers/<route>.py` (or add a new router)               |
| New tool for the agent                   | `reasoning/agents/tools.py`                                  |
| New reranker                             | `retrieval/reranking.py`                                     |
| New query backend (cache / rate limit)   | `api/query_cache.py` or `api/rate_limit.py` (Protocol)       |
| New JobStore backend                     | `jobs/store.py` (ABC) + `jobs/postgres_store.py` (or new)    |
| New cross-cutting utility                | `core/` (only if used by ≥2 subpackages)                    |
| New eval metric                          | `evaluation/metrics.py`                                      |

### Where NOT to add functionality

- **Do not add HTTP route logic to `services/`.** Services are
  reusable across CLI + API. HTTP concerns (Pydantic schemas,
  exception envelopes, dependency wiring) belong in `api/`.
- **Do not add LLM calls to `retrieval/`.** Retrieval is LLM-free by
  design. The agent loop is the only path that calls an LLM.
- **Do not add DB queries to `retrieval/` or `reasoning/`.** The
  Postgres session lives in `storage/postgres`. Retrieval goes
  through `storage.vector` and `storage.postgres.graph_queries`.
- **Do not put files in the source root.** The source root holds
  exactly: `__init__.py`, `cli.py`, `worker.py`, `audit.py`.
  Everything else is a subpackage.
- **Do not add a new subpackage to bypass an existing boundary.** If
  a function feels like it belongs in `core` but `core` already has
  three other responsibilities, the answer is to refine the
  abstraction, not to introduce `core2/`.

### Package ownership boundaries

Each subpackage owns its own piece of the system, including:

- Its own Pydantic models (when domain-specific).
- Its own Protocol / ABC definitions.
- Its own factories.
- Its own tests under `tests/unit/<package>/`.

Subpackages do **not** own:

- Cross-cutting settings (owned by `core/settings.py`).
- The HTTP error envelope (owned by `api/exception_handler.py`).
- The metrics registry (owned by `core/metrics.py`).
- The logger configuration (owned by `core/observability.py`).

If a subpackage wants to *use* one of those, it imports the helper;
it never redefines or forks it. This is what makes the boundaries
stick over time.

---

## 7. Extension points

Every place the package is intentionally pluggable. The list below is
the short version; the [developer guide](../../docs/developer_guide.md)
chapter 13 has the long version.

- **Parsers** — implement `Parser` Protocol
  (`ingestion/parsers/interfaces.py`); register in
  `ingestion/parsers/pipeline.py`.
- **Chunkers** — implement `Chunker` Protocol
  (`ingestion/chunking/interfaces.py`); register in
  `ingestion/chunking/pipeline.py`.
- **Embedding providers** — implement `EmbeddingProvider`
  (`ingestion/embedding/interfaces.py`); add case in
  `build_embedding_provider` (`ingestion/embedding/factory.py`); add
  setting to `core/settings.py`.
- **LLM providers** — implement `LLMProvider`
  (`reasoning/llm/interfaces.py`); add case in `build_llm`
  (`reasoning/llm/factory.py`); add setting to `core/settings.py`.
- **Vector store backends** — implement `VectorStore`
  (`storage/vector/interfaces.py`); update selection in the call
  sites that construct one (`retrieval/vector_search_adapter.py`,
  `services/services.py`).
- **Graph store backends** — implement `GraphStore`
  (`storage/graph/interfaces.py`); wire the optional fallback in
  `retrieval/graph_expansion.py`.
- **Query cache / rate limiter backends** — fill the abstract class
  in `api/query_cache.py` or `api/rate_limit.py`; selection is via
  `SEMRAG_QUERY_CACHE_BACKEND` / `SEMRAG_RATE_LIMIT_BACKEND`. Redis
  clients come from `api/redis_client.py`.
- **JobStore backends** — fill `JobStore` ABC
  (`jobs/store.py`); selection is via
  `SEMRAG_JOB_STORE_BACKEND`.
- **API modules** — add a new file under `api/routers/`, register the
  router in `api/app.py`. Add a Pydantic model in `api/schemas.py`.

In every case, the pattern is the same: implement the Protocol, add
one entry to a factory, add (where appropriate) one new field to
`Settings`. Nothing else.

---

## 8. Common navigation paths

A short list of "I want to do X" → "start here". Each path tells you
where to write the change, not how.

- **I want to add a parser for a new language.**
  → `src/semrag/ingestion/parsers/<language>/parser.py`. Then register
  it in `ingestion/parsers/pipeline.py`. Tests under
  `tests/unit/parsers/`.

- **I want to debug retrieval quality for a query.**
  → `src/semrag/retrieval/pipeline.py` (`HybridRetriever`). Run with
  `SEMRAG_LOG_LEVEL=DEBUG` and read `.semrag/debug/<run_id>/retrieval/ranking_trace.json`.

- **I want to add an API endpoint.**
  → `src/semrag/api/routers/<resource>.py`. Wire the dependency
  chain (`require_api_key`, `get_project_authorizer`,
  `enforce_rate_limit`). Register the router in
  `src/semrag/api/app.py`.

- **I want to change how indexing works.**
  → `src/semrag/services/services.py:IngestionService.index()`. The
  orchestrator is one function you read top-to-bottom; every stage is
  a labeled `with stage(...)` block. Add a new stage there, or look
  at the relevant subpackage to change an existing one.

- **I want to add a new embedding provider.**
  → `src/semrag/ingestion/embedding/<provider>.py` implementing
  `EmbeddingProvider`. Add the case in
  `ingestion/embedding/factory.py` and the setting in
  `core/settings.py`.

- **I want to change the agent loop's behavior.**
  → `src/semrag/reasoning/agents/agent_loop.py`. The class is small
  (a `run` method with a `for` loop); read it once before editing.

- **I want to add a new metric.**
  → `src/semrag/core/metrics.py`. Add a helper function; never import
  a Prometheus collector from outside this file.

- **I want to change the system prompt.**
  → `src/semrag/reasoning/prompting/templates.py`.

- **I want to change how citations are emitted.**
  → `src/semrag/reasoning/prompting/citation.py` (formatter) and
  `src/semrag/evaluation/citation_validator.py` (validator). Both
  layers must agree on the format.

- **I want to debug a job that's stuck.**
  → `src/semrag/jobs/worker.py`. The job state machine is in
  `src/semrag/jobs/store.py`; the transition logic is in
  `src/semrag/jobs/manager.py`.

- **I want to debug a 5xx response.**
  → `src/semrag/api/exception_handler.py` (envelope) +
  `src/semrag/api/middleware.py` (request ID, log line).

- **I want to understand how `settings.py` validates at startup.**
  → `src/semrag/api/startup_validation.py`.

---

## 9. Related documentation

- [Architecture](../../ARCHITECTURE.md) — module layout, end-to-end
  data-flow diagrams, settings / services / providers overview. The
  authoritative source for "how does the system work?" diagrams.
- [Codebase guide](../../docs/CODEBASE_GUIDE.md) — file-by-file
  catalog of every directory; "where is X?" and "where do I change
  Y?" lookup tables.
- [Developer guide](../../docs/developer_guide.md) — every chapter
  this README references, but explained: end-to-end lifecycles,
  dependency injection, storage layer, jobs, security, observability,
  error handling, debugging workflow, architectural rules.
- [Documentation index](../../docs/README.md) — entry point for
  every other documentation page.
- [Repository guidelines](../../AGENTS.md) — contribution workflow;
  what the project values.