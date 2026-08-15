# `tests/` — Test suite

The complete test directory for SemRAG. Three families of tests live
here, each with a different runtime and a different purpose:

- **`tests/unit/`** — fast, in-process pytest suite. The gate that
  must pass on every PR. Talks to a real PostgreSQL when one is
  reachable and skips gracefully when it isn't; everything else
  runs in-process.
- **`tests/load/`** — load-test scripts invoked manually against a
  running API. Not part of CI.
- **`tests/perf/`** — performance regression suite. Two modes:
  in-process (for CI) and live-URL (for SLO measurement).

The test suite follows a few rules that the rest of this document
makes explicit:

1. **Live services are gated on reachability.** A PostgreSQL
   instance is required for the storage and API tests; a Neo4j
   instance is required for the graph tests. Both have
   skip-gracefully fixtures so a missing service never blocks
   the rest of the suite.
2. **External HTTP is mocked, never real.** LLM and embedding
   providers are exercised against faked HTTP responses (via
   `MockLLM` for the LLM and `requests`-mock patterns for
   embedders), not against live endpoints. CI does not make
   network calls.
3. **Tests don't truncate tables.** The shared `db_session`
   fixture runs each test inside a transaction that rolls back
   on teardown — so the real `semrag_test` database stays
   empty across runs without a per-test or per-class reset.

This is **not** a guide to writing tests; it's a map of the
existing test architecture. For pytest specifics, see the
[pytest documentation](https://docs.pytest.org/).

---

## 1. Purpose of the tests directory

`tests/` exists to give the rest of the codebase a gate. Three
gates, actually:

- **Correctness.** Every public path in `src/semrag/` is
  exercised by a unit test against a real PostgreSQL.
- **Regression on perf.** The performance suite catches "this
  code got 10x slower overnight" regressions, not absolute
  SLO measurement.
- **Capacity shape.** The load suite tells the team what
  traffic the API can actually serve under realistic
  concurrent queries + indexing.

The directory does **not** exist to:

- Test third-party libraries. The embedding provider and LLM
  provider tests do exercise the *contract* of OpenAI / Ollama
  / etc., but against faked HTTP, not against a live server.
- Validate a deployed cluster. That's the
  [`docs/operations/`](../docs/operations/README.md) on-call
  runbooks' job.
- Provide a benchmarking dashboard. The perf suite emits JSON
  shaped for a downstream dashboard; it does not build one.

---

## 2. Directory layout

```
tests/
├── README.md                    (this file)
│
├── unit/                        pytest suite (CI gate)
│   ├── conftest.py              shared fixtures (Postgres skip-gracefully)
│   ├── llm/                     reasoning/llm tests
│   ├── core/                    core/ tests (settings, observability, metrics, ...)
│   ├── agents/                  reasoning/agents tests
│   ├── vector/                  storage/vector tests
│   ├── storage/                 storage/postgres tests
│   ├── evaluation/              evaluation/ tests
│   ├── api/                     api/ tests + conftest.py (api_client, auth_headers)
│   ├── graph/                   storage/graph tests + conftest.py (Neo4j skip-gracefully)
│   ├── ir/                      ingestion/ir tests
│   ├── embedding/               ingestion/embedding tests
│   ├── jobs/                    jobs/ tests
│   ├── services/                services/ tests
│   ├── retrieval/               retrieval/ tests
│   ├── chunking/                ingestion/chunking tests
│   ├── parsers/                 ingestion/parsers tests
│   ├── performance/             core/performance tests
│   ├── prompting/               reasoning/prompting tests
│   └── scanner/                 ingestion/scanner tests
│
├── load/                        manual load-test scripts
│   ├── README.md
│   ├── locustfile.py            Locust scenarios (QueryUser, AnswerUser, IndexUser, HealthUser)
│   └── run_load_test.py         dependency-free smoke load test
│
└── perf/                        performance regression suite
    ├── README.md
    └── run_perf_suite.py        in-process + live-URL modes, JSON output
```

The `unit/` subdirectories mirror the source tree one-to-one —
`tests/unit/storage/` tests `src/semrag/storage/`,
`tests/unit/retrieval/` tests `src/semrag/retrieval/`, and so on.
Every package under `src/semrag/` has a matching `tests/unit/`
directory; this is a project convention, not enforced by pytest.

The `__init__.py` files inside `tests/unit/<sub>/` and
`tests/load/`, `tests/perf/` are empty — they exist to make the
directories importable from the project root and to let pytest
discover tests through them.

---

## 3. Unit tests

The pytest suite. 109 test files as of this writing, split
across the directories above. Every test file is named
`test_<module>.py` and contains `test_<thing>()` functions; this
is pytest's default discovery, not project-specific.

### 3.1 Subpackage test counts (approximate, by directory)

| Directory                          | Tests `src/semrag/...`         | Notes                                |
| ---------------------------------- | ------------------------------ | ------------------------------------ |
| `tests/unit/llm/`                  | `reasoning/llm/`               | HTTP-mocked; no live LLM             |
| `tests/unit/core/`                 | `core/`                        | In-process; settings, observability, metrics, redaction, debug artifacts, query trace, tracing, performance |
| `tests/unit/agents/`               | `reasoning/agents/`            | `MockLLM` for the loop; one end-to-end integration test |
| `tests/unit/vector/`               | `storage/vector/`              | Postgres-backed; skipped if no DB    |
| `tests/unit/storage/`              | `storage/postgres/`            | Postgres-backed; uses `db_session` fixture |
| `tests/unit/evaluation/`           | `evaluation/`                  | Per-evaluator tests + harness integration |
| `tests/unit/api/`                  | `api/`                         | FastAPI `TestClient` + `api_client` fixture; ~17 files |
| `tests/unit/graph/`                | `storage/graph/`               | Neo4j-backed; **skip** if not reachable (the `neo4j_store` module is real but unverified) |
| `tests/unit/ir/`                   | `ingestion/ir/`                | In-process; deterministic node IDs   |
| `tests/unit/embedding/`            | `ingestion/embedding/`         | HTTP-mocked; tests for the OpenAI-shape + TFIDF providers |
| `tests/unit/jobs/`                 | `jobs/`                        | In-process; in-memory + Postgres-backed store tests |
| `tests/unit/services/`             | `services/`                    | End-to-end orchestrator tests       |
| `tests/unit/retrieval/`            | `retrieval/`                   | In-process + Postgres-backed; ranks/fusion/expansion |
| `tests/unit/chunking/`             | `ingestion/chunking/`          | In-process; language-aware chunkers  |
| `tests/unit/parsers/`              | `ingestion/parsers/`           | In-process; per-language parser tests |
| `tests/unit/performance/`          | `core/performance/`            | In-process; benchmark helpers        |
| `tests/unit/prompting/`            | `reasoning/prompting/`         | In-process; budget, citation, templates |
| `tests/unit/scanner/`              | `ingestion/scanner/`           | In-process; filesystem walk + change-detect |

### 3.2 Integration tests (inside `tests/unit/`)

Most files in `tests/unit/` are *unit* tests in the strict sense
(one function under test, mocked collaborators). A handful are
*integration* tests that exercise multiple packages against a
real PostgreSQL:

- `tests/unit/storage/test_full_pipeline_integration.py` —
  the entire ingest → chunk → embed → persist → retrieve
  pipeline against a real Postgres.
- `tests/unit/api/test_app_integration.py` — the FastAPI app
  end-to-end via `TestClient`, with the real DB session
  overridden to the test transaction.
- `tests/unit/agents/test_agent_retrieval_integration.py` —
  the agent loop driving real `HybridRetriever` calls
  (mocked LLM, real DB).
- `tests/unit/evaluation/test_harness_integration.py` — the
  eval harness running canned benchmarks end-to-end.

The naming convention `*_integration.py` is the signal that a
file is one of these; it's a project convention, not pytest's.

---

## 4. Integration tests

The unit-test directory contains both unit and integration
tests, distinguished by the `*_integration.py` naming
convention. There is no separate top-level `tests/integration/`
directory — every test that needs real services lives in
`tests/unit/`.

Why: pytest's discovery and fixtures work the same way for
both; the only difference is what they exercise. Putting them
in the same tree keeps the CI command simple (`pytest
tests/unit/`) and lets a developer run "all the fast tests" or
"all the storage tests" with the same pattern.

The integration tests that exist today:

| Test file                                       | Exercises                                                                                     |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `tests/unit/storage/test_full_pipeline_integration.py` | Scanner → parser → IR → chunker → embedder → `PgvectorStore` → `HybridRetriever`, end-to-end |
| `tests/unit/api/test_app_integration.py`        | The FastAPI app, full DI graph, real DB session via the `api_client` fixture                  |
| `tests/unit/agents/test_agent_retrieval_integration.py` | `AgentLoop` driving `search_codebase` tool calls, `MockLLM` for the LLM, real DB          |
| `tests/unit/evaluation/test_harness_integration.py` | The full eval harness running a canned benchmark against the real index                       |

These are the highest-value tests in the suite — they're the
closest thing to "does the system work as a whole." They are
also the slowest, and the most likely to need a real Postgres.

---

## 5. Performance tests

[`tests/perf/`](./perf/README.md) — the performance regression
suite. The `README.md` in that directory is the source of truth;
this section summarizes.

`tests/perf/run_perf_suite.py` measures:

- **Retrieval latency** at p50 / p95 / p99
- **Answer latency** at p50 / p95 / p99
- **Indexing throughput** (chunks/sec; in-process only)
- **Cache hit rate**
- **Peak memory** (RSS via `resource.getrusage`)

Two modes:

- **In-process** (default) — runs via FastAPI `TestClient` with
  TFIDF embeddings and a `MockLLM`. No live services. Catches
  regressions in CI. Output:

  ```bash
  python tests/perf/run_perf_suite.py --output bench.json
  ```

- **Live URL** — runs against a real SemRAG API. For SLO
  measurement. Output:

  ```bash
  python tests/perf/run_perf_suite.py --url http://localhost:8000 \
      --api-key dev-local-key --project rtl_uvm --output bench.json
  ```

The JSON output (`BenchmarkResult` dataclass) is shaped for
direct ingestion into a benchmarking dashboard — top-level
keys map to the required metrics.

What the perf suite does **not** do:

- Run inside pytest. It's a `python` script, not a `pytest`
  test. CI invokes it directly.
- Measure absolute SLO. The in-process numbers are useful for
  catching "this code got 10x slower overnight" regressions;
  the live-URL numbers are the closest thing to absolute but
  still depend on the host.
- Track baseline numbers across runs. The output is a snapshot;
  a downstream dashboard does the trend analysis.

---

## 6. Load tests

[`tests/load/`](./load/README.md) — manual load-test scripts.
The `README.md` in that directory is the source of truth; this
section summarizes.

Two scripts that intentionally overlap:

- **`tests/load/locustfile.py`** — Locust scenarios. The
  production-grade option. Run with:

  ```bash
  pip install locust
  locust -f tests/load/locustfile.py --host http://localhost:8000
  ```

  Defines four Locust user classes:
  `QueryUser` (concurrent queries), `AnswerUser` (mixed query
  + LLM), `IndexUser` (mixed indexing + query), `HealthUser`
  (kubelet-style `/livez` / `/readyz` / `/metrics` traffic).
  Configurable via `SEMRAG_LT_HOST`, `SEMRAG_LT_API_KEY`,
  `SEMRAG_LT_PROJECT`, `SEMRAG_LT_INDEX_ROOT`.

- **`tests/load/run_load_test.py`** — dependency-free smoke
  load test using only `requests`. Run with:

  ```bash
  python -m tests.load.run_load_test --url http://localhost:8000 \
      --concurrency 4 --duration 30 --json
  ```

  No `pip install` step; usable in CI smoke tests where
  Locust isn't installed.

What the load tests do **not** do:

- Run as part of `pytest tests/unit/`. They're invoked
  manually, against a running API, by an operator or CI
  pipeline that wants to validate capacity.
- Generate dashboards. Both scripts emit raw stats (Locust's
  web UI; the standalone script's JSON); a downstream tool
  visualizes.
- Mock the API. They need a real one.

---

## 7. Fixtures

The fixtures live in three `conftest.py` files, scoped to
the directories where they're needed:

### 7.1 `tests/unit/conftest.py`

The shared Postgres fixtures. Two fixtures:

- **`pg_engine`** (`scope="session"`) — builds a SQLAlchemy
  engine from `SEMRAG_TEST_DATABASE_URL` (default
  `postgresql+psycopg2://semrag:semrag_test@localhost:5432/semrag_test`),
  does a `SELECT 1` to confirm the DB is reachable, and
  `pytest.skip`s with a clear message if it isn't. Yields the
  engine; disposes on teardown.

- **`db_session`** (`scope="function"`) — opens a connection,
  starts a transaction, builds a `sessionmaker` bound to that
  connection, and yields a `Session`. On teardown:
  `session.close()` → `transaction.rollback()` →
  `connection.close()`. **Never commits.**

The "never commits" rule is the whole point: every test
starts in a clean state and leaves no residue. New tables
don't need a per-test truncate; tests can't poison each other
through uncommitted writes; the `semrag_test` database stays
empty across runs without a per-class or per-session reset.

### 7.2 `tests/unit/api/conftest.py`

API-specific fixtures. Two fixtures + a documented decision:

- **`api_client`** — sets `SEMRAG_API_KEYS`,
  `SEMRAG_EMBEDDING_PROVIDER=tfidf`, and
  `SEMRAG_JOB_STORE_BACKEND=memory` via `monkeypatch`,
  resets the settings cache and the job store singleton, and
  builds the app via `create_app()`. Overrides
  `get_db_session` to yield the same `db_session` fixture
  every other live-Postgres test uses (transactional,
  rolled back on teardown). Yields a `TestClient`.

  The job store override is necessary because the production
  default is `database`, and tests need the in-memory store
  (no Postgres dependency for the store itself). Tests that
  specifically exercise the database-backed job store
  override this fixture.

- **`auth_headers`** — `{"Authorization": "Bearer test-api-key-12345"}`.

- **No client-wide default auth header.** The fixture
  deliberately does not set a default `Authorization` header
  on the `TestClient`. `httpx` merges per-call headers with
  client defaults rather than letting an empty per-call dict
  clear them, so a client-wide default would make it
  impossible for any test to genuinely send an
  unauthenticated request. Tests that need auth pass
  `auth_headers` explicitly; the
  `test_protected_endpoint_rejects_missing_auth` test
  actively exercises the no-header path and was the
  original catch for this fixture's behavior.

### 7.3 `tests/unit/graph/conftest.py`

Neo4j-specific fixtures. One fixture:

- **`neo4j_store`** (`scope="session"`) — builds a
  `Neo4jGraphStore` from `SEMRAG_TEST_NEO4J_URI` (default
  `bolt://localhost:7687`), calls
  `_driver.verify_connectivity()` to confirm Neo4j is
  reachable, and `pytest.skip`s if it isn't. Calls
  `ensure_constraints()` before yielding; closes on
  teardown.

The skip-gracefully pattern mirrors `tests/unit/conftest.py`
exactly. The Neo4j store is real code but unverified against
a live server in this codebase's development environment;
the moment a Neo4j instance is reachable (e.g. via the
`docker-compose.yml` the deployment will eventually build),
the tests run for real with zero code changes — just point
`SEMRAG_TEST_NEO4J_URI` at it.

---

## 8. Test utilities

There is no `tests/utils/` or `tests/conftest_helpers.py`
directory today. What looks like a utility is either:

- A `conftest.py` fixture (above) — a fixture is just a
  function pytest calls before each test; promoting one to
  "utility" is anti-pattern when `conftest.py` is the
  scoped, discoverable place for it.
- An in-script helper inside the test file (e.g. the
  `_QUERIES` tuple and the per-file fixture builders in
  `tests/perf/run_perf_suite.py`).
- A class-level constant. Some test files (e.g.
  `tests/load/locustfile.py`,
  `tests/load/run_load_test.py`) keep their queries,
  headers, and pacing constants at module top-level — they
  are not shared across files, so promoting them to
  `tests/conftest.py` would be premature.

The shared utilities that exist are:

- **`MockLLM`** in `src/semrag/reasoning/llm/mock_llm.py` —
  used by `tests/unit/llm/`, `tests/unit/agents/`, and the
  perf suite's in-process mode. Not in `tests/` itself; the
  reasoning layer ships the mock because it's useful in
  non-test contexts too (e.g. local development without an
  LLM).
- **The `db_session` / `pg_engine` fixtures** — the only
  utilities that span multiple subpackages.
- **The `api_client` / `auth_headers` fixtures** — span the
  entire `tests/unit/api/` tree.

This is the intended shape: a small number of cross-cutting
fixtures, and per-test-file helpers inlined. When a helper
grows to be used in five places, promote it to `conftest.py`
at the appropriate scope; until then, keep it local.

---

## 9. Mocking strategy

The suite has one rule for external dependencies and one
rule for internal collaborators:

### 9.1 External dependencies are mocked at the HTTP boundary

LLM providers and embedding providers talk HTTP. The unit
tests do not hit live endpoints; they inject mocked
responses (via `MockLLM` for the LLM, and via `httpx`-style
mock patterns for embedders). Two reasons, both explicit in
the relevant test docstrings:

- **CI determinism.** A CI run that talks to a real LLM is
  slow, flaky, costs money, and leaks prompt text. Tests
  should not exercise network code at all.
- **Provider isolation.** The provider's job is request
  shaping, retry policy, and response parsing. A
  mocked-HTTP test exercises exactly that surface; a
  live-LLM test exercises a model in addition to the
  provider, and a model regression would mask a provider
  bug.

`MockLLM` (in `src/semrag/reasoning/llm/mock_llm.py`) is
the canonical example — used by
`tests/unit/agents/test_agent_loop.py`,
`tests/unit/llm/test_factory.py`, and the perf suite's
in-process mode. The mock returns scripted
`CompletionResult`s in order; streaming emits one chunk
per response. No network, no real model.

The `OpenAICompatibleLLM` and `OllamaLLM` tests
(`tests/unit/llm/test_openai_compatible_llm.py`,
`tests/unit/llm/test_ollama_llm.py`) mock the HTTP layer
to validate request shaping, retry policy, and response
parsing against faked `requests` responses.

### 9.2 Internal collaborators are passed via constructor

The rest of the codebase depends on Protocols, not on
concrete classes. Tests take advantage of this: a test for
`AgentLoop` constructs it with a `MockLLM` and an empty
`ToolRegistry`; a test for the API constructs a
`TestClient` with overridden dependencies. The Protocol
boundary is the seam that makes this clean.

For the storage layer, the seam is the SQLAlchemy
`Session` — tests pass the `db_session` fixture, which
runs inside a transaction that rolls back on teardown.
No mocking of `PgvectorStore` etc.; the test hits the
real implementation against a real Postgres. The
implementation is small enough that mocking it would
be more code than just using it.

### 9.3 What the suite does **not** mock

- **PostgreSQL itself.** The storage tests use a real
  Postgres. SQLite is not a viable substitute because the
  tests exercise pgvector operators and recursive CTEs
  that SQLite does not implement. The skip-gracefully
  fixture keeps the suite runnable without one.
- **The vector store and the graph queries.** Tests use
  `PgvectorStore` and `PostgresGraphQueries` directly.
  Mocking them would test the mock, not the integration.
- **The FastAPI app's request-handling machinery.** API
  tests use `TestClient` and exercise the real app; only
  the DB session and the job store are overridden.

---

## 10. Database testing

The unit suite's database tests use a real PostgreSQL.
There's no SQLite fast-path, no in-memory Postgres mock.

### 10.1 The `semrag_test` database

Default URL:
`postgresql+psycopg2://semrag:semrag_test@localhost:5432/semrag_test`.

Override with `SEMRAG_TEST_DATABASE_URL`. The database
should be migrated via `alembic upgrade head` before
running the suite; the fixtures don't run migrations
themselves.

### 10.2 Transactional isolation

Every test that needs a `Session` gets one from the
`db_session` fixture. The fixture:

1. Opens a connection.
2. Begins a transaction.
3. Builds a `sessionmaker` bound to the connection.
4. Yields the session.
5. On teardown: `session.close()` →
   `transaction.rollback()` → `connection.close()`.

The session is **never committed**. This means:

- Tests don't need a per-test truncate.
- Tests can't poison each other through uncommitted
  writes (a future test starts in the same initial
  state as the first one).
- New tables don't need a "remember to truncate" line
  in the fixture.
- The `semrag_test` database stays empty across runs
  without a per-session reset.

The cost: tests can't use `COMMIT` themselves (the
fixture would roll back the work). That's by design —
a test that needs commit-and-discard behavior (e.g.
testing the worker's failure-recording path) is one
that tests `commit()` itself, and the test creates
its own session for that.

### 10.3 Reachability gate

If `pg_engine` can't reach the database, the entire
`pg_engine` fixture `pytest.skip`s with a clear message:

```
PostgreSQL not available at <URL>: <exception>
```

Storage, API, and integration tests that need Postgres
are skipped individually (the session-scoped fixture
fails fast for the whole process). This means a
developer without Postgres can still run the rest of
the suite — including IR, chunking, parsers,
retrieval unit tests, and the perf suite's in-process
mode.

### 10.4 What the DB tests do **not** do

- **Run migrations.** `alembic upgrade head` is the
  caller's responsibility.
- **Reset the schema between sessions.** The
  transactional isolation handles it.
- **Use SQLite.** The tests rely on pgvector
  operators, `ts_rank_cd`, recursive CTEs, and
  `INSERT ... ON CONFLICT DO UPDATE` — all
  Postgres-specific.
- **Test schema migrations.** Migration tests live
  in `alembic/versions/` (no `tests/` directory
  today); a new migration should add its own
  upgrade/downgrade test alongside.

---

## 11. API testing

`tests/unit/api/` is the largest subdirectory. The
fixtures are in [`tests/unit/api/conftest.py`](#72-testsunitapiconftestpy)
above; the test files cover:

- `test_auth.py`, `test_project_auth.py` — API key
  validation, project-level authorization
- `test_app_integration.py` — full FastAPI app
  end-to-end via `TestClient`
- `test_projects_jobs.py` — `/projects` and `/jobs`
  endpoints
- `test_answer.py` — `/answer` endpoint (full RAG
  pipeline)
- `test_query_cache.py`, `test_rate_limit.py` —
  cache + rate-limit middleware
- `test_middleware.py` — request ID, tracing, security
  headers, body-size cap
- `test_exception_handler.py` — the standard error
  envelope
- `test_audit.py` — the security audit log
- `test_health.py` — `/livez`, `/readyz`, `/version`
- `test_metrics.py` — Prometheus exposition format
- `test_startup.py`, `test_startup_validation.py` —
  app factory and `Settings` validation
- `test_enterprise_fixes.py` — historical
  regression suite for known production incidents

The `api_client` fixture builds the app via
`create_app()` and overrides `get_db_session` to
yield the `db_session` fixture. Tests get the same
transactional isolation the storage tests get,
without writing a second connection strategy.

`TestClient` is the only HTTP client used — no
`httpx.AsyncClient`, no live server. The exception
handler tests verify the JSON envelope shape; the
middleware tests verify headers; the auth tests
verify the 401/403 responses. Everything goes
through the real app.

---

## 12. Background job testing

`tests/unit/jobs/` covers the `semrag.jobs` package
(manager, store, executor, polling worker). Two test
shapes:

- **In-memory store tests** — the default. Use
  `SEMRAG_JOB_STORE_BACKEND=memory` (set automatically
  by the `api_client` fixture), build the store via
  `get_job_store()`, exercise the manager and
  executor. Fast, no Postgres required for the store
  itself.
- **Postgres store tests** — override the
  `api_client` fixture (or build their own
  `db_session`) to use the `database` backend.
  These are the ones that need a real Postgres
  and exercise the SQL-specific paths.

The job executor and the polling worker are tested
in-process via the in-memory store. The integration
test that exercises the Postgres store against a
live worker is in `tests/unit/api/test_projects_jobs.py`
(it goes through the API, which uses the real
DB session).

The fixture's `reset_job_store_for_tests()` is the
mechanism that makes "switch backends per test"
work: it tears down the module-level job store
singleton, the next `get_job_store()` rebuilds it
against the current `SEMRAG_JOB_STORE_BACKEND` env
var. Without this, the singleton would be bound to
whatever was the default at module import time and
the per-test override would be a no-op.

---

## 13. Running tests

The unit suite is the one CI runs. From the project
root:

```bash
# Everything
pytest tests/unit/

# A single subpackage
pytest tests/unit/retrieval/

# A single file
pytest tests/unit/llm/test_factory.py

# A single test
pytest tests/unit/llm/test_factory.py::test_ollama_provider

# With coverage
pytest tests/unit/ --cov=semrag --cov-report=term-missing
```

Useful flags:

- `-x` — stop on first failure
- `-v` — verbose (per-test output)
- `-k <pattern>` — only tests matching the pattern
- `--tb=short` — short traceback format
- `-p no:cacheProvider` — disable the pytest cache
  (faster CI cold-start)

The Postgres tests skip automatically if
`SEMRAG_TEST_DATABASE_URL` is unreachable. The
Neo4j tests skip automatically if
`SEMRAG_TEST_NEO4J_URI` is unreachable.

The perf suite is not a pytest test:

```bash
# In-process (no live services; default mode)
python tests/perf/run_perf_suite.py --output bench.json

# Live URL
python tests/perf/run_perf_suite.py --url http://localhost:8000 \
    --api-key dev-local-key --project rtl_uvm --output bench.json
```

The load tests are manual:

```bash
# Locust (production-grade)
pip install locust
locust -f tests/load/locustfile.py --host http://localhost:8000

# Dependency-free smoke test
python -m tests.load.run_load_test --url http://localhost:8000 \
    --concurrency 4 --duration 30 --json
```

---

## 14. Adding new tests

The convention:

1. **Mirror the source tree.** A new module under
   `src/semrag/<package>/<file>.py` should have its
   test at `tests/unit/<package>/test_<file>.py`.
   This is a project convention, not enforced by
   pytest, but it makes `find tests/unit/<package>/`
   work for "where do I add a test for X?"
2. **Use the existing fixtures.** A test that needs
   a real Postgres takes `db_session`. A test that
   needs the FastAPI app takes `api_client`. A test
   that needs Neo4j takes `neo4j_store`. New fixtures
   go in the appropriate `conftest.py`.
3. **Mock external HTTP, not internal code.** Use
   `MockLLM` for the LLM; mock `requests` for
   embedding providers. Don't mock `PgvectorStore`
   or `PostgresGraphQueries` — use the real
   implementation against a real DB.
4. **Never commit in tests.** The `db_session`
   fixture rolls back on teardown; a test that
   commits defeats the isolation. If a test needs
   commit semantics, build its own session.
5. **Prefer in-process over live service.** A test
   that can run via `TestClient` should. A test that
   needs a real Ollama should set
   `SEMRAG_LLM_PROVIDER=mock` (or use
   `MockLLM` directly) unless the test's *purpose*
   is to exercise the LLM HTTP client.
6. **Name integration tests with `_integration.py`.**
   The convention is documented in
   [§3.2](#32-integration-tests-inside-testsunit)
   and helps reviewers find the end-to-end tests
   quickly.
7. **No new top-level test directories.** The
   three top-level directories (`unit/`, `load/`,
   `perf/`) are the only ones. A new kind of test
   goes inside one of them, with a clear README
   if it grows large.

---

## 15. Common debugging paths

| Symptom                                                | Look first                                                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `PostgreSQL not available at <URL>...`                 | `tests/unit/conftest.py:pg_engine` skipped; start the local Postgres, or set `SEMRAG_TEST_DATABASE_URL` |
| `Neo4j not available at <URL>...`                      | `tests/unit/graph/conftest.py:neo4j_store` skipped; Neo4j is real-but-unverified; only `tests/unit/graph/` is affected |
| Tests pass in isolation, fail in batch                 | Usually an order-dependent shared state; check for uncommitted writes (the `db_session` fixture rolls back — but tests that build their own session don't) |
| `ToolCall` argument validation error in a test         | The LLM mock's `arguments` must be a `dict` (already-parsed), not a JSON string; `OpenAICompatibleLLM._parse_message` does the parsing in production, but the mock should hand a `dict` directly |
| `MockLLM exhausted` warning                            | The mock's scripted responses ran out; `mock.queue(...)` more, or use a longer fixture script |
| `TestClient` errors with `httpx` merge behavior        | Don't set a default `Authorization` on the client; pass `auth_headers` per-call (see `tests/unit/api/conftest.py:55-62`) |
| Job store tests are using Postgres in-memory fallback  | `reset_job_store_for_tests()` wasn't called; the singleton was bound at import time, the env-var change is a no-op |
| `TfidfEmbeddingProvider` test fails                    | The TFIDF provider fits on first use; the test must call `fit()` before `embed()` or pass pre-fitted chunks |
| `Embedding dimension mismatch`                         | `EMBEDDING_DIMENSION = 768` in `pgvector_models.py`; tests that build chunks need `len(embedding) == 768` |
| Recursive CTE hangs                                    | Cycle guard missing in `PostgresGraphQueries._traverse`; the storage package's bug, not a test problem |
| Perf suite numbers look wildly different across runs   | In-process mode shares the process with whatever else is running; close other processes, or run in a fresh shell |
| Load test returns all 429s                             | Rate limit is doing its job; the test is correct, the API is at capacity — back off concurrency |
| `pytest tests/unit/` runs forever                      | An integration test is doing real DB work across many small transactions; consider `pytest --co` to see what's collected, then narrow with `-k` |

---

## 16. Common mistakes

1. **Importing from `semrag.retrieval` instead of submodule paths.** Every `__init__.py` in `src/semrag/` is empty. Tests import from the submodule path (`semrag.retrieval.pipeline.HybridRetriever`), not the package root.
2. **Committing inside a test.** The `db_session` fixture rolls back on teardown; a `session.commit()` inside a test is wasted work that will be undone. If the test's *purpose* is to verify commit semantics, build a session manually.
3. **Forgetting to call `reset_job_store_for_tests()`.** The job store is a module-level singleton. A test that sets `SEMRAG_JOB_STORE_BACKEND=memory` without resetting the singleton will silently use the store that was built at import time.
4. **Mocking `PgvectorStore` or `PostgresGraphQueries`.** Use the real implementation. The tests are exercising the SQL; mocking would test the mock.
5. **Adding a test that requires a live Ollama.** LLM HTTP tests are against mocked responses. A test that needs a real Ollama should set `SEMRAG_LLM_PROVIDER=mock` unless its purpose is to exercise the LLM client.
6. **Setting a default `Authorization` header on the `TestClient`.** The `api_client` fixture deliberately doesn't; per-test `auth_headers` is the right pattern. See `tests/unit/api/conftest.py:55-62` for the original catch.
7. **Truncating tables in a fixture.** The transactional isolation handles it. A `TRUNCATE` in a fixture is redundant at best and order-dependent at worst.
8. **Promoting a one-off helper to `conftest.py` after the second use.** Three or more files is the threshold; before that, the helper lives in the test file that uses it. The promotion is cheap when the threshold is reached, expensive when premature.
9. **Running the perf suite in CI without pinning the baseline.** The in-process numbers are noisy across hosts and processes. The output is a JSON snapshot; a downstream dashboard does the trend analysis. CI should run the suite for "did it run" not "is it fast enough."
10. **Running the load suite against a real production API.** The load scripts don't differentiate between dev and prod. Use a dedicated load-test environment, or use the in-process perf suite instead.
11. **Using `pytest.skip` outside a fixture.** Inside a test, prefer `pytest.mark.skip(reason=...)` or `pytest.mark.skipif(...)`; the skip-gracefully pattern in the conftest fixtures is for "this whole subtree needs a service that isn't reachable."
12. **Forgetting that `MockLLM.stream` is a single chunk.** A test that loops on `StreamChunk` to build a typewriter UI will see the full response appear in one tick. Use a real provider for streaming UX tests.
13. **Adding tests to the perf suite's `in-process` mode that depend on live services.** The in-process mode uses TFIDF embeddings and `MockLLM`. A test that calls a real Ollama will fail in CI.
14. **Constructing `create_app()` in a fixture that runs before `monkeypatch` activates.** The `api_client` fixture sets env vars first, then calls `create_app()`. A test that constructs the app in a different order will pick up the wrong `Settings`.

---

## 17. Architectural boundaries

What the `tests/` directory **does not** do:

- **Test third-party libraries.** Provider tests
  exercise the *contract* of OpenAI / Ollama / etc.
  against faked HTTP, not against a live server.
- **Validate a deployed cluster.** That's
  [`docs/operations/`](../docs/operations/README.md).
- **Provide a benchmarking dashboard.** The perf
  suite emits JSON; a downstream tool visualizes.
- **Generate or migrate schemas.** Migrations are
  Alembic's. Tests use a pre-migrated `semrag_test`
  database.
- **Replace documentation.** Tests document
  behavior, not intent. The
  [`docs/`](../docs/README.md) tree is the
  source of truth for "why is it shaped this way."
- **Test the LLM's outputs.** The mock returns
  scripted responses; tests assert on
  request-shaping, retry policy, and message
  construction, not on whether the LLM answered
  the question correctly. That's the eval
  harness's job (`src/semrag/evaluation/`).
- **Run as part of `pytest tests/unit/`.** The
  perf and load suites are invoked manually. CI
  runs `pytest tests/unit/` only.

What it **does** own:

- The `db_session` transactional-isolation
  pattern and its fixture.
- The `api_client` / `auth_headers` FastAPI
  test fixtures and the no-client-default-auth
  rule that goes with them.
- The `MockLLM` location and behavior (it lives
  in `src/semrag/reasoning/llm/` because it's
  useful outside tests, but the tests are its
  primary consumer).
- The integration-test naming convention
  (`*_integration.py`).
- The skip-gracefully pattern for live-service
  fixtures.
- The perf suite's in-process / live-URL mode
  split.
- The load suite's Locust + dependency-free
  split.

---

## 18. Related documentation

- [`docs/developer_guide.md`](../docs/developer_guide.md)
  — testing patterns in the developer workflow
  context; the chapter on adding a new test for a
  new component.
- [`docs/CODEBASE_GUIDE.md`](../docs/CODEBASE_GUIDE.md)
  — the source tree map; the per-package test
  directories mirror the source tree.
- [`docs/operations/README.md`](../docs/operations/README.md)
  — on-call runbooks. The on-call job is *not*
  running tests; it's resolving live incidents
  using diagnostics the tests taught the team to
  trust.
- [`tests/load/README.md`](./load/README.md) —
  the load-test README, the source of truth for
  the load scripts.
- [`tests/perf/README.md`](./perf/README.md) —
  the perf-test README, the source of truth for
  the perf suite.
- [`src/semrag/storage/README.md`](../src/semrag/storage/README.md)
  — the storage layer that the `db_session`
  fixture targets; the dependency-injection
  conventions the `api_client` fixture relies
  on.
- [`src/semrag/api/README.md`](../src/semrag/api/README.md)
  — the HTTP API; `TestClient` exercises the
  real app, the `api_client` fixture overrides
  the DB session.
- [`src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md)
  — the LLM-facing layers; `MockLLM` lives here
  because it's useful outside tests too.
- [`src/semrag/evaluation/README.md`](../src/semrag/evaluation/README.md)
  — the eval harness; the only place that
  asserts on LLM answer *quality* (the unit
  tests assert on LLM *behavior*).
- [`src/semrag/services/README.md`](../src/semrag/services/README.md)
  — the orchestrators that the integration
  tests exercise end-to-end.

---

## Skipped (and why)

- **SQLite fast-path.** The storage tests rely
  on pgvector operators, `ts_rank_cd`, recursive
  CTEs, and `INSERT ... ON CONFLICT DO UPDATE`.
  SQLite doesn't implement any of them. The
  skip-gracefully fixture keeps the suite
  runnable without one.
- **Live LLM/embedding tests.** CI doesn't make
  network calls. Provider tests mock HTTP; the
  perf suite's in-process mode uses
  TFIDF + `MockLLM`; the perf suite's live-URL
  mode is a manual invocation, not a CI gate.
- **A `tests/utils/` or `tests/helpers/`
  directory.** Premature abstraction. Shared
  utilities are conftest fixtures; per-test
  helpers are inlined in the test file.
- **A separate `tests/integration/` top-level
  directory.** The unit-test directory contains
  both unit and integration tests, distinguished
  by `*_integration.py`. Putting them in the
  same tree keeps CI's command simple and lets
  the same fixture scope apply.
- **A pytest-based perf suite.** The perf suite
  is a `python` script, not a pytest test. CI
  invokes it directly. The output is a JSON
  snapshot; a downstream dashboard does the
  trend analysis.
- **Truncate-table fixtures.** The
  transactional-isolation `db_session` fixture
  makes them redundant.
- **Per-test-class setup / teardown.** The
  per-function `db_session` fixture is enough.
  Tests that need session-level setup (e.g.
  the `pg_engine` fixture) use pytest's
  `scope="session"` for that one resource.
- **Coverage gating in CI.** Coverage reports
  are useful for local development (`pytest
  --cov`), but the CI gate is "all tests pass,"
  not "X% covered." Coverage tools tell you
  what's *not* tested; they don't tell you
  whether the tests are *good*.
- **A test for the eval harness's LLM-judge
  output.** The eval harness's job is to
  measure LLM answer quality; testing the
  measurement with a mock LLM would test the
  mock, not the harness. Add a real-benchmark
  integration test instead, against a canned
  benchmark set (the harness ships with one).
- **A test for the perf suite's percentile
  math.** The `_percentile` function in
  `tests/perf/run_perf_suite.py` is pure; a
  unit test would be one-line. Add when the
  perf suite grows a second mode that uses
  different percentile math.
- **A test for the load suite's job-store
  integration.** The load suite targets a
  running API, not a test fixture. The job
  store's behavior under load is a
  load-test outcome, not a unit-test one.

---

## Add when

- **A new subpackage under `src/semrag/`.** Add
  a `tests/unit/<subpackage>/` directory. Mirror
  the source tree.
- **A new integration test.** Add a
  `tests/unit/<package>/test_<thing>_integration.py`.
  Use the existing fixtures; do not build a
  parallel integration framework.
- **A new live-service dependency.** Add a
  skip-gracefully fixture in `conftest.py`,
  mirroring `tests/unit/conftest.py:pg_engine`
  or `tests/unit/graph/conftest.py:neo4j_store`.
  The pattern is identical for every service.
- **A new LLM or embedding provider.** Add a
  mocked-HTTP test under
  `tests/unit/llm/test_<provider>.py` and a
  factory-dispatch test under
  `tests/unit/llm/test_factory.py`. Do not
  add a live-server test.
- **A new shared fixture used in 3+ test
  files.** Promote it to the appropriate
  `conftest.py`. Below 3, keep it local.
- **A new test directory inside `tests/unit/`.**
  Only when a new package warrants it; the
  current 18 directories mirror the source
  tree 1:1.
- **A new top-level test directory under
  `tests/`.** Only when the test kind is
  substantively different from the three
  existing ones (unit, load, perf). A new
  "smoke" or "regression" directory should
  instead go inside `tests/unit/` with a
  clear `__init__.py` or `conftest.py`.
- **Coverage reporting in CI.** Add a
  coverage step; pair with a baseline. The
  baseline is the current coverage — set it
  to fail CI on `coverage < baseline` and
  raise the baseline as the test suite
  grows.
- **A real-benchmark integration test for
  the eval harness.** Add a
  `tests/unit/evaluation/test_harness_real_benchmark.py`
  that runs a canned benchmark against the
  real `HybridRetriever` and the real
  citation validator. The mock LLM returns
  pre-scripted "grounded" answers; the test
  asserts that the validator flags the
  ungrounded ones. Add when the eval
  harness grows a second canned benchmark.
- **A perf suite baseline-tracker.** Pair
  the perf suite's JSON output with a
  baseline JSON and fail on
  `metric > baseline * threshold`. Add when
  the in-process perf suite's noise floor
  is well-characterized.
- **A second perf-suite mode.** The script
  already has in-process and live-URL. Add
  a third mode (e.g. profile-guided) only
  when a new measurement is needed.
- **A second load-suite script.** The
  Locust + dependency-free split is
  sufficient for the current load shapes.
  Add a new script only when a new shape
  (e.g. burst traffic, soak test) needs
  its own pacing.
