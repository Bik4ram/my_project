# `semrag.storage` — Persistence layer

Persistence adapters for the three planes the rest of the codebase reads
from and writes to: relational (Postgres), vector (pgvector), and graph
(Neo4j, experimental). Every adapter is behind a Protocol so callers do
not bind to a specific engine, and every Postgres repository takes a
`Session` so the caller controls the transaction boundary — not the
storage layer.

This is **not** an API reference, **not** a deployment guide, and **not**
the place to learn how to wire `Settings` — see
[`docs/developer_guide.md` §6](../../../docs/developer_guide.md#6-storage-layer) for
the latter. This document describes what the storage package owns, what
it explicitly does not own, and how the pieces fit together.

---

## 1. Purpose of the package

The rest of SemRAG must not depend on Postgres, pgvector, or Neo4j
directly. `semrag.storage` exists to:

- **Hide the driver.** Callers see `VectorStore.search()`,
  `IRGraphRepository.save()`, `PostgresGraphQueries.transitive_dependencies()`,
  not raw SQL, raw `MERGE` Cypher, or pgvector operators.
- **Own schema definitions.** `storage/postgres/models.py` is the single
  source of truth for the relational schema; every other file imports
  its tables from there.
- **Centralise transaction conventions.** `storage/postgres/session.py`
  is the one place the engine, pool sizing, and `session_scope`
  context manager are assembled. Nothing else builds an engine.
- **Define the swappable surfaces.** The `VectorStore` and `GraphStore`
  Protocols are the contracts a new backend must satisfy; the
  relational repositories are concrete (no Protocol yet — see
  [§10 Protocols vs concrete implementations](#10-protocols-vs-concrete-implementations)).

---

## 2. Package structure

```
storage/
├── __init__.py            (empty — no re-exports)
├── postgres/
│   ├── __init__.py        (empty)
│   ├── session.py         engine, sessionmaker, session_scope
│   ├── models.py          Base + Project / ScanManifestRow / FileRow /
│   │                       IRNodeRow / IREdgeRow / ParseDiagnosticRow
│   ├── manifest_store.py  PostgresManifestStore
│   ├── ir_repository.py   IRGraphRepository
│   ├── diagnostics_repository.py  DiagnosticsRepository
│   └── graph_queries.py   PostgresGraphQueries (recursive-CTE traversal)
├── vector/
│   ├── __init__.py        (empty)
│   ├── interfaces.py      VectorStore Protocol
│   ├── models.py          VectorChunk / VectorSearchFilter / VectorSearchResult
│   ├── pgvector_models.py ChunkRow (SQLAlchemy ORM, pgvector column,
│   │                       HNSW + GIN indexes)
│   └── pgvector_store.py  PgvectorStore (implements VectorStore)
└── graph/
    ├── __init__.py        (empty)
    ├── interfaces.py      GraphStore Protocol + GraphPathResult dataclass
    └── neo4j_store.py     Neo4jGraphStore (experimental, see §7)
```

Every `__init__.py` in this package is empty. Callers always import from
the submodule path (e.g. `semrag.storage.postgres.session`,
`semrag.storage.vector.pgvector_store`), never from `semrag.storage`
directly. This is intentional — it makes grep-for-callers work without
fighting re-exports.

---

## 3. Responsibilities of every module

### `postgres/session.py`

- `get_database_url()` — reads `SEMRAG_DATABASE_URL`, falls back to
  `postgresql+psycopg2://semrag:semrag@localhost:5432/semrag`.
- `build_engine(...)` — constructs the SQLAlchemy `Engine` with
  `pool_pre_ping=True` plus pool sizing from `Settings` (`db_pool_size`,
  `db_max_overflow`, `db_pool_recycle`, `db_pool_timeout`).
- `build_session_factory(...)` — wraps the engine in a `sessionmaker`
  with `expire_on_commit=False`.
- `session_scope(...)` — context manager. Commits on clean exit,
  rolls back on exception, always closes. Provided for callers that
  don't need to share a session across multiple repository calls.

The module **does not** register schema or run migrations — that is
Alembic's job (see [§6.1 Postgres](#61-postgres)). It also **does not**
decide *when* to commit; that decision lives with the caller (see
[§9 Transaction ownership](#9-transaction-ownership)).

### `postgres/models.py`

The declarative `Base` plus the eight tables. Three design choices
matter:

- **Deterministic PKs on `ir_nodes.id` and `ir_edges.id`.** Both are
  natural keys produced by the IR builder's `id_scheme.py`. Every
  write is therefore an upsert (`INSERT ... ON CONFLICT DO UPDATE`),
  not an insert.
- **JSONB-with-variant.** A `_JSONType = JSON().with_variant(JSONB(), "postgresql")`
  means the schema can be unit-tested against SQLite (the JSON
  variant) while still using real JSONB + GIN indexes on Postgres.
- **JSONB only for unindexed payload shapes.** `attributes`,
  `candidate_ids`, `git_metadata`, and `raw_dependency_refs` are JSONB
  because no retrieval-stage query filters on their contents. Everything
  retrieval filters on — `project_id`, `kind`, `relative_path`,
  `resolved`, `severity` — is a typed column with an index.

`ScanManifestRow` is append-only audit; `FileRow` is current-state; the
two are deliberately not merged into a single JSONB blob (see
`PostgresManifestStore.get_latest_manifest` for why). `ParseDiagnosticRow`
is not a foreign key to `ir_nodes.id` on purpose: a file that fails to
parse entirely has diagnostics and *no* node.

### `postgres/manifest_store.py`

`PostgresManifestStore` — Postgres implementation of the scanner's
`ManifestStore` Protocol. Two methods:

- `get_latest_manifest(project_id)` — joins the latest `ScanManifestRow`
  with all `FileRow` rows for that project, reconstructs the in-memory
  `ScanManifest` shape the scanner expects. Returns `None` when no scan
  has been recorded.
- `save_manifest(manifest)` — upserts the manifest header and every
  file in it (one statement per file), then deletes any `FileRow` for
  this project whose `relative_path` is not in the manifest.

The `_sync_files` ordering matches `IRGraphRepository.save`: present
state wins, missing state is deleted (not retained as a "history" of
files).

### `postgres/ir_repository.py`

`IRGraphRepository` — owns the `ir_nodes` and `ir_edges` tables.

- `save(graph)` — upserts nodes, then upserts edges; deletes stale
  edges before stale nodes are pruned (FK ordering, see [§9 Transaction
  ownership](#9-transaction-ownership)).
- `load(project_id)` — reconstructs an `IRGraph` from stored rows.
  Used by tests for round-trip verification; available to any future
  caller that needs the graph without re-running the IR builder.
- `bulk_delete_stale_nodes(project_id, keep_ids)` — bulk delete by
  exclusion set; returns the rowcount for observability. The caller
  is responsible for stale-edge ordering if it uses this in isolation.

`_ensure_project` inserts a placeholder `Project` row if the
`project_id` hasn't been seen yet, so a fresh index call doesn't
violate the `ir_nodes.project_id → projects.project_id` FK.

### `postgres/diagnostics_repository.py`

`DiagnosticsRepository` — append-only diagnostics per scan.

- `record(project_id, scan_id, results)` — flattens `ParseResult.diagnostics`
  into `ParseDiagnosticRow` rows in one `add_all`.
- `latest_diagnostics_for_project(project_id)` — finds the latest
  `scan_id` by `started_at`, returns the matching diagnostic rows.

Old diagnostics are **not** deleted. Queries filter to the latest scan
rather than mutating history; the trade-off is documented in the
module docstring (history is cheap; deleting it would lose the
"did this file's diagnostics get worse over time" signal).

### `postgres/graph_queries.py`

`PostgresGraphQueries` — graph traversal over `ir_nodes` / `ir_edges`
using recursive CTEs. Three methods, all sharing one `_traverse`
implementation:

- `transitive_dependencies(project_id, start_node_id, max_depth=10)` —
  follows edges forward (source → target).
- `transitive_dependents(project_id, start_node_id, max_depth=10)` —
  follows edges backward.
- `shortest_path(project_id, source_id, target_id, max_depth=15)` —
  BFS by edge count via the first arrival at `target_id`.

Both recursive terms carry an explicit cycle guard
(`NOT node_id = ANY(visited_ids)`). This is mandatory: a recursive CTE
over a cyclic `ir_edges` graph does not terminate, and RTL hierarchies
with circular instantiation are a real (if unusual) bug pattern, not a
theoretical one.

This is the **production** graph backend. `Neo4jGraphStore` exists but
has no production caller — see [§7 Graph storage](#7-graph-storage).

### `vector/interfaces.py`

`VectorStore` Protocol. Four methods: `upsert_chunks`,
`delete_chunks`, `delete_by_project`, `search`. The Protocol is what
makes pgvector-vs-Qdrant a swappable implementation detail; see
[§10 Protocols vs concrete implementations](#10-protocols-vs-concrete-implementations).

### `vector/models.py`

Pydantic models for the vector plane:

- `VectorChunk` — the minimal shape storage needs: id, project, path,
  type, source-node reference, text, line range, embedding.
- `VectorSearchFilter` — optional `project_id`, `chunk_type`,
  `relative_path`.
- `VectorSearchResult` — what `search()` returns. `score` is
  `1 - cosine_distance` (higher-is-better) so retrieval fusion can
  treat all signals uniformly.

### `vector/pgvector_models.py`

The `ChunkRow` SQLAlchemy model. Notable details:

- `EMBEDDING_DIMENSION = 768` is hardcoded in this file (column type
  is `Vector(768)`).
- A Postgres `TSVECTOR` **generated** column over `text` (`to_tsvector('english', text)`)
  backs lexical/BM25-style search (real Postgres FTS — `ts_rank_cd`,
  not literal BM25).
- Three indexes: a composite `(project_id, chunk_type)`,
  `HNSW (vector_cosine_ops)` on `embedding`, and `GIN` on `text_search`.

The dimension is a real schema constraint, not a config flag. Changing
it requires a migration **and** a full re-embed of every project's
chunks. See [§11 Extension points](#11-extension-points) for what
changing it would actually cost.

### `vector/pgvector_store.py`

`PgvectorStore` — implements `VectorStore`. Methods:

- `upsert_chunks(chunks)` — single `INSERT ... ON CONFLICT DO UPDATE`
  over the chunk batch. Strips `\x00` from `chunk.text` before binding
  (Postgres TEXT/VARCHAR rejects NUL). Defends against the chunker
  ever mis-classifying a binary file.
- `upsert_chunks_bulk(chunks, *, page_size=500)` — pages the work to
  stay inside one statement's parameter limit on very large ingestions.
  Same semantics, more round-trips.
- `delete_chunks(chunk_ids)`, `delete_by_project(project_id)`,
  `delete_by_paths(project_id, relative_paths)` — single SQL DELETE
  each; `delete_by_paths` is the bulk form used when the scanner
  reports deleted files.
- `search(query_vector, top_k, filters)` — orders by `cosine_distance`
  ascending, converts to `score = 1 - distance`. Filters apply to
  `project_id`, `chunk_type`, `relative_path` only — those are the
  only indexed filter columns.

### `graph/interfaces.py`

`GraphStore` Protocol with three methods:
`transitive_dependencies`, `transitive_dependents`, `shortest_path` —
the same three questions `PostgresGraphQueries` answers. Same questions
on purpose: it makes the two implementations genuinely comparable.

### `graph/neo4j_store.py`

`Neo4jGraphStore` — Neo4j driver v5+ implementation. Methods:

- `ensure_constraints()` — idempotent `CREATE CONSTRAINT IF NOT EXISTS`
  on `(project_id, id)` of `IRNode`. Called once at startup; safe to
  repeat.
- `sync_graph(graph)` — `MERGE` nodes, `MERGE` edges, then a bounded
  `DETACH DELETE` pass for nodes/edges belonging to the project that
  weren't touched by this sync (same upsert-and-prune philosophy as
  the Postgres repositories).
- `_traverse(...)`, `transitive_dependencies`,
  `transitive_dependents`, `shortest_path` — match the
  `PostgresGraphQueries` surface. Variable-length hop counts
  (`*1..N`) are interpolated as integer literals (Cypher cannot bind
  them as parameters); `max_depth` is validated as a positive int
  before interpolation.

**Important:** this module is **written against the real Neo4j Python
driver but not run against a live Neo4j server** — see the module
docstring and [§7 Graph storage](#7-graph-storage). The corresponding
unit test is currently skipped.

---

## 4. Storage architecture

Three planes, three Protocol contracts (one already, one for vectors,
one for graphs), one shared relational plane with concrete repository
classes:

```
┌─────────────────────────────┐
│        callers              │
│  (services, retrieval,      │
│   jobs/worker, api routers) │
└──────────┬──────────────────┘
           │
           ├── Postgres plane ──────────────────────────────┐
           │   IRGraphRepository                             │
           │   PostgresManifestStore                         │
           │   DiagnosticsRepository                        │
           │   PostgresGraphQueries                         │
           │        │                                        │
           │   session.Session (sqlalchemy.orm)             │
           │        │                                        │
           │   SQLAlchemy Engine ──► PostgreSQL + pgvector   │
           │                                             (chunks table)
           │
           ├── Vector plane ────────────────────────────────┤
           │   PgvectorStore ─► VectorStore Protocol         │
           │        │                                        │
           │   SQLAlchemy Session ──► PostgreSQL (chunks)    │
           │
           └── Graph plane ─────────────────────────────────┘
               Neo4jGraphStore ─► GraphStore Protocol
                       │
               neo4j driver ──► Neo4j (experimental, unwired)
```

Callers never construct the engine, sessionmaker, or driver — they
receive them via dependency injection (see
[`src/semrag/api/README.md`](../api/README.md) for the DI wiring, and
[`src/semrag/services/README.md`](../services/README.md) for the
service layer that wires repositories into ingest and query).

Two details that look architectural but are actually just conventions:

- **Repositories take a `Session`, not the other way around.** Every
  `IRGraphRepository(...)`, `PostgresManifestStore(...)`,
  `PgvectorStore(...)` constructor takes a live `Session` and uses it
  for the duration of one method call. The caller decides when the
  session starts, when it commits, when it closes.
- **Graph traversal lives in storage, not retrieval.** `PostgresGraphQueries`
  is here because it's SQL over `ir_edges`. The *use* of that
  traversal (graph expansion during retrieval) is in
  [`src/semrag/retrieval/`](../retrieval/README.md) — the
  boundary is "SQL goes in storage, ranking goes in retrieval."

---

## 5. Database/session lifecycle

Every Postgres path goes through one `Engine` and one
`sessionmaker[Session]` (built by `build_engine` /
`build_session_factory`). The sessionmaker is configured with
`expire_on_commit=False` so ORM objects remain usable after the
surrounding transaction commits.

Three lifetime patterns, depending on caller:

### 5.1 Single-request, single-transaction (services, query handlers)

`src/semrag/services/services.py` opens one `Session` at the start of an
ingest call, passes it to every repository (`IRGraphRepository`,
`PgvectorStore`) for the duration of the pipeline, and commits once
at the end of the `PERSISTING` stage. See `services.py:238` for the
single `commit()` boundary; on exception, `services.py:417` rolls back.

### 5.2 One session per request (API key store)

`PostgresAPIKeyStore` keeps a `sessionmaker` and calls
`self._session_factory()` per operation. See `api/api_key_store.py`.

### 5.3 One session per pipeline (worker)

`src/semrag/jobs/worker.py` opens one `Session` for the entire indexing
pipeline, hands it to the repositories, commits once at the end. On
exception, `_report_failure` opens a **fresh** session to record
`FAILED` so the rollback can't take the failure log down with it
(documented in [`src/semrag/jobs/README.md`](../jobs/README.md#11-failure-handling)).

### 5.4 `session_scope` for ad-hoc callers

`session_scope()` exists for code paths that don't need to share a
session across multiple repositories — callers that just want
"open, do one thing, commit-or-rollback, close." It is **not** used by
the ingest or query paths because they need to share one transaction
across multiple repositories.

---

## 6. Vector storage

### 6.1 Postgres

The vector store is a single table (`chunks`) in Postgres. Why pgvector
over Qdrant is a deliberate V1 decision grounded in an actual benchmark
run (numbers and caveats in the `PgvectorStore` module docstring) —
relevant here: the table lives in the same Postgres database as
everything else, so a single transaction covers IR + chunks + diagnostics.

Key constraints and indexes:

- `embedding Vector(768)` — fixed dimension, enforced by Postgres.
  Changing it requires a migration **and** a full re-embed.
- `text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`
  — Postgres-maintained full-text search column; populated on every
  insert/update, never written to directly.
- HNSW index on `embedding` with `vector_cosine_ops` — cosine distance
  is what most code-embedding models are trained for.
- GIN index on `text_search` — backs the lexical/BM25-style retrieval
  signal used alongside vector search (see
  [`src/semrag/retrieval/lexical_search.py`](../retrieval/lexical_search.py)).

### 6.2 Operations

| Operation               | Method                                       | Transactional? |
| ----------------------- | -------------------------------------------- | -------------- |
| Insert/update chunks    | `PgvectorStore.upsert_chunks`                | Yes (caller's) |
| Bulk insert w/ paging   | `PgvectorStore.upsert_chunks_bulk`           | Yes (caller's) |
| Delete specific chunks  | `PgvectorStore.delete_chunks`                | Yes (caller's) |
| Delete whole project    | `PgvectorStore.delete_by_project`            | Yes (caller's) |
| Delete by file paths    | `PgvectorStore.delete_by_paths`              | Yes (caller's) |
| Top-k similarity search | `PgvectorStore.search`                       | Yes (caller's) |

Every method takes or shares the caller's `Session`; none of them
commit. The caller controls the transaction boundary.

### 6.3 Search semantics

`search()` returns `VectorSearchResult` ordered by `cosine_distance`
ascending, with `score = 1 - cosine_distance` so callers work in
higher-is-better uniformly. The filter fields are limited to indexed
columns (`project_id`, `chunk_type`, `relative_path`); richer filters
would require SQL directly, not this Protocol.

### Skipped (and why)

- **Per-row similarity threshold.** Filtering by score threshold is
  deliberately left to the caller — `services/services.py:retrieve()`
  applies `top_k` and the re-ranker, not `PgvectorStore.search`.
- **Multi-vector / sparse-vector search.** pgvector is dense-only. If
  sparse (SPLADE / BM25 alone) becomes a first-class signal at scale,
  that's a new Protocol implementation, not a pgvector feature.
- **Vector cache.** Not part of this layer. The retrieval layer's
  `QueryCache` lives in `src/semrag/api/` (see
  [`docs/developer_guide.md` §6.4](../../../docs/developer_guide.md#64-caches-and-rate-limiter-api)).

---

## 7. Graph storage

Two implementations behind one Protocol (`GraphStore`), three methods
each (`transitive_dependencies`, `transitive_dependents`,
`shortest_path`):

### 7.1 `PostgresGraphQueries` — production

Recursive CTEs over the existing `ir_edges` table. No new storage
system, no schema migration, no extra operational surface. The same
`Session` used for IR writes can serve traversals, so a single
transaction covers both.

This is the implementation used by
[`src/semrag/retrieval/graph_expansion.py`](../retrieval/graph_expansion.py)
when graph expansion is enabled.

### 7.2 `Neo4jGraphStore` — experimental

Real Neo4j driver v5+ code. Standard Cypher — `MERGE` for upserts,
`CREATE CONSTRAINT IF NOT EXISTS` for idempotent startup, recursive
variable-length patterns for traversal. The implementation mirrors
`PostgresGraphQueries` deliberately so the two answer the same three
questions.

**It is not wired into the retrieval path.** No production code
constructs a `Neo4jGraphStore` or passes it anywhere. Retrieval
expansion only ever sees `PostgresGraphQueries`. The unit test for
this module (`tests/unit/graph/test_neo4j_graph_store.py`) is
currently skipped — the module was written against the real driver
API but never run against a live Neo4j server in this codebase's
development environment (the module docstring has the full
explanation).

### 7.3 Why both?

`PostgresGraphQueries` answers the question this part was scoped to
answer: for the traversal depths this system's actual data has (RTL
hierarchies are rarely deeper than 5–10 levels), does a dedicated
graph database earn its operational cost? The benchmark in the design
doc is the answer; the conclusion is "not at this scale, not yet."
`Neo4jGraphStore` exists so the comparison is empirical and not
rhetorical — there's a real second implementation to point at, not
just an argument.

### Skipped (and why)

- **A `GraphStore` Protocol usage in `retrieval/graph_expansion.py`.**
  The retrieval module currently imports `PostgresGraphQueries`
  directly. Wiring it through the Protocol would add an indirection
  with no production caller on the other side; it would be a
  placeholder, not an extension point. When a deployment actually
  switches to Neo4j, switch the import to a Protocol-resolved type.
- **Cypher passthrough.** The Protocol does not expose a `query(...)`
  method. Every caller should go through one of the three named
  operations; a general Cypher escape hatch would let retrieval
  silently bind to Neo4j-specific syntax.
- **Graph materialisation.** Both implementations return
  `GraphPathResult` lists — they answer "who depends on X" but do not
  materialise the reverse direction. That's `transitive_dependents`'s
  job.

---

## 8. Repository responsibilities

| Class                          | Owns                                              | Does NOT own                                |
| ------------------------------ | ------------------------------------------------- | ------------------------------------------- |
| `PostgresManifestStore`        | `scan_manifests`, `files` (current state)         | History of file deletions                   |
| `IRGraphRepository`            | `ir_nodes`, `ir_edges`                           | Project metadata, scan metadata             |
| `DiagnosticsRepository`        | `parse_diagnostics`                              | Latest scan selection logic (delegated to `ScanManifestRow`) |
| `PostgresGraphQueries`         | Read-side traversal over `ir_edges`              | Writes to `ir_edges` (delegated to `IRGraphRepository`) |
| `PgvectorStore`                | `chunks`                                          | Embedding generation, project metadata     |
| `Neo4jGraphStore`              | `IRNode`, `DEPENDS_ON` labels/relationships       | Production wiring (not currently used)      |

A repository never reaches into another repository's table. Cross-table
coordination is the caller's job — and that is exactly the role of
`services.py`'s ingest pipeline.

---

## 9. Transaction ownership

The storage layer **does not** own transactions. It owns *how* a
repository's SQL runs; the caller owns *when* it commits. This is
visible in three places:

1. **Repositories take a `Session` parameter.** They never call
   `sessionmaker()`, never call `commit()`, never call `rollback()`.
   The Session comes from outside.
2. **`session_scope` is a convenience, not the default.** Ingest and
   query paths share one session across multiple repositories so the
   whole operation is atomic; `session_scope` would force one
   transaction per repository and break that property.
3. **Upsert-and-prune ordering is enforced *within* a repository, not
   across repositories.** `IRGraphRepository.save` deletes stale
   edges, then upserts nodes, then upserts edges — all in one
   transaction owned by the caller. `PostgresManifestStore._sync_files`
   upserts then deletes in one transaction. The two operations never
   share a transaction; if they did, the FK relationship between
   `ir_nodes` and `ir_edges` would still be honoured, but no caller
   does this today.

### 9.1 Why ordering is load-bearing inside `IRGraphRepository.save`

`ir_edges` has FK columns pointing to `ir_nodes.id` (`source_id`,
`target_id`). If you delete stale edges *after* pruning stale nodes,
you can hit a FK violation on the about-to-be-deleted nodes. If you
upsert edges *before* nodes, an `ON CONFLICT DO UPDATE` on an edge can
reference a node that hasn't been inserted yet. Both have to happen on
opposite sides of the node sync, which is what
`save(self, graph)` does:

```text
1. _delete_stale_edges(graph)         # FK-safe deletion first
2. _sync_nodes(graph)                 # nodes in
3. _upsert_edges(graph)               # edges last
```

The same shape applies to `PostgresManifestStore._sync_files`, though
without the FK subtlety (no table refers to `files.relative_path`).

### 9.2 Cancellation / rollback during ingest

The ingest pipeline commits once at the end of `PERSISTING`. If a
stage fails or the worker detects cancellation, the whole transaction
rolls back — there is no partial-save state to recover. The failure
itself is then recorded in a *separate* session by the worker
(`_report_failure`), so a rollback in the pipeline can't take the
failure log down with it. See
[`src/semrag/jobs/README.md` §11](../jobs/README.md#11-failure-handling).

---

## 10. Protocols vs concrete implementations

Two Protocols today, both for *non-relational* planes:

| Protocol       | Location                              | Production impl       | Experimental impl    | Used by                                               |
| -------------- | ------------------------------------- | --------------------- | -------------------- | ----------------------------------------------------- |
| `VectorStore`  | `storage/vector/interfaces.py`        | `PgvectorStore`       | —                    | `retrieval/vector_search_adapter.py`, `services.py` ingest, jobs worker |
| `GraphStore`   | `storage/graph/interfaces.py`         | `PostgresGraphQueries`* | `Neo4jGraphStore`  | `retrieval/graph_expansion.py` (currently `*` directly) |

\* `PostgresGraphQueries` is **not** formally a `GraphStore` — its
method signatures happen to match but it does not declare the Protocol
and is imported by name. Wiring it through the Protocol is future work;
see [§11 Extension points](#11-extension-points).

### 10.1 Why the relational plane has no Protocol

The relational repositories (`IRGraphRepository`, `PostgresManifestStore`,
`DiagnosticsRepository`, `PostgresGraphQueries`) are concrete classes,
not Protocols. Two reasons:

1. **There is no second implementation.** Postgres is the only
   relational target. Adding a Protocol with one implementation is the
   classic "interface for one implementation" anti-pattern flagged in
   the project's coding standards.
2. **Schema is shared with the vector plane.** `chunks` (pgvector) and
   `ir_nodes` / `ir_edges` live in the same database. Swapping the
   relational plane would mean swapping the database — which would
   also mean rewriting `PgvectorStore`.

If a future driver appears (e.g. a SQLite test backend), it should
extend each repository's surface directly rather than introduce a
Protocol just to satisfy "everything has an interface."

### 10.2 What the Protocols protect

- **Vector backend.** A `QdrantStore` or `WeaviateStore` implementing
  `VectorStore` drops in without changes to `retrieval/`,
  `services.py`, or the worker — the call sites all depend on the
  Protocol, not on `PgvectorStore`. Schema portability is the caller's
  problem (Qdrant won't auto-migrate Postgres rows); the
  *application-level* swap is one constructor change.
- **Graph backend.** If a deployment runs Neo4j, the swap is a
  construction-site change once `retrieval/graph_expansion.py` is
  wired to accept either. Today, it does not — `PostgresGraphQueries`
  is imported by name. See [§11](#11-extension-points).

---

## 11. Extension points

The seams below are the only places new backends, models, or
behaviours should attach. Anywhere else is a layering violation.

| Need                                   | Touch                                                            | What to add                                          |
| -------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| New vector backend                     | `storage/vector/`                                                | `<name>_store.py` implementing `VectorStore`         |
| New graph backend                      | `storage/graph/`                                                 | `<name>_store.py` implementing `GraphStore`          |
| New table for current-state data       | `storage/postgres/models.py`                                     | New ORM class + repository module                    |
| New append-only audit log              | Same as above, but query by `started_at` desc, not current-state  | Same as above, plus repository with `record()` only  |
| Change embedding dimension             | `storage/vector/pgvector_models.py` (`EMBEDDING_DIMENSION`) + `core/settings.py` (`embedding_dimension`) + **migration + full re-embed** | See below |
| Switch to Neo4j in production          | `retrieval/graph_expansion.py`                                    | Replace the `PostgresGraphQueries` import with a Protocol-resolved injection (and a `Neo4jGraphStore` factory call site) |
| Add a new SQL pattern over IR          | `storage/postgres/`                                              | A new module (e.g. `graph_queries.py` was that)      |

### Changing embedding dimension (real cost, not a config knob)

`EMBEDDING_DIMENSION = 768` is hardcoded in `pgvector_models.py` as a
column type (`Vector(768)`). To change it:

1. Add a Postgres migration that drops the HNSW index, the column, and
   recreates them at the new dimension. PGVector's `ALTER TABLE ...
   ALTER COLUMN ... TYPE vector(N)` is supported but requires the
   index drop.
2. Update `EMBEDDING_DIMENSION` in `pgvector_models.py`.
3. Update `embedding_dimension` in `core/settings.py`.
4. Update `startup_validation.py`'s `SEMRAG_EMBEDDING_DIMENSION`
   guard.
5. **Re-embed every project's chunks.** There is no online dimension
   migration; old chunks are unusable.

This is intentionally not driven by `Settings.embedding_dimension`
alone — pgvector requires the dimension at table-creation time, and
`core/settings.py`'s value is for *validation* against the embedding
provider, not schema generation.

---

## 12. Common debugging paths

| Symptom                                              | Look first                                                              |
| ---------------------------------------------------- | ----------------------------------------------------------------------- |
| `ForeignKeyViolationError` on `ir_edges` insert      | Ordering inside `IRGraphRepository.save` — caller ran a stage that bypassed `_sync_nodes` first |
| `DataError: invalid input syntax for type vector`    | Embedding dimension mismatch — `len(chunk.embedding) != 768`            |
| `psycopg2.DataError: A string literal cannot contain NUL` | Chunk text contains `\x00` — `PgvectorStore.upsert_chunks` strips it; check the chunker if it ever bubbles up |
| Vector search returns wrong chunks for a project     | `chunk.project_id` mismatch in the filter; check `VectorSearchFilter.project_id` |
| `IRNode` constraint violation in Neo4j               | `(project_id, id)` uniqueness — `sync_graph` should upsert, but check for in-flight concurrent syncs |
| HNSW index missing / slow queries                    | Was the migration run? Check `pg_indexes` for `ix_chunks_embedding_hnsw` |
| Recursive CTE hangs                                  | `_traverse` cycle guard missing — verify `NOT node_id = ANY(path)` is present |
| `connection refused` on first query                  | `SEMRAG_DATABASE_URL`; `pool_pre_ping=True` doesn't connect until the first checkout |
| `PoolTimeoutError`                                   | `db_pool_size` / `db_max_overflow` too small for concurrent load        |
| Graph queries slow on deep traversals                | Switch to Neo4j (experimental) **only** if `PostgresGraphQueries`'s recursive-CTE ceiling is reached in production, not in micro-benchmarks |
| `IndexingJobRow` rows missing `created_at`           | That's an ORM column, not a dataclass field — see `src/semrag/jobs/README.md` §8 |
| `Neo4jGraphStore` test is skipped                     | Expected — the module isn't verified against a live server             |

---

## 13. Common mistakes

1. **Calling `session.commit()` inside a repository.** Repositories
   share the caller's transaction; an inner commit would silently
   break the atomicity the caller is relying on (e.g. ingest writes
   IR + chunks atomically).
2. **Creating a second `Engine` per request.** Every constructor should
   receive the shared `sessionmaker`. A second engine means two
   connection pools, two `pool_pre_ping` cycles, and (worst case)
   two different `Settings` reading different env values mid-process.
3. **Inserting into `chunks` with `chunk_id` not derived
   deterministically.** `chunk_id` is the natural key; a fresh UUID
   per re-embed would create duplicates instead of upserting.
4. **Deleting `ir_nodes` directly without deleting `ir_edges` first.**
   Will FK-violate. Use `IRGraphRepository.save`, which orders the
   operations correctly.
5. **Building an `Engine` from `os.environ` directly in a module.**
   All env reads should go through `core/settings.py`'s `Settings`.
   `session.py:get_database_url` is the single permitted direct read,
   and only because `Settings` itself depends on the URL at
   construction time.
6. **Putting unindexed data in JSONB that should be queryable.**
   `attributes` is JSONB because nothing filters on it; if a new query
   ever needs to filter on a field inside `attributes`, that's a
   sign it should become a real column, not a JSONB path expression
   with a GIN index. GIN works but is the wrong shape for the query.
7. **Wiring Neo4j because "we might want it someday."** The
   implementation is real but unverified. Switching today is shipping
   an experimental backend to production — run the (skipped) test
   suite against a live server first.
8. **Sharing `Session` across threads.** SQLAlchemy `Session` is
   **not** thread-safe. Each thread needs its own session, ideally
   via `sessionmaker()` scoped to the request, not a module-global.
9. **Reading `Settings` inside a repository.** Repositories should
   receive their config (the `Session`) by argument. Reading
   `Settings` inside a repository binds it to global state and makes
   unit tests harder to write.
10. **Treating `FileRow` as history.** It isn't. A file deleted from
    the repo is deleted from `files`. Use `scan_manifests` for
    history-queries at the scan-summary level.
11. **Filtering on `attributes->>'something'` with a query planner
    surprise.** GIN on JSONB works, but the planner won't always
    pick it. Verify with `EXPLAIN ANALYZE` before shipping.
12. **Reusing `_session_scope` in the worker.** The worker opens its
    own session and commits once at the end of PERSISTING; using
    `session_scope` there would commit twice and break the atomic
    property the worker relies on.

---

## 14. Architectural boundaries

The storage package **does not**:

- **Own the pipeline.** Scan → parse → IR → chunk → embed → persist
  is composed in [`src/semrag/services/services.py`](../services/README.md),
  not here. The repositories accept inputs; they don't orchestrate.
- **Own HTTP concerns.** No router, no request/response models, no
  auth. API code lives in [`src/semrag/api/`](../api/README.md).
- **Own embedding generation.** Providers live in
  [`src/semrag/ingestion/embedding/`](../ingestion/README.md). The
  vector store receives `VectorChunk` objects whose `embedding` field
  is already populated.
- **Own scheduling.** Background job dispatch lives in
  [`src/semrag/jobs/`](../jobs/README.md) and `src/semrag/worker.py`.
- **Own schema migrations.** Tables are defined here; migrations are
  Alembic's. (See `alembic/versions/` for the migration history.)
- **Own observability.** `settings`, metrics, traces live in
  [`src/semrag/core/`](../core/README.md). Repositories emit logs;
  they do not register metrics.
- **Own retrieval logic.** Ranking, fusion, intent boosting,
  re-ranking, lexical search — all in
  [`src/semrag/retrieval/`](../retrieval/README.md). Storage answers
  "which chunks match this vector" and "which nodes depend on this
  node"; retrieval decides *what to do with those answers*.
- **Cache or rate-limit.** The query cache, rate limiter, and Redis
  client live in [`src/semrag/api/`](../api/README.md). The storage
  layer is not a cache, and not rate-limited — it's the durable store.

What it **does** own:

- Table definitions, index choices, and column types.
- The exact upsert-and-prune SQL for every mutable table.
- The Protocol contracts for swappable backends (vector + graph).
- Session/engine assembly, because every other package needs them and
  there should be exactly one place to read the database URL.
- One-line cycle-safe traversal over the IR edge table.

---

## 15. Related documentation

- [`docs/developer_guide.md` §6](../../../docs/developer_guide.md#6-storage-layer) —
  config knobs (`SEMRAG_DB_POOL_SIZE`, `SEMRAG_DATABASE_URL`, etc.),
  cache/rate-limit backends (which live in `api/`), the Redis
  integration, and the cache + DB transaction model.
- [`docs/developer_guide.md` §13.5](../../../docs/developer_guide.md#135-add-a-storage-backend) —
  the cookbook for adding a new vector or graph backend.
- [`docs/CODEBASE_GUIDE.md`](../../../docs/CODEBASE_GUIDE.md) — high-level map of
  where each storage plane sits in the source tree and which call
  sites depend on which backend.
- [`src/semrag/api/README.md`](../api/README.md) — DI wiring of
  the shared `session_factory` and how request-scoped sessions are
  produced.
- [`src/semrag/services/README.md`](../services/README.md) — how the
  ingest pipeline composes `IRGraphRepository` and `PgvectorStore` in
  one transaction; how the query path composes `PostgresGraphQueries`
  and `PgvectorStore` over a single read session.
- [`src/semrag/jobs/README.md`](../jobs/README.md) — how the
  background worker uses one session for the whole pipeline and what
  happens on rollback (`_report_failure` in a fresh session).
- [`src/semrag/retrieval/README.md`](../retrieval/README.md) —
  lexical / vector / graph signals, fusion, intent boosting; the
  consumers of `VectorStore` and `PostgresGraphQueries`.
- [`src/semrag/core/README.md`](../core/README.md) — `Settings`,
  metrics, logging; the env-var schema and observability surface.
