# `src/semrag/services` — Service Layer

> **The composition root for business operations.** This README explains
> what the service layer is, why it exists, and how `IngestionService` and
> `QueryService` orchestrate the lower-level packages. It is service-
> specific; the end-to-end stage walkthroughs live in
> [developer_guide.md §3 / §4](../../../docs/developer_guide.md) and the
> HTTP API's translation seam is in
> [src/semrag/api/README.md](../api/README.md). For "where do I change X?"
> use [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md).

Intended audience: **Developers** and **Maintainers** who touch the
orchestration layer — the two classes that compose every other package
into a usable query or index run.

---

## 1. Purpose of the service layer

The service layer exists to give every other entry point exactly one
business operation to call:

- The CLI's `semrag index` and `semrag ask` commands.
- The HTTP API routers (`POST /v1/projects/{id}/index`,
  `POST .../answer`, `POST .../answer/stream`).
- The evaluation harness (`evaluation/evaluator.py`).
- Tests that need to exercise the full pipeline without HTTP.

Without this layer, every caller would have to know the full pipeline —
which scanner to use, which parser registry to build, how to wire the
chunker, embedder, IR repository, vector store, retriever, prompt
builder, agent loop, citation validator. The service layer keeps that
knowledge in two classes so the lower-level packages can stay focused
on a single responsibility each.

**Why not just put orchestration in the API routers?** Because the CLI,
the evaluation harness, and tests all need the same composition.
Duplicating it across routers (and keeping them in lockstep with each
other and with the CLI) is exactly the kind of drift the service layer
prevents. The cross-package boundary rule is the consequence: **routes
call services, never primitives.** See [Boundaries](#11-architectural-boundaries).

---

## 2. Package structure

```
src/semrag/services/
├── __init__.py                  exports IngestionService, IndexResult,
│                                 QueryService, QueryAnswer
└── services.py                  IngestionService, QueryService,
                                  IndexResult, QueryAnswer,
                                  DetailedQueryAnswer, DebugSummary,
                                  and the chunk-hygiene helpers used
                                  by QueryService.ask().
```

Two files. The `services.py` module is intentionally a single module
rather than a sub-package: it holds exactly two service classes plus
the dataclasses and helpers those classes need. There is no
`__init__` of singletons; nothing is created at import time. Every
collaborator is a constructor argument.

Public exports (verified against `services/__init__.py`):

| Name                | Kind        | Purpose                                                       |
| ------------------- | ----------- | ------------------------------------------------------------- |
| `IngestionService`  | class       | Composes `index(project_id, root_path) -> IndexResult`.       |
| `IndexResult`       | dataclass   | Summary of an index run: counts + change-set deltas.          |
| `QueryService`      | class       | Composes `ask(project_id, question) -> QueryAnswer` and the non-agent `retrieve(...)`. |
| `QueryAnswer`       | dataclass   | Backwards-compatible subset of `DetailedQueryAnswer` (see §3). |

> `DetailedQueryAnswer` and `DebugSummary` are also defined here but are
> **not** re-exported from `__init__.py` — callers import them directly
> from `semrag.services.services` (see `api/routers/answer.py`).

---

## 3. Responsibilities of every service

### `IngestionService` — scan → parse → IR → chunk → embed → persist

`IngestionService.index(project_id: str, root_path: str) -> IndexResult`
is the only entry point. The constructor takes a SQLAlchemy `Session`,
an `EmbeddingProvider`, an `EmbeddingCache`, `Settings`, and an optional
`DebugArtifactWriter`. The parser registry is built inside the
constructor via `build_default_registry()` — there is no caller-side
configuration of parsers; the registry is fixed.

What `index()` does, in execution order (stage names match the
`stage(...)` context managers in `services.py`):

| Stage         | Calls                                                        | What it persists                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| `ingest.scan` | `Scanner(manifest_store=PostgresManifestStore(session)).scan(...)` | Scan manifest (file metadata + change-set). |
| `ingest.parse`| `parser_registry.parse_file(root / rel, rec)` per file        | Parse results held in-memory for the IR builder.  |
| `ingest.ir`   | `IRBuilder().build(project_id, manifest.files, parse_results)` | IR graph held in-memory.                          |
| `ingest.chunk`| `chunk_project(str(root), manifest.files, parse_results, ir_graph)` | `list[Chunk]` held in-memory.                |
| `ingest.embed`| `embed_chunks(chunks, embedding_provider, embedding_cache)`  | `list[VectorChunk]` (text + embedding) in-memory. |
| `ingest.persist` | `IRGraphRepository(session).save(ir_graph)` then `PgvectorStore(session).upsert_chunks(...)` then `session.commit()` | Nodes + edges + vector rows in Postgres. |

The session is committed **once**, after both the IR repository and the
vector store have written, so a failure between them rolls back the
whole run. The service also calls `session.flush()` between stages so
constraint errors surface at the stage boundary they belong to rather
than at the final commit.

`IndexResult` carries back enough information to be useful in CLI
output and HTTP responses: `project_id`, `scan_id`, `file_count`,
`chunk_count`, `added`, `modified`, `deleted`.

### `QueryService` — embed → retrieve → prompt → agent → answer

`QueryService.ask(project_id: str, question: str) -> DetailedQueryAnswer`
is the full RAG path. A second entry point, `retrieve(project_id,
question, top_k=None)`, returns just the fused chunk list without
prompt construction or an agent loop — used by API clients that want
chunks, not an answer. The constructor takes a `Session`, an
`EmbeddingProvider`, an `LLMProvider`, `Settings`, an optional
`project_root` (which enables the `read_file` agent tool), and an
optional `DebugArtifactWriter`.

`QueryService` owns three small private builders:

- `_build_retriever()` — assembles a `HybridRetriever` using
  `Settings.reranker` (`lexical_overlap` vs `None`) and the graph /
  intent-boost / noisy-source settings.
- `_build_tools(project_id)` — returns a `ToolRegistry` containing
  `SearchCodebaseTool` (which re-uses the retriever + a closure over
  the embedding provider), `ListFilesTool(session, project_id)`, and
  optionally `ReadFileTool(project_root)` when one was configured.
- `_project_file_index(project_id)` — cheap `SELECT` over `FileRow`
  returning a category-grouped tree, used as the `FILE INDEX` block
  for structural queries (see §6).

What `ask()` does, in execution order:

| Stage             | Calls                                                                            | Notes                                                  |
| ----------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `query.retrieve`  | `HybridRetriever.retrieve(...)` after `embedding_provider.embed_texts([question])` | Includes RRF + graph + intent-boost + rerank + truncation. |
| dedup             | `_deduplicate_chunks(results)`                                                    | Drops pure log noise + near-identical chunks within a path. |
| structural filter | `_filter_for_structural(results)` when `is_structural_query(question)` is true    | Prefers source chunks over runtime-artifact chunks.   |
| file-index        | `_project_file_index(project_id)` when structural                                 | Inlined before retrieved context for structural queries. |
| `query.prompt`    | `PromptBuilder(...).build(question, results, file_index=..., max_chunks=6/None)`  | Trims to `llm_context_window - llm_reserved_for_response`. |
| `query.agent`     | `AgentLoop(llm, tools, max_iterations=...).run([m.model_dump() for m in prompt.messages])` | The seeded conversation is what makes the answer grounded. |
| citation analysis | `validate_citations(result.final_answer, prompt.included_chunks)`                | Summary dict on the response payload.                  |

Two dataclasses are produced:

- **`QueryAnswer`** — `final_answer`, `iterations_used`,
  `hit_iteration_limit`, `messages`, optional `debug: DebugSummary`.
  Backwards-compatible surface used by the CLI and the `QueryAnswer`
  import in `__init__.py`.
- **`DetailedQueryAnswer`** — adds `considered_chunks_count`,
  `included_chunks`, `agent_messages`, `citation_analysis`. The HTTP
  `answer.py` router maps this into `AnswerResponse` with citations,
  sources, retrieval stats, tools used, and metadata.

### Why seed the agent with retrieved context instead of letting it call `search_codebase` itself

The agent loop has access to `SearchCodebaseTool`. We *also* pre-fetch
via the retriever and pass the assembled context in the first user
message. This is documented in the `ask()` docstring and is the
behavior verified by `test_ask_seeds_agent_with_retrieved_context` in
`tests/unit/services/test_services.py`. Without seeding, every question
costs at least one LLM call to discover the same context the retriever
already has, and small models often pick the wrong query against the
tool. The agent's tools remain useful for follow-up refinement
(`read_file`, etc.).

### Helpers (file-private)

Three helpers live in `services.py` because they are only used by
`QueryService.ask()`:

- `_deduplicate_chunks(results)` — drops pure log noise (single-line
  repetition past `_MAX_REPEATED_LINE_COUNT` ×
  `_MAX_REPEATED_LINE_FRACTION`) and, within a single file, collapses
  near-identical chunks to the highest-scored one via a hash-of-lines
  signature. Returns `(kept, dropped_count)` so the dropped count can
  surface in the debug summary.
- `_filter_for_structural(results)` — for structural queries only:
  strips chunks whose `chunk_type` is in `_NOISE_CHUNK_TYPES`
  (`log`, `coverage_report`, `lint_report`, `timing_report`,
  `waveform`) or whose path matches `_NOISE_FILE_EXTENSIONS` /
  `_NOISE_PATH_FRAGMENTS`. The filter is a preference, not a hard cut:
  if stripping would leave fewer than `_MIN_KEEP = 4` chunks, the
  unfiltered results are returned.
- `_summarize_parse(result)` — coarse JSON summary (`type`,
  `node_count`, `token_count`, `text_length`) for the debug-artifacts
  bundle. Full IR graph is written separately by `DebugArtifactWriter`.
- `_project_file_index(session, project_id)` — category-grouped tree
  rendered as plain text. One `SELECT` against `FileRow`. The format
  separates the indexing `category:` label from the actual relative
  path so a small model doesn't mistake the label for a filesystem
  directory.

---

## 4. Service lifecycle

Both services are constructed per call by the entry point that owns
the session. Nothing is cached at module scope.

| Entry point                          | Construction path                                                          | Teardown                              |
| ------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------- |
| CLI `cmd_index` / `cmd_ask`          | `cli.py:_build_ingestion_service` / `_build_query_service` → `finally: session.close()` | session closed in `finally`    |
| HTTP request                         | `api/dependencies.py:get_query_service(db)` → route handler → response    | session closed by `get_db_session` generator (commit on success, rollback on exception) |
| HTTP `/v1/projects/.../index`        | Enqueues a job; the worker constructs an `IngestionService` per job run    | session closed by the worker          |
| Evaluation harness                   | `evaluation/evaluator.py:PipelineEvaluator(query_service: QueryService)`   | service owned by the harness; harness manages lifetime |
| Tests                                | `IngestionService(...)` / `QueryService(...)` constructed directly         | depends on the test (no global teardown) |

The constructor takes every collaborator by argument. There is no
module-level rate limiter, query cache, or job manager held by the
service layer itself — those live in `api/` and are applied
*upstream* of the service (in the router or middleware), not inside
the service. This is what lets a test build an `IngestionService`
against a tempdir without touching the module-global rate limiter or
query cache.

`QueryService` is built **per request** by `get_query_service` because
the embedding/LLM providers are read from `Settings` at call time, and
Settings can change between requests (especially in tests). The
`IngestionService` follows the same shape — the worker constructs one
per job run.

---

## 5. Service orchestration

### Ingestion orchestration

```
IngestionService.index(project_id, root_path)
  └─ Scanner(PostgresManifestStore(session)).scan(project_id, root)
        └─ write manifest via change detection
  └─ parser_registry.parse_file(...)         × N files
  └─ IRBuilder().build(...)                   → nodes + edges
  └─ chunk_project(...)                       → list[Chunk]
  └─ embed_chunks(...)                        → list[VectorChunk]
  └─ IRGraphRepository(session).save(ir_graph)
  └─ PgvectorStore(session).upsert_chunks(vector_chunks)
  └─ session.commit()                         ← single transaction
```

The orchestration is a **linear pipeline** with **one DB transaction**.
Every flush happens between stages so failures surface at the stage
boundary they belong to; the final commit is the only durable
boundary. A failure anywhere between `scan` and `commit` rolls back
the entire index run; partial state is not persisted.

### Query orchestration

```
QueryService.ask(project_id, question)
  ├─ is_structural_query(question)              (intent detection)
  ├─ embedding_provider.embed_texts([question]) → query vector
  ├─ HybridRetriever.retrieve(...)              → fused + ranked chunks
  ├─ _deduplicate_chunks(chunks)                → dedup + dropped count
  ├─ [structural] _filter_for_structural(chunks)
  ├─ [structural] _project_file_index(project_id) → FILE INDEX block
  ├─ PromptBuilder.build(question, chunks, file_index, max_chunks)
  │       → prompt.messages, included_chunks, excluded_chunk_count,
  │         estimated_total_tokens
  ├─ AgentLoop(llm, tools, max_iterations).run(prompt.messages)
  │       → final_answer, messages, iterations_used,
  │         hit_iteration_limit
  └─ validate_citations(final_answer, included_chunks)
                                              → citation summary
```

Three observations:

1. **Two passes happen before the agent loop.** The retriever does the
   ranked selection; the prompt builder trims to the token budget.
   These are distinct concerns — never collapse them.
2. **Tools are bound at agent time, not at construction.** The
   `ToolRegistry` is rebuilt inside `ask()` because `SearchCodebaseTool`
   needs a fresh `HybridRetriever` (whose `reranker` /
   `enable_graph_expansion` / etc. are read from settings at call
   time) and `ReadFileTool` needs the configured `project_root`.
3. **Citation analysis runs after the agent returns**, against the
   prompt builder's `included_chunks` (the chunks the LLM actually
   saw), not the retriever's full result list.

---

## 6. Dependencies

What each service depends on, and why:

| Dependency              | Ingestion | Query | Why                                                              |
| ----------------------- | --------- | ----- | ---------------------------------------------------------------- |
| `Session`               | ✓         | ✓     | Postgres session — single transaction for ingestion; read-only for query (no writes happen inside `ask()`). |
| `EmbeddingProvider`     | ✓         | ✓     | Index: embed chunks. Query: embed the question.                  |
| `EmbeddingCache`        | ✓         | —     | Index only — embedding the same chunk twice across re-indexes hits the cache. The query path does not benefit from a cache. |
| `LLMProvider`           | —         | ✓     | Query only — the agent loop is the LLM caller.                   |
| `Settings`              | ✓         | ✓     | Read inside the constructor (`debug_artifacts_dir`, `log_level`, `retention`) and again inside `ask()` / `_build_retriever()`. |
| `DebugArtifactWriter`   | optional  | optional | When omitted, the service builds one from `Settings`.        |

The parser registry (`build_default_registry()`) is built inside the
`IngestionService` constructor — callers do not pass one in. This is
intentional: there is no supported use case for swapping parsers at
the service boundary.

---

## 7. Transaction / resource ownership

### Ingestion — single transaction

`IngestionService.index()` owns the session for the duration of the
index run. The service **commits once** at the end of the
`ingest.persist` stage and **flushes** between stages:

```text
ingest.scan     → flush
ingest.parse    → (no DB write)
ingest.ir       → (no DB write)
ingest.chunk    → (no DB write)
ingest.embed    → (no DB write)
ingest.persist  → flush (IR), flush (chunks), commit (single commit)
```

Implications:

- A failure between `flush` and `commit` rolls back the entire run.
- A scan-then-fail-during-parse leaves the previous index in place —
  the scanner wrote its manifest through `Scanner.scan` but no IR
  / vector rows were written yet. The next index run will detect
  `added` / `modified` / `deleted` against the previous good
  manifest.
- Callers **must not** open a second transaction on the same session
  concurrently (the SQLAlchemy session is not safe for concurrent
  use).

### Query — read-only with respect to the DB

`QueryService.ask()` does not write to the DB. It reads through the
session (the retriever's graph queries, `FileRow` for the file
index, pgvector's `search`) and writes only the debug-artifacts
bundle on disk (under `Settings.debug_artifacts_dir`) when debug
artifact logging is active.

This means:

- The HTTP request can share a session with rate-limiting / caching
  without worrying about a partial-write failure mode in the service.
- `session.commit()` is never called by `QueryService.ask()`.
  Commit/rollback is the `get_db_session` generator's job.

### Resource ownership at the entry points

| Entry point    | Who owns the session?                              | When does it close?                          |
| -------------- | -------------------------------------------------- | -------------------------------------------- |
| CLI            | `cli.py:_build_ingestion_service` / `_build_query_service` | `session.close()` in `finally`            |
| HTTP API       | `api/dependencies.py:get_db_session` generator     | After the route handler returns / raises     |
| Worker         | `jobs/worker.py:indexing_worker(...)`              | After the pipeline finishes / raises         |

The service itself **does not close the session** — the entry point
that built it does. This keeps the service free of I/O lifecycle
details and makes it safe to embed in contexts (tests, evaluation)
with their own session lifecycle.

---

## 8. Extension points

The service layer exposes a small, well-defined surface for
extension:

| Need                              | Where to change                                          | Notes                                                     |
| --------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| Add a new `EmbeddingProvider`     | `ingestion/embedding/`, register in `build_embedding_provider(settings)` | No service change needed. The factory is called by the entry point, which then constructs the service. |
| Add a new `LLMProvider`           | `reasoning/llm/`, register in `build_llm(settings)`      | No service change needed. `QueryService` calls `LLMProvider.complete(...)` and `LLMProvider.stream(...)`. |
| Add a new persistence backend     | `storage/postgres/` (and/or `storage/vector/`, `storage/graph/`) | The service's constructor still takes a `Session`; the session is bound to a SQLAlchemy engine, which is the only constraint. |
| Add a new index pipeline stage    | New module under `ingestion/`, called from `IngestionService.index()` between existing stages | Keep the linear shape; add a `flush()` if the stage writes. |
| Add a new chunk-hygiene filter    | `_filter_for_structural` / `_deduplicate_chunks` in `services.py` | Both are file-private helpers; add to the existing pair of passes, do not introduce a third. |
| Add a new debug-artifact stage    | `DebugArtifactWriter` + a `write_*` call inside the service | `DebugArtifactWriter` is the canonical seam. |
| Add a new tool to the agent       | `reasoning/agents/tools.py` + register it in `QueryService._build_tools()` | A new tool is a new constructor argument to `ToolRegistry`. |
| Add a new evaluation metric       | `evaluation/evaluator.py`                               | `QueryService.ask()` returns enough information already (consider `DetailedQueryAnswer` over `QueryAnswer`). |

> **What is *not* an extension point.** The internal stage order
> (`ingest.scan → ingest.parse → ingest.ir → ingest.chunk →
> ingest.embed → ingest.persist`) is part of the contract that the
> ingestion worker relies on for progress reporting. Reordering
> stages will desync `PipelineStage` weights in `jobs/pipeline.py`
> without raising any error. Treat the order as fixed.

> **Future services (eval, list-files, delete-project).** The
> module docstring notes that future services should follow the same
> shape: a class with explicit constructor dependencies, no module-
> level singletons, and a debug-artifacts bundle when
> `Settings.log_level` is DEBUG or TRACE.

---

## 9. Debugging

### Debug artifacts bundle

When `Settings.log_level` is `DEBUG` or `TRACE`,
`DebugArtifactWriter` writes a per-run bundle under
`Settings.debug_artifacts_dir` (default `.semrag/debug/<run_id>/`).
The writer is a no-op at INFO+.

Both services call `debug_writer.start_run(run_id)` at the top of
their main entry point and `debug_writer.run_completed()` in a
`finally` so the bundle is closed cleanly even on exceptions.

The bundle contents (verified against `services.py` and
`core/debug_artifacts.py`):

| Stage file (under `debug/<run_id>/`) | Service   | What is written                                            |
| ------------------------------------ | --------- | ---------------------------------------------------------- |
| `scan/manifest.json`                 | Ingestion | The full scan manifest (file metadata + change-set).       |
| `parse/<rel_path>.json`              | Ingestion | One per parsed file, via `_summarize_parse(...)`.          |
| `ir/graph.json`                      | Ingestion | The full IR graph (`model_dump(mode="json")`).             |
| `knowledge_graph/graph_summary.json` | Ingestion | `{"node_count", "edge_count"}`.                            |
| `chunks/chunk_metadata.json`         | Ingestion | `chunk_id`, `relative_path`, `chunk_type`, `line_start`, `line_end` per chunk. |
| `embeddings/embedding_stats.json`    | Ingestion | Model name + embedded-chunk counts.                        |
| `timings/pipeline.json`              | Ingestion | Marker; per-stage durations are emitted as `stage=... event=finish elapsed_ms` log records. |
| `retrieval/ranking_trace.json`       | Query     | The `RankingTrace` from the retriever, if available.       |
| `retrieval/top_results.json`         | Query     | Top-K results after dedup + structural filter.             |
| `prompt/final_prompt.txt`            | Query     | The full prompt string sent to the LLM.                    |
| `llm/request.json` + `llm/response.json` | Query | System + user + provider/model + chunk IDs; final answer + iteration count. |
| `query_trace.json`                   | Query     | The full `QueryTrace` (preprocessing, embedding, retrieval, context assembly, prompt construction, token estimation, LLM call, citation extraction, final answer). |

### Structured stages

Every numbered step in both services runs inside a `stage(...)`
context manager (`core/observability.py`). Each stage emits:

- A `stage=... event=start` log line at entry.
- A `stage=... event=finish elapsed_ms=...` line at exit (with
  accumulated `stats` as extra fields).
- A `stage=... event=error` line on exception.

These log lines are the primary trace for "where did this index run
spend its time?" — the timings file in the debug bundle is just a
marker pointing at them.

Stage names match the development guide (`query.retrieve`,
`query.prompt`, `query.agent`, `ingest.scan`, `ingest.parse`,
`ingest.ir`, `ingest.chunk`, `ingest.embed`, `ingest.persist`).

### Common failure modes

| Symptom                                                      | Where to look                                                                       |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `NotADirectoryError` from `IngestionService.index(...)`       | Bad root path. The service rejects non-existent paths and files (not dirs) before any I/O. |
| Index run hangs after the scan, no parse log lines            | A single parse is stuck. Check the parser registry — language detection is in `parsers/pipeline.py`. |
| `session.flush()` succeeds but `commit()` raises             | Constraint violation in the IR repository or pgvector upsert. Check the traceback; the entire transaction rolls back. |
| `QueryService.ask()` returns `final_answer=None` with `hit_iteration_limit=True` | The agent loop ran out of iterations. Inspect the agent messages in `DetailedQueryAnswer` and the prompt in `debug/<run_id>/prompt/final_prompt.txt`. |
| Citation summary shows fabricated citations                  | The LLM cited a path it was not given. Inspect `included_chunks` (the chunks the LLM actually saw) — these come from the prompt builder, not the retriever. |
| Structural query returns irrelevant chunks                   | The file-index section is missing (check `debug_summary.file_index_included`) or the structural filter is being applied to a non-structural query (`is_structural_query` returned False). |
| Debug artifacts bundle missing despite `log_level=DEBUG`     | The directory is empty because `Settings.debug_artifacts_dir` resolves somewhere unwritable. The writer is silent when it cannot write. |
| Query is slower than expected                                | Look at the `stage=... event=finish elapsed_ms` lines. `query.retrieve` dominates for large projects; `query.agent` dominates for small projects. |

For more debugging material, see
[developer_guide.md §14](../../../docs/developer_guide.md#14-debugging-workflow)
and the operations runbooks under
[docs/operations/](../../../docs/operations).

---

## 10. Common mistakes

These are mistakes the codebase has already made once or is set up
to make; each explains *why* it is wrong, not just *what* to do.

- **Putting pipeline logic in a router.** A router that builds a
  parser registry, calls `IRBuilder`, and writes chunks to pgvector is
  a router that needs to be kept in lockstep with every other
  pipeline caller. Add a method to `IngestionService` and call it from
  every entry point; do not duplicate the composition.
- **Catching broad exceptions inside a service method.** The service
  layer lets exceptions propagate. The CLI catches them at the
  command boundary (`cmd_index` / `cmd_ask`), the API catches them
  via the global exception handler, and tests assert on them
  (`NotADirectoryError` is verified by
  `test_ingestion_rejects_nonexistent_root`). A `try/except` inside
  the service hides failures from every one of those.
- **Holding a session beyond the request in `QueryService`.** The
  service writes to the session during ingestion only (single
  transaction) and never commits in the query path. Stashing the
  session on a singleton and reusing it across requests breaks
  SQLAlchemy's session semantics and means a rollback in one request
  silently affects the next.
- **Building a `QueryService` at module scope.** The constructor
  captures `Settings` for the *embedding provider's model name* and
  for `debug_artifacts_dir` / `log_level` / `retention`. Settings
  are mutable; constructing once means stale settings on the next
  request. Build it per request via `get_query_service(...)`.
- **Letting the agent loop be the only entry point for retrieval.**
  `QueryService.ask()` pre-fetches via `HybridRetriever` *and* gives
  the agent its own `SearchCodebaseTool`. Pre-fetching is what makes
  the first-turn answer grounded (verified by
  `test_ask_seeds_agent_with_retrieved_context`); do not collapse
  the two paths.
- **Adding a `Settings` field to a service constructor and reading
  it at import time.** Read inside the method (so per-request
  `get_settings()` is honoured), not inside `__init__`. The
  construction-time reading is what the API README warns against;
  the same applies here.
- **Inventing a new transaction boundary inside `IngestionService`.**
  The service commits once at the end of `ingest.persist`. A
  mid-pipeline commit would let a partial index become durable
  without a coherent manifest, and the re-index path's
  added/modified/deleted detection would drift.
- **Returning a domain object from a router without mapping.**
  `DetailedQueryAnswer` is not a wire model. `api/routers/answer.py:
  map_to_answer_response` does the mapping; new routers must do the
  same with a pydantic response model.
- **Treating the stage order as a suggestion.** The order
  (`scan → parse → ir → chunk → embed → persist`) is part of the
  contract the worker relies on for progress reporting
  (`PipelineStage` weights). Reorder only with the worker in mind.
- **Running `IngestionService.index()` from inside a request
  handler.** It is synchronous and will block the worker thread /
  API thread for the duration of the index. The HTTP API correctly
  enqueues a job (`POST .../index` → `JobManager.create_job`) and
  the worker calls `IngestionService.index()` out-of-band. The CLI
  bypasses jobs because the CLI is a one-shot command, not a long-
  lived service.

---

## 11. Architectural boundaries

The service layer enforces four boundaries that, together, keep the
package composable.

### 1. Routes call services, never primitives

> **From [src/semrag/README.md](../README.md):**
> *"Routes call services, never the lower-level primitives."*

A FastAPI router that imports `HybridRetriever`, `PgvectorStore`, or
`IRBuilder` directly has skipped the composition step. The router
should call `QueryService.ask(...)` or `IngestionService.index(...)`
and map the result.

### 2. Lower-level packages do not import from `services/`

`ingestion/`, `retrieval/`, `reasoning/`, and `storage/` do not
import from `services/`. The dependency arrow points one way. The
service layer is the only place that knows the full pipeline.

### 3. Evaluation and jobs are allowed to call services

- `evaluation/evaluator.py:PipelineEvaluator` takes a
  `QueryService` — eval is a query-time consumer.
- `jobs/worker.py:indexing_worker` constructs an `IngestionService`
  per job run — jobs *are* the API's way of running services
  asynchronously.

These are the two callers other than the CLI and the HTTP API. New
callers that need to run the full pipeline should follow the same
pattern.

### 4. No HTTP knowledge in the service layer

The service layer does not import from `fastapi`, does not raise
`HTTPException`, and does not know about `QueryResponse` /
`AnswerResponse`. Those are wire models in `api/schemas.py`; the
mapping happens in `api/routers/answer.py:map_to_answer_response`.
If a service method starts looking at request objects or status
codes, the boundary has slipped.

---

## 12. Related documentation

- [src/semrag README](../README.md) — package boundaries and the
  dependency-rule summary at the package level.
- [src/semrag/api/README.md](../api/README.md) — how routes translate
  HTTP requests into service calls; the dependency provider
  (`api/dependencies.py:get_query_service`) is the canonical service
  construction site for the API.
- [developer_guide.md](../../../docs/developer_guide.md) — end-to-end
  stage walkthroughs (§3 query, §4 indexing), DI philosophy (§5),
  debugging workflow (§14), and the architectural rules that
  constrain where new code can go.
- [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md) — file-by-file
  ownership and "where do I change X?" tables.
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — long-form architecture
  with data-flow diagrams.
- [docs/operations/README.md](../../../docs/operations/README.md) — on-
  call runbooks. The `indexing-failure.md`,
  `high-latency.md`, and `job-backlog.md` runbooks are the most
  relevant when the failure mode is inside the service layer.
- [tests/unit/services/test_services.py](../../../tests/unit/services/test_services.py)
  — the unit tests that pin the dataclass shapes, the
  `NotADirectoryError` contract, and the seeded-context guarantee
  on `QueryService.ask()`.