# Architecture

This document describes the SemRAG platform end-to-end: how code
is organised, how data flows through the pipelines, and what the
extensibility points are.

---

## Module layout

```
src/semrag/
├── core/                    Cross-cutting primitives
│   ├── settings.py          Pydantic-settings, single env-var source of truth
│   ├── observability.py     Logging + structured metrics + per-stage tracing
│   └── debug_artifacts.py   Optional debug-artifact writer with retention
├── services/                Business orchestration
│   ├── __init__.py
│   └── services.py          IngestionService, QueryService — compose primitives
├── ingestion/               Indexing pipeline (one-way)
│   ├── scanner/             Walk, hash, classify files; compute change sets
│   ├── parsers/             Language-aware parsing; produce parse results
│   ├── ir/                  Two-pass IR builder; deterministic node IDs
│   ├── chunking/            Chunk-by-language strategies
│   └── embedding/           Pluggable embedding providers + caching + batched pipeline
├── retrieval/               Query pipeline (query-side)
│   ├── pipeline.py          HybridRetriever: vector + lexical + RRF + rerank + source authority + diversity
│   ├── reranking.py         Rerankers
│   ├── graph_expansion.py   Graph expansion over the IR
│   ├── source_authority.py  Document-aware ranking prior
│   ├── ranking/             Post-fusion ranking stages (intent, source, graph, diversity)
│   └── fusion.py            Reciprocal Rank Fusion
├── reasoning/               LLM agent
│   ├── llm/                 LLMProvider abstraction + Ollama/OpenAI-compat impls
│   ├── agents/              Tool-calling ReAct loop + ToolRegistry
│   └── prompting/           System prompt + templates
├── storage/                 Persistence adapters
│   ├── postgres/            SQLAlchemy repos, migrations, manifest store
│   ├── vector/              Pgvector adapter
│   └── graph/               Experimental Neo4j adapter (disabled by default)
└── api/                     FastAPI surface
    ├── app.py               App factory, middleware, lifecycle
    ├── routers/             /projects, /query, /query/stream, /eval
    ├── query_cache.py       TTL cache for completed query responses
    └── rate_limit.py        Token-bucket per API key
```

---

## End-to-end indexing data flow

```
                            +-------------------+
                            |  run_local_qa.py  |
                            |    index ...      |
                            +---------┬---------+
                                      |
                                      v
                            +-------------------+
                            |  IngestionService |
                            |     .index()      |
                            +---------+---------+
                                      |
        +-----------+-----------+-----+----+---------+---------+
        |           |           |     |    |         |         |
        v           v           v     v    v         v         v
  scanner.scan  parse each  IRBuilder   |   embed_chunks  IRGraphRepo  PgvectorStore
   (manifest,    file      (nodes+edges)|  (provider +     .save()    .upsert_chunks()
    change_set)                        |   cache + RPS)
                                      |
                                      v
                              Postgres tables:
                                 files
                                 scan_manifests
                                 ir_nodes, ir_edges
                                 chunks + embeddings
                                 tsvectors (via DB triggers/indexes)
```

A complete stage-by-stage run is instrumented with `stage(...)` and
`timer(...)` lines on the project logger; per-run debug artifacts land
under `.semrag/debug/<run-id>/` when `SEMRAG_LOG_LEVEL` is DEBUG or
TRACE. See [README.md](./README.md#logging--observability) for the
artifact layout.

---

## End-to-end query data flow

```
                       +----------------+
                       | run_local_qa.py|
                       |   ask ...      |
                       +-------+--------+
                               |
                               v
                     +----------------+
                     | QueryService   |
                     |   .ask()       |
                     +-------+--------+
                             |
                             v
                     +----------------+   embed question
                     | build Tools    +----------------+
                     |  + AgentLoop   |  retrieval.retrieve(...)
                     +-------+--------+  (vector + lexical + RRF + graph
                                          + source authority + rerank + diversity)
                             |
                             v
                  +--------+--------+
                  |  AgentLoop.run  |
                  |                 |
                  |  for iter 1..N:  |
                  |   llm.complete  |
                  |     -> tool_call |
                  |     -> execute  |
                  |     -> tool_msg  |
                  |                 |
                  |  on no tool_call |
                  |   -> final_answer
                  +--------+--------+
                           |
                           v
                       QueryAnswer
                       (final_answer,
                        iterations_used,
                        hit_iteration_limit,
                        messages)
```

---

## Settings & configuration

`semrag.core.settings.Settings` is the single source of truth for every
runtime tunable. It is a `pydantic.BaseSettings` with `SEMRAG_` env-var
prefix; services and the API both receive it explicitly rather than
reading env vars in ad-hoc modules. Tests reset the cached singleton via
`reset_settings_cache()`.

---

## Service layer

`semrag.services.services` is the only module the CLI and the API both
call. It composes Scanner → ParserRegistry → IRBuilder → chunk_project →
embed_chunks → IRGraphRepository / PgvectorStore during ingestion, and
retriever + agent during query. Both services accept an explicit
`DebugArtifactWriter` so tests can pin the writer to a tempdir without
modifying the singletons.

---

## Provider abstractions

Three protocol-based abstractions keep the business logic free of
provider-specific code:

- `EmbeddingProvider` (`embedding/interfaces.py`) — `embed_texts(list[str])
  -> list[list[float]]`. Concrete impls: `OllamaEmbeddingProvider`,
  `OpenAICompatibleEmbeddingProvider`, `TfidfEmbeddingProvider`.
- `LLMProvider` (`reasoning/llm/interfaces.py`) — `complete()` and
  `stream()`. Concrete impls: `OllamaLLM`, `OpenAILLM`,
  `OpenAICompatibleLLM` (covers vLLM and any OpenAI-compat shim),
  `MockLLM`.
- `VectorStore` / `ManifestStore` (`storage/*/interfaces.py`) — protocol
  interfaces with Postgres-backed production implementations.

Adding a provider is a one-line addition in the matching `factory.py`;
no business code changes.

---

## Logging, metrics, debug artifacts

Three outputs flow through `semrag.core.observability`:

1. **Stdlib logger** — ordinary `info`/`debug`/`warning` lines shaped
   either human-friendly or JSON. `Settings.log_format` switches.
2. **`Metrics`** — emits one `metric=<name> value=<v> tags=...` record per
   counter / observation / timer on the `semrag.metrics` logger. A
   downstream log shipper can split metric events from ordinary lines
   without parsing free-form text.
3. **Per-stage `stage()`** — emits start/finish log lines tagged with
   `stage`, `event`, `status`, `elapsed_ms`, plus any stats the caller
   stuffed into the yielded `StageStats` dict (e.g. `chunk_count`).

Debug artifacts land in `Settings.debug_artifacts_dir` only at DEBUG or
TRACE; at INFO they are a no-op so production is unaffected.

---

## Testing strategy

- `tests/unit/llm/` — provider tests with `requests.post` mocked. No
  network.
- `tests/unit/agents/` — agent loop + tools, with `MockLLM` and
  in-memory fakes. No network.
- `tests/unit/vector/` & `tests/unit/storage/` — pgvector & graph tests
  that build an ephemeral Postgres via fixtures, but skip when no DB is
  reachable.
- `tests/unit/api/` — FastAPI router tests via the test client.
- `tests/unit/retrieval/` — RRF / reranking / fusion unit tests with
  in-memory fakes.

Integration tests requiring live Postgres/Ollama are skipped if the
endpoints aren't reachable so `pytest tests/unit` works in CI without
external services.
