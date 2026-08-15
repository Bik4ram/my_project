# Repository Guidelines

## Goal

A production Semiconductor RAG platform.

Primary users:

- Design Verification Engineers
- RTL Engineers
- Verification Leads

---

## Where to start

- [README.md](./README.md) — quickstart, env vars, observability.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — module layout, data flow,
  provider abstractions.
- [CHANGELOG.md](./CHANGELOG.md) — release notes.

---

## Tech Stack

- Python 3.12
- PostgreSQL 16 + pgvector
- Ollama (default LLM and embedding provider)
- FastAPI
- pydantic-settings

---

## Repository Areas

src/semrag/

- core/         settings, observability, debug artifacts
- services/     IngestionService, QueryService (the only entry points)
- ingestion/    scanner, parsers, IR, chunking, embeddings
- retrieval/    hybrid retriever, RRF, rerank, graph expansion
- reasoning/    LLM abstraction, agent loop, tools, prompting
- storage/      postgres, pgvector, experimental neo4j
- api/          FastAPI routers, query cache, rate limit
- evaluation/   eval harness

---

## Expected Workflow

When fixing bugs:

1. Reproduce.
2. Find root cause.
3. Implement fix.
4. Re-run the failing test that triggered the bug.
5. Verify no regressions.

Never guess.

Never hide exceptions.

Never delete failing code to make tests pass.

When adding providers (LLM, embedding, vector store), follow the
existing protocol-based pattern: define the Protocol, ship the concrete
implementation, register it in the factory. Do not import concrete
classes from business code.

---

## Priorities

Correctness

↓

Reliability

↓

Performance

↓

Readability

↓

Convenience

