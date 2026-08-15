# SemRAG

A production-grade Retrieval-Augmented Generation platform for
semiconductor design repositories. Indexes RTL, UVM, SystemVerilog, PDF,
Excel, and Markdown files; answers natural-language questions with cited
chunks pulled from a vector + lexical hybrid retriever.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](../../actions)
[![Coverage](https://img.shields.io/badge/coverage-see%20artifact-blue)](../../actions)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()

---

## Who it's for

- **Design Verification Engineers** — "where is this FIFO overflow
  handled?" across a thousand-file testbench tree.
- **RTL Engineers** — "find every module that resets a particular FSM."
- **Verification Leads** — "summarise the assertions covering a phase of
  the protocol."

---

## Architecture (one-paragraph version)

The platform walks the source tree, hashes every supported file, parses
text and HDL files into an intermediate representation, chunks by
language, embeds chunks with a pluggable embedding provider, persists
chunks in PostgreSQL with pgvector for vector retrieval and a tsvector
index for lexical retrieval, fuses both via Reciprocal Rank Fusion,
expands with a graph over the IR, reranks, and feeds the top-N into a
tool-calling ReAct agent that produces a final answer with citations.

For the long version see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Installation

```bash
# Clone and enter the repo
git clone <repo-url> semrag
cd semrag

# Create a virtualenv and install pinned dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Install the package in editable mode (registers the `semrag` CLI)
pip install -e .

# Copy the example config and edit
cp .env.example .env
```

> The CLI is installed as `semrag`. The legacy `python run_local_qa.py`
> form continues to work.

---

## Running locally

```bash
# 1. Make sure Postgres + pgvector and Ollama are up (see SETUP_POSTGRES.md)
export SEMRAG_DATABASE_URL='postgresql+psycopg2://semrag:semrag@localhost:5432/semrag'
export SEMRAG_LLM_PROVIDER=ollama
export SEMRAG_LLM_MODEL=qwen2.5:7b
export SEMRAG_EMBEDDING_PROVIDER=ollama
export SEMRAG_EMBEDDING_MODEL=nomic-embed-text

# 2. Index a project
semrag index --project-id rtl_uvm --root ~/projects/RTL_UVM

# 3. Ask a question
semrag ask --project-id rtl_uvm --question "how is the AXI burst transaction handled?"
```

For the HTTP API, see [RUN_API.md](./RUN_API.md).

### Using `make`

```bash
make test          # unit tests
make lint          # ruff
make format        # black --check
make typecheck     # mypy
make check         # lint + format + typecheck
make coverage      # tests with coverage report
make run-api       # uvicorn semrag.api.app:app
make docker-build  # production image
make docker-run    # run the image
make lock          # regenerate requirements*.txt
make clean         # remove caches / build artifacts
```

---

## Documentation

- [Documentation Index](./docs/README.md) — entry point for guides,
  reference, architecture, operations, and contributing material.
- [Architecture](./ARCHITECTURE.md) — long-form system architecture and
  data flow.
- [Deployment](./DEPLOYMENT.md) — local, Docker, Compose, Kubernetes, and
  Helm deployment recipes.

---

## Docker

```bash
docker build -t semrag:dev .
docker run --rm -p 8000:8000 --env-file .env semrag:dev
```

The image is multi-stage, runs as a non-root user, exposes `/livez` for
container probes, and pins the Python base image for reproducibility.

For Docker Compose, Kubernetes, and Helm deployments, see
[DEPLOYMENT.md](./DEPLOYMENT.md).

### Health & version endpoints

```bash
curl http://localhost:8000/livez    # liveness, no deps
curl http://localhost:8000/readyz   # readiness, every dependency
curl http://localhost:8000/version  # version, git commit, build time
```

---

## Running tests

```bash
pytest tests/unit -q                # all unit tests
pytest tests/unit/api -q            # API tests only
pytest --cov=semrag tests/unit      # with coverage
make coverage                       # same, via Makefile
```

Integration tests that need a live Postgres or LLM are skipped by
default; bring those services up locally to enable them.

---

## Environment variables

All settings are loaded from `SEMRAG_*` environment variables (or a
`.env` in the working directory). See [`.env.example`](./.env.example)
for every variable with a safe default. Full schema and validation
lives in `src/semrag/core/settings.py`.

| Variable                       | Default                            | Description                                |
|--------------------------------|------------------------------------|--------------------------------------------|
| `SEMRAG_DATABASE_URL`          | `postgresql+psycopg2://semrag:...` | DB connection string                       |
| `SEMRAG_LOG_LEVEL`             | `INFO`                             | `TRACE` \| `DEBUG` \| `INFO` \| ...        |
| `SEMRAG_LOG_FORMAT`            | `human`                            | `human` (color) or `json`                  |
| `SEMRAG_EMBEDDING_PROVIDER`    | `ollama`                           | `ollama` \| `openai_compatible` \| `tfidf` |
| `SEMRAG_EMBEDDING_MODEL`       | `nomic-embed-text`                 | Model name as the provider reports it      |
| `SEMRAG_EMBEDDING_DIMENSION`   | `768`                              | Must match the model's output              |
| `SEMRAG_LLM_PROVIDER`          | `ollama`                           | `ollama` \| `openai` \| `openai_compatible` \| `vllm` \| `mock` |
| `SEMRAG_LLM_MODEL`             | `qwen2.5:7b`                       | Model name                                 |
| `SEMRAG_VECTOR_TOP_K`          | `20`                               | Vector retrieval depth                     |
| `SEMRAG_LEXICAL_TOP_K`         | `20`                               | Lexical retrieval depth                    |
| `SEMRAG_FINAL_TOP_K`           | `10`                               | Chunks presented to the agent              |
| `SEMRAG_API_KEYS`              | `dev-local-key`                    | Comma-separated keys accepted by the API   |
| `SEMRAG_RATE_LIMIT_PER_MINUTE` | `60`                               | Per-API-key rate limit                     |

---

## Logging & observability

- **INFO**: concise pipeline summaries (`index complete`, `agent
  finished iteration 2/8`).
- **DEBUG**: per-stage start/finish events with file counts, node
  counts, embedding stats. Per-stage structured metrics (`metric=...`)
  on the `semrag.metrics` logger.
- **TRACE**: every LLM request, embedding call, retrieval hit.

Debug artifacts (only emitted at DEBUG/TRACE):

```
.semrag/debug/<run-id>/
    scan/manifest.json
    parse/<source-file>.parse.json
    ir/graph.json
    chunks/chunk_metadata.json
    embeddings/embedding_stats.json
    knowledge_graph/graph_summary.json
    retrieval/...
    prompt/final_prompt.txt
    llm/request.json
    llm/response.json
    timings/pipeline.json
```

Retention defaults to 5 most recent runs (configurable via
`SEMRAG_DEBUG_ARTIFACT_RETENTION`).

---

## Development

```bash
make test
make lint
make format
make typecheck
```

See [AGENTS.md](./AGENTS.md) for the contribution policy.

---

## License

Internal project. No public license declared.
