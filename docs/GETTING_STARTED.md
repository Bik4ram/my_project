# Getting Started with SemRAG

A hands-on, lab-manual walkthrough for someone who has **never used SemRAG
before**. Read the steps in order. Every step says what is happening, why it
is happening, what success looks like, and what to do if it fails.

> **Audience.** Design-verification engineers, RTL engineers, and verification
> leads who want to index a repository and ask questions of it. Assumes
> basic terminal knowledge (cd, ls, export, curl). Does not assume Python
> packaging, Docker, or Postgres experience.

> **Scope.** This document gets you from a fresh `git clone` to a working
> indexed project and a first answer. It does **not** cover production
> deployment, multi-replica clusters, or extending the platform with new
> parsers/providers — those are in [DEPLOYMENT.md](../DEPLOYMENT.md),
> [ENGINEERING_PLAYBOOK.md](./ENGINEERING_PLAYBOOK.md), and the package
> READMEs.

---

## 1. What is SemRAG?

SemRAG is a retrieval-augmented generation (RAG) platform purpose-built for
semiconductor design repositories. It indexes RTL, UVM, SystemVerilog, PDF,
Excel, and Markdown files and answers natural-language questions about them
with cited chunks.

The pipeline in one sentence:

> walk the source tree → parse to an IR → chunk by language → embed with a
> pluggable provider → store in PostgreSQL + pgvector (vector) and a
> tsvector index (lexical) → fuse with Reciprocal Rank Fusion → expand with a
> graph over the IR → rerank → feed the top-N into a tool-calling ReAct
> agent → final answer with citations.

Read [ARCHITECTURE.md](../ARCHITECTURE.md) for the long-form picture.

---

## 2. Who should use it?

- **Design verification engineers** asking "where is this FIFO overflow
  handled?" across a thousand-file testbench tree.
- **RTL engineers** looking for every module that resets a particular FSM.
- **Verification leads** wanting a summary of the assertions covering a
  phase of the protocol.

If your repository is mostly prose, this is overkill. If it is mostly
RTL/UVM/SystemVerilog, this is the right tool.

---

## 3. Minimum hardware requirements

- **CPU:** 4 cores (the indexing pipeline uses a bounded thread pool;
  `pool_size` is `5` by default).
- **RAM:** 8 GB total — 2 GB for Postgres, 8 GB for Ollama if you run it
  locally, 1 GB for the API. Workers are limited to 4 GB by default in
  `docker-compose.yml`.
- **Disk:** 20 GB free — Postgres data, embedding cache, Ollama model
  blobs (~5 GB for `nomic-embed-text` and ~5 GB for `qwen2.5:7b`).

---

## 4. Recommended hardware

- **CPU:** 8 cores.
- **RAM:** 16 GB.
- **Disk:** 50 GB SSD.
- **GPU:** optional; Ollama will use one if present and dramatically
  improves LLM throughput.

---

## 5. Supported operating systems

- **Linux** (Ubuntu 22.04 / 24.04, Debian 12) — primary target.
- **macOS** (Apple Silicon or Intel) — works; Docker Desktop is the
  simplest path.
- **Windows** via WSL2 (Ubuntu 24.04) — works; tested with the project's
  Windows WSL2 environment.
- **Native Windows** (without WSL) — not supported.

---

## 6. Prerequisites

You need the following before step 7:

| Tool | Version | Why |
|------|---------|-----|
| Python | **3.11 or 3.12** (`pyproject.toml` declares `requires-python = ">=3.11"`) | Runtime |
| PostgreSQL | **16** with the `pgvector` extension | Vector + lexical storage |
| Redis | **7+** (optional) | Cache + rate-limit backend |
| Ollama | **latest** (optional) | LLM + embedding provider |
| Docker + Docker Compose | **v2** (optional) | Containerised path |
| Git | any | Cloning the repo |

> **Why these specifically.** SemRAG pinnenly targets Python 3.12 in the
> Dockerfile (`PYTHON_VERSION=3.12`) and Python 3.11 in `pyproject.toml`.
> Postgres 16 is required because the production Docker image uses
> `pgvector/pgvector:pg16`. Ollama is the default LLM/embedding provider
> but is optional — you can point at any OpenAI-compatible server.

---

## 7. Clone the repository

```bash
git clone <repo-url> semrag
cd semrag
```

**What you are doing:** fetch the source tree and its `.git` history.

**Why:** every subsequent step assumes you are inside the project root.

**Expected result:** a `semrag/` directory with `pyproject.toml`, `src/`,
`tests/`, `deploy/`, `docs/`, and a `Dockerfile` at the top.

**Common failure:** `git: command not found` — install Git first.

---

## 8. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**What you are doing:** creating an isolated Python environment so the
project's dependencies don't conflict with system Python.

**Why:** the project pins ~25 runtime dependencies and a similar number of
dev dependencies. Without isolation, `pip install` will leak into your
system Python.

**Expected result:** the prompt is prefixed with `(.venv)`.

**Common failure:** `python: command not found` — install Python 3.11+.
**Fix:** on Ubuntu, `sudo apt-get install python3.12 python3.12-venv`.

---

## 9. Install dependencies

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Or use the Makefile target:

```bash
make dev-install
```

**What you are doing:** installing the project's pinned dependencies
(runtime + dev) and then installing SemRAG itself in *editable* mode.

**Why editable mode:** it registers the `semrag` console script so you can
run `semrag index` and `semrag ask` from any directory. The legacy
`python run_local_qa.py` form keeps working too.

**Why `requirements-dev.txt`:** it includes the runtime deps
(`requirements.txt` is also installed transitively) plus the test
toolchain. If you only need to run the API, `pip install -e .` is enough
on its own.

**Verify success:**

```bash
semrag --help
```

You should see argparse output with `index` and `ask` subcommands listed.

**Common failure:** `pip install` fails compiling `psycopg2-binary` or
`pgvector`. **Fix:** `pip install --upgrade pip` then re-run. The
`manylinux` wheels are bundled in the pinned versions.

---

## 10. Configure environment

Copy the example file and edit it:

```bash
cp .env.example .env
```

The default `.env.example` is **already correct for local development with
Ollama**. The bare minimum you need to know:

```bash
# — Database
SEMRAG_DATABASE_URL=postgresql+psycopg2://semrag:semrag@localhost:5432/semrag

# — Embedding provider (defaults to Ollama + nomic-embed-text)
SEMRAG_EMBEDDING_PROVIDER=ollama
SEMRAG_EMBEDDING_MODEL=nomic-embed-text
SEMRAG_EMBEDDING_BASE_URL=http://localhost:11434

# — LLM provider (defaults to Ollama + qwen2.5:7b)
SEMRAG_LLM_PROVIDER=ollama
SEMRAG_LLM_MODEL=qwen2.5:7b
SEMRAG_LLM_BASE_URL=http://localhost:11434

# — API keys (development only — production rejects the default)
SEMRAG_API_KEYS=dev-local-key
SEMRAG_ENVIRONMENT=development
```

**What you are doing:** setting the env vars that SemRAG's Pydantic
`Settings` model (`src/semrag/core/settings.py`) reads on every call.

**Why:** the defaults in `core/settings.py` are written for development
on `localhost`. They are *not* safe for production (the dev key is
rejected at startup when `SEMRAG_ENVIRONMENT=production`).

**Verify success:**

```bash
python -c "from semrag.core.settings import get_settings; print(get_settings().database_url)"
```

This should print the connection string you set.

**Common failure:** `Settings` raises a `ValueError` saying the dev key
is not allowed in production. **Fix:** set `SEMRAG_ENVIRONMENT=development`
*or* set a real `SEMRAG_API_KEYS` value.

---

## 11. PostgreSQL setup

You need Postgres 16 with the `pgvector` extension. Pick **one** path.

### Path A — Docker (recommended)

```bash
docker run -d --name semrag-postgres \
  -e POSTGRES_DB=semrag \
  -e POSTGRES_USER=semrag \
  -e POSTGRES_PASSWORD=semrag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

This is the same image the production `docker-compose.yml` uses.

### Path B — Native install (Ubuntu)

```bash
sudo apt-get install -y postgresql-16 postgresql-16-pgvector
sudo service postgresql start
sudo -u postgres psql -c "CREATE USER semrag WITH PASSWORD 'semrag' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE semrag OWNER semrag;"
```

**Verify success:**

```bash
psql "postgresql://semrag:semrag@localhost:5432/semrag" -c "SELECT 1"
```

This should print `1`.

**Common failure:** `connection refused`. **Fix:** `sudo service
postgresql start` (Ubuntu) or `docker ps | grep semrag-postgres` (Docker).

---

## 12. pgvector setup

The `pgvector/pgvector:pg16` image already has the extension installed —
you just need to enable it in the database:

```bash
psql "postgresql://semrag:semrag@localhost:5432/semrag" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**What you are doing:** enabling the `vector` type and index support.

**Why:** SemRAG's chunk table (`chunks`) stores the embedding as a
`Vector(768)` column. Without pgvector, every insert into `chunks` raises
a type error.

**Verify success:**

```bash
psql "postgresql://semrag:semrag@localhost:5432/semrag" \
  -c "SELECT extname FROM pg_extension WHERE extname='vector';"
```

This should print `vector`.

**Common failure:** `extension "vector" is not available`. **Fix:** you
are not on the `pgvector/pgvector` image. Re-create the container with
the image shown in Path A above.

---

## 13. Redis (optional)

Skip this step if you are happy with the in-memory cache and rate-limit
backends. The defaults work fine for single-user development.

To enable Redis-backed caching and rate limiting:

```bash
docker run -d --name semrag-redis -p 6379:6379 redis:7-alpine
```

Then in `.env`:

```bash
SEMRAG_REDIS_URL=redis://localhost:6379/0
SEMRAG_QUERY_CACHE_BACKEND=redis
SEMRAG_RATE_LIMIT_BACKEND=redis
```

**Verify success:**

```bash
docker exec semrag-redis redis-cli ping
```

This should print `PONG`.

---

## 14. Ollama (optional)

Skip this if you have an OpenAI-compatible embedding/LLM server already
(e.g. vLLM, llama.cpp's server). Otherwise, install Ollama:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# macOS: download from https://ollama.com/download
# Docker: the production docker-compose.yml has an `ollama` profile
```

Pull the two models SemRAG defaults to:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

**Verify success:**

```bash
curl http://localhost:11434/api/tags
```

This should return JSON listing both models.

**Common failure:** `connection refused` on `localhost:11434`. **Fix:**
`ollama serve` must be running. On Linux it is typically a systemd unit
(`sudo systemctl start ollama`).

---

## 15. Other supported LLM providers

SemRAG supports three providers for each of LLM and embeddings:

| Provider | `SEMRAG_*_PROVIDER` | Notes |
|----------|---------------------|-------|
| **Ollama** | `ollama` | Default. Local model server. |
| **OpenAI-compatible** | `openai_compatible` | Point at any server implementing `/v1/chat/completions` and `/v1/embeddings` (Ollama, vLLM, llama.cpp). |
| **TF-IDF** (embeddings only) | `tfidf` | No external server. Fits on the indexed corpus. Useful for CI and air-gapped environments. |

For the OpenAI-compatible provider, set the base URL and an API key
(leave blank for servers that don't require one):

```bash
SEMRAG_LLM_PROVIDER=openai_compatible
SEMRAG_LLM_BASE_URL=http://my-llm-host:8000
SEMRAG_LLM_API_KEY=sk-...
```

For TF-IDF embeddings:

```bash
SEMRAG_EMBEDDING_PROVIDER=tfidf
```

The TF-IDF provider is fit on the corpus during indexing and persisted to
`SEMRAG_EMBEDDING_PROVIDER_DIR` (default `/tmp/semrag_embedding_providers`).

---

## 16. Run database migrations

```bash
alembic upgrade head
```

**What you are doing:** applying the Alembic migration chain
(`alembic/versions/`) which creates the `manifests`, `chunks`,
`indexing_jobs`, `api_keys`, and full-text-search tables.

**Why:** the schema is owned by Alembic, not by the application. The
`alembic_version` table is the source of truth for what migrations have
been applied.

**Verify success:**

```bash
alembic current
```

This should print the latest revision hash. The six revisions in order
are:

1. `692caefcb795` — initial schema
2. `f5e3cd5b5b42` — chunks table for pgvector
3. `0190b83318c7` — full-text search to chunks
4. `e28abd752a06b245` — api keys table
5. `d5b1cc8385d4f2b7` — indexing jobs table
6. `a1c06e93d2f4` — chunk metadata columns (RI-06)

`alembic current` should print `a1c06e93d2f4` (head).

**Common failure:** `database "semrag" does not exist`. **Fix:** create
the database (step 11), then re-run.

---

## 17. Verify installation

Run the full pre-flight in one command:

```bash
python - <<'PY'
from semrag.core.settings import get_settings
from semrag.core.observability import configure_logging
configure_logging(get_settings())
print("settings loaded OK")
print("embedding:", get_settings().embedding_provider, get_settings().embedding_model)
print("llm:", get_settings().llm_provider, get_settings().llm_model)
print("environment:", get_settings().environment)
PY
```

If you see the three pairs printed, the configuration is wired up. If
the call raises, the error message will tell you exactly which env var
is missing or wrong.

---

## 18. Start the API

```bash
uvicorn semrag.api.app:app --reload --port 8000
```

Or via the Makefile:

```bash
make run-api
```

**What you are doing:** launching the FastAPI server on port 8000 with
auto-reload.

**Why:** the API server is the surface that CLI, web UI, and CI all
talk to. The CLI (`semrag index` / `semrag ask`) is a thin wrapper that
talks to the same pipeline.

**Verify success:**

```bash
curl -s http://localhost:8000/livez | python -m json.tool
```

Expected:

```json
{ "status": "alive" }
```

**Common failure:** `ModuleNotFoundError: semrag`. **Fix:** activate
the virtualenv (`source .venv/bin/activate`) and ensure `pip install -e .`
ran successfully.

---

## 19. Start background worker (if applicable)

The API runs an in-process worker by default (`SEMRAG_API_RUNS_WORKER=true`).
For multi-replica deployments, or for isolating the indexing pipeline from
request handling, run the worker as a separate process:

```bash
SEMRAG_API_RUNS_WORKER=false \
SEMRAG_JOB_STORE_BACKEND=database \
python -m semrag.worker
```

**What you are doing:** polling the persistent job store (Postgres) and
running the indexing pipeline as jobs are enqueued.

**Why standalone:** the API process is request-bound; running indexing
in the same process means a slow scan stalls user-facing queries.
Splitting them is the production pattern.

**Verify success:** the worker logs `worker loop started` at INFO
level when the poll loop begins.

**Common failure:** worker can't connect to Postgres. **Fix:** ensure
`SEMRAG_DATABASE_URL` is set in the worker's environment.

---

## 20. Health check

Three endpoints, three contracts:

```bash
# Liveness — process is alive
curl -s http://localhost:8000/livez | python -m json.tool

# Readiness — every dependency is reachable
curl -s http://localhost:8000/readyz | python -m json.tool

# Build metadata — version, git commit, python version
curl -s http://localhost:8000/version | python -m json.tool
```

**Verify success:** `/readyz` returns HTTP 200 and `"status": "ready"`
with every check (`database`, `llm`, `embedding`, `job_executor`,
`cache`, `rate_limiter`) reporting `"ok"`.

**Common failure:** `/readyz` returns 503 with `database: "..."`. **Fix:**
see step 32 (Common database failures).

---

## 21. First project indexing using the CLI

The fastest way to index a project from your local machine:

```bash
semrag index --project-id myproj --root /absolute/path/to/project
```

The legacy form keeps working:

```bash
python run_local_qa.py index --project-id myproj --root /absolute/path/to/project
```

**What you are doing:** walking the source tree, parsing every supported
file, chunking by language, embedding each chunk, and persisting the
manifest, chunks, and IR graph to Postgres.

**Why a project ID:** it scopes the index. You can index multiple
repositories under different IDs and query them independently.

**Why an absolute path:** the project root is resolved with
`Path.resolve()`, so symlinks and relative paths are normalised but the
path must exist on the machine running the command.

**Verify success:** the CLI prints

```
Indexed N chunks across M files (A added, B modified, D deleted).
```

(For a first-time index, `A` is the total file count and `B = D = 0`.)

**Verify success in the database:**

```bash
psql "postgresql://semrag:semrag@localhost:5432/semrag" \
  -c "SELECT project_id, count(*) FROM chunks WHERE project_id='myproj' GROUP BY project_id;"
```

This should print one row with the chunk count.

**Common failure:** `Project root does not exist`. **Fix:** `ls -la
/path/to/project` to confirm the path is reachable from this machine.

---

## 22. Monitor indexing

While the CLI is running, the indexing progress is streamed to stdout at
DEBUG level. To see it:

```bash
SEMRAG_LOG_LEVEL=DEBUG semrag index --project-id myproj --root /path/to/project
```

The 22-stage pipeline (`SCANNING`, `PARSING`, `CHUNKING`, `EMBEDDING`,
`PERSISTING`, etc.) is logged as it transitions.

Indexes via the HTTP API return a `job_id` you can poll:

```bash
curl -s -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/projects/myproj/jobs/<job_id> | python -m json.tool
```

Polling returns `state` (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`,
`CANCELLED`), `progress` (0–100), `current_stage`, and `statistics`.

**Common failure:** progress stuck at 0%. **Fix:** `tail -f` the
log output; the worker may be waiting on Postgres or the embedding
provider.

---

## 23. First query

```bash
semrag ask --project-id myproj --question "how is the AXI burst transaction handled?"
```

Add `--debug` for a single-screen summary of retrieval (top paths,
intent, included/excluded chunks, iteration count):

```bash
semrag ask --project-id myproj --question "..." --debug
```

**What you are doing:** asking the ReAct agent to answer a question
using the indexed project. The agent has three tools: `search_codebase`,
`read_file`, `list_files`.

**Why `--debug`:** it shows you what was retrieved, what was excluded,
and whether the agent hit its `max_iterations` hard cap. Useful for
sanity-checking retrieval before trusting the answer.

**Verify success:** you see a non-empty answer with citations of the
form `[Source: <path>, lines X-Y]`.

**Common failure:** `(Agent hit its iteration limit without a final answer.)`
**Fix:** the agent has `max_iterations: 8` by default. The query may be
too narrow to converge; rephrase or raise `SEMRAG_LLM_MAX_ITERATIONS`.

---

## 24. Streaming query example

The HTTP API exposes a Server-Sent Events stream for the query endpoint:

```bash
curl -N -X POST http://localhost:8000/v1/projects/myproj/query/stream \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "how is reset handled?"}'
```

You will see events of the form `event: stage` followed by a JSON
`data:` line. The full event sequence is:

```
event: stage   {"stage": "embedding_query"}
event: stage   {"stage": "vector_and_lexical_search"}
event: stage   {"stage": "graph_expansion"}   (only when enabled)
event: stage   {"stage": "reranking"}         (only when enabled)
event: stage   {"stage": "assembling_prompt"}
event: result
```

The `result` event payload is the same as the non-streaming endpoint's
response.

**Common failure:** the connection closes mid-stream. **Fix:** the
client must support SSE; `curl -N` works, but `curl` without `-N` does
not.

---

## 25. API examples

### Indexing via HTTP

```bash
curl -X POST http://localhost:8000/v1/projects/myproj/index \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"root_path": "/absolute/path/to/project"}'
```

Returns `HTTP 202 Accepted` with a body containing the project ID and
placeholder counters. The actual scan completes asynchronously.

### Querying via HTTP

```bash
curl -X POST http://localhost:8000/v1/projects/myproj/query \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "how is reset handled?", "top_k": 10}'
```

Returns a `QueryResponse` with `messages`, `included_chunks`,
`excluded_chunk_count`, `estimated_total_tokens`, and `from_cache`.

### Answering via HTTP (uses the LLM)

```bash
curl -X POST http://localhost:8000/v1/projects/myproj/answer \
  -H "Authorization: Bearer dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "summarise the assertions covering the AXI handshake"}'
```

Returns an `AnswerResponse` with `answer`, `citations`, `sources`,
`iterations`, `tools_used`, `retrieval`, and `timings`.

### Listing files in an indexed project

```bash
curl -H "Authorization: Bearer dev-local-key" \
  http://localhost:8000/v1/projects/myproj/files | python -m json.tool
```

### Python (httpx)

```python
import httpx

headers = {"Authorization": "Bearer dev-local-key"}
base = "http://localhost:8000"

# Index asynchronously
resp = httpx.post(f"{base}/v1/projects/myproj/index",
                  headers=headers, json={"root_path": "/abs/path/to/project"},
                  timeout=30)
print(resp.json())

# Ask
resp = httpx.post(f"{base}/v1/projects/myproj/answer",
                  headers=headers, json={"query": "how is reset handled?"},
                  timeout=120)
print(resp.json()["answer"])
```

(Synchronous `requests` works too — `httpx` is just used here for the
streaming example.)

### Why two endpoints: `/query` vs `/answer`

`/query` returns retrieval output (the prompt that *would* be sent to the
LLM, the chunks, message history). `/answer` returns the final answer
after the LLM has run. Use `/query` when you want to debug retrieval;
use `/answer` when you want the user-facing answer.

---

## 26. Understanding the response

The `AnswerResponse` has six top-level fields:

| Field | Meaning |
|-------|---------|
| `answer` | The final natural-language answer, with `[Source: ...]` citations. |
| `citations` | `{valid_count, fabricated_count, fabrication_rate, cited_paths}` — whether each citation actually maps to a retrieved chunk. |
| `sources` | The chunks the agent used, with `relative_path`, `line_start`, `line_end`, and `text`. |
| `iterations` | How many ReAct iterations the agent ran. `hit_iteration_limit: true` is a soft failure signal. |
| `tools_used` | List of agent tools invoked (`search_codebase`, `read_file`, `list_files`). |
| `retrieval` | Per-stage retrieval stats (vector score, lexical score, fused score, etc.). |
| `timings` | Per-stage wall-clock durations. |

High `fabrication_rate` (>0.2) suggests the agent is hallucinating
citations. **Fix:** tighten `query_cache_ttl_seconds` is not the right
answer — instead, re-index (the IR may be stale) or rephrase the
question.

---

## 27. Viewing logs

Logs go to stdout in JSON or human format depending on
`SEMRAG_LOG_FORMAT` (`json` or `human`):

```bash
SEMRAG_LOG_FORMAT=human semrag index --project-id myproj --root /abs/path
```

For ongoing operations, the access log and the structured event log are
the canonical records. `tail -f` the process or use `docker compose
logs -f api`.

**Verify success:** every log line is a single line of JSON or human
format. Multi-line stack traces only appear on errors.

---

## 28. Viewing metrics

Prometheus-format metrics are exposed at `/metrics`:

```bash
curl -s http://localhost:8000/metrics | head -20
```

The eight Grafana dashboards (api, retrieval, llm, indexing, jobs,
database, cache, rate-limit) are loaded from
`deploy/monitoring/grafana/dashboards/`. See [DEPLOYMENT.md](../DEPLOYMENT.md)
for the Prometheus + Grafana setup.

**Verify success:** the response is `text/plain; version=0.0.4` and
includes `semrag_uptime_seconds` and at least one `semrag_*` counter.

**Common failure:** `404 Not Found`. **Fix:** enable metrics with
`SEMRAG_METRICS_ENABLED=true` (the default is `true`).

---

## 29. Viewing traces

OpenTelemetry traces are *off* by default. Enable them:

```bash
SEMRAG_TRACING_ENABLED=true \
SEMRAG_OTLP_ENDPOINT=http://localhost:4317 \
uvicorn semrag.api.app:app --port 8000
```

Point at any OTLP-compatible collector (Jaeger, Tempo, the OpenTelemetry
Collector). The exporter is configured in `core/tracing.py`.

**Verify success:** the trace appears in the collector within a few
seconds of a request hitting `/v1/projects/*/answer`.

---

## 30. Common configuration mistakes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Default development API key 'dev-local-key' is not allowed in production` | `SEMRAG_ENVIRONMENT=production` with default key | Set `SEMRAG_API_KEYS` to a real key, or set `SEMRAG_ENVIRONMENT=development`. |
| `embedding dimension mismatch: configured 768, server returned N` | `SEMRAG_EMBEDDING_DIMENSION` does not match the model | Set `SEMRAG_EMBEDDING_DIMENSION` to match the model's output dimension. |
| `connection refused` on `localhost:11434` | Ollama not running | `sudo systemctl start ollama` or `docker compose --profile ollama up -d`. |
| `pgvector extension not installed` | Wrong Postgres image | Use `pgvector/pgvector:pg16` (Path A in step 11). |
| `Job store backend "memory" is not safe in production` | Multi-replica deployment with `job_store_backend=memory` | Set `SEMRAG_JOB_STORE_BACKEND=database`. |
| `Rate limit backend "memory" is not safe in production` | Same problem for rate limiting | Set `SEMRAG_RATE_LIMIT_BACKEND=redis` (and configure Redis). |

---

## 31. Common startup failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: semrag` | Virtualenv not activated or `pip install -e .` not run | `source .venv/bin/activate && pip install -e .` |
| `Settings validation error: ...` | Missing env var | Read the message — it names the var. Set it in `.env`. |
| `Address already in use` on port 8000 | Another process on 8000 | `lsof -i :8000` to find it, or use `--port 8001`. |
| `database "semrag" does not exist` | Forgot to create the DB | Step 11. |
| uvicorn reload loops forever | Filesystem watcher on `.semrag/` debug artifacts dir | Set `SEMRAG_DEBUG_ARTIFACTS_DIR` outside the project root. |

---

## 32. Common database failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/readyz` reports `database: "FATAL: ..."` | Postgres down or credentials wrong | `psql "$SEMRAG_DATABASE_URL" -c "SELECT 1"`. |
| `extension "vector" is not available` | Postgres image does not have pgvector | Switch to `pgvector/pgvector:pg16`. |
| `value too long for type character varying(255)` | Postgres schema mismatch | `alembic upgrade head` — the schema may have drifted. |
| `remaining connection slots are reserved` | `pool_size + max_overflow` exceeded | Raise `SEMRAG_DB_MAX_OVERFLOW` or lower concurrent request count. |
| `permission denied for table api_keys` | The app user lacks DDL | `GRANT ALL ON SCHEMA public TO semrag;` |

---

## 33. Common embedding failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Failed to reach embedding endpoint http://localhost:11434 after 3 attempts` | Ollama down or stuck | `ollama pull nomic-embed-text` (re-pulls), then `curl http://localhost:11434/api/tags`. |
| `Embedding dimension mismatch: configured 768, server returned 1024` | Wrong model or wrong `SEMRAG_EMBEDDING_DIMENSION` | Pick one — the model is the source of truth. For `nomic-embed-text` it's 768. |
| `Max retries exceeded` on every chunk | Embedding endpoint overloaded | Lower `SEMRAG_MAX_CONCURRENT_PARSE_WORKERS` (default 4) or scale Ollama vertically. |
| Embedding cache grows without bound | `SEMRAG_EMBEDDING_CACHE_BACKEND=jsonfile` and no cleanup | Set `SEMRAG_EMBEDDING_CACHE_DIR` to a managed location and add a cron job, or switch to `memory`. |

---

## 34. Common LLM failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `(Agent hit its iteration limit without a final answer.)` | Query too narrow, agent looped | Raise `SEMRAG_LLM_MAX_ITERATIONS` (default 8) or rephrase. |
| `Failed to reach LLM endpoint` | Ollama down or wrong model name | `ollama pull qwen2.5:7b` (or whatever `SEMRAG_LLM_MODEL` is). |
| `llm_probe_at_startup: probe failed` | LLM endpoint not reachable at startup | Either fix the server or set `SEMRAG_LLM_PROBE_AT_STARTUP=false` (not recommended for prod). |
| `context window exceeded` | Query + retrieved chunks > `SEMRAG_LLM_CONTEXT_WINDOW` | Lower `SEMRAG_FINAL_TOP_K` or raise `SEMRAG_LLM_CONTEXT_WINDOW`. |
| `num_predict` truncates the answer | `SEMRAG_LLM_NUM_PREDICT` too low | Raise it (default 800). |

---

## 35. Docker workflow

The simplest path is the published image. Build it locally:

```bash
docker build -t semrag:dev .
```

Run it:

```bash
docker run --rm -p 8000:8000 --env-file .env semrag:dev
```

The image is multi-stage, runs as a non-root user (`uid=1001`), exposes
the API on port 8000, and uses `tini` for clean signal handling.

**Common failure:** container exits immediately. **Fix:** `docker run
--rm -p 8000:8000 --env-file .env --entrypoint /bin/bash semrag:dev`, then
run `uvicorn semrag.api.app:app --port 8000` inside to see the error.

---

## 36. Docker Compose workflow

The production `docker-compose.yml` brings up Postgres, the API, the
indexing worker, and optionally Redis and Ollama. The worker service
is always-on (not profile-gated): it polls the persistent job store
and runs the indexing pipeline.

```bash
docker compose up -d                  # Postgres + API + worker
docker compose --profile redis up -d  # + Redis
docker compose --profile ollama up -d # + Ollama
```

**What you are doing:** standing up the full stack with one command.

**Why profiles:** Redis and Ollama are opt-in. The dev default is
"Postgres + API + worker, point at external services".

**Verify success:**

```bash
curl -s http://localhost:8000/livez
```

Common failure: `port 8000 already in use`. **Fix:** set
`SEMRAG_API_PORT=8080` in `.env` before `docker compose up`.

---

## 37. Kubernetes workflow (high level)

The deployment manifest lives at `deploy/k8s/`. Eight numbered YAML
files:

```
00-namespace.yaml
01-configmap.yaml
02-secret.yaml        # placeholder — never commit real secrets
03-pvc.yaml
04-deployment.yaml    # API
05-service.yaml
06-ingress.yaml
07-worker-deployment.yaml
```

For a Helm chart, see `deploy/helm/semrag/`.

Read [DEPLOYMENT.md](../DEPLOYMENT.md) for the full procedure. For
on-call runbooks, see [docs/operations/](./operations/).

---

## 38. Performance tips

- **Embed in batches.** Don't index one file at a time; the default
  `pipeline.py` already batches.
- **Use TF-IDF for CI.** `SEMRAG_EMBEDDING_PROVIDER=tfidf` removes the
  external embedding roundtrip and is sufficient for benchmark queries.
- **Tune `top_k`.** `SEMRAG_FINAL_TOP_K=10` is the default. Lower it for
  faster queries at the cost of recall.
- **Cache hit rate.** `from_cache: true` in the response means the
  query was served from the in-memory or Redis cache. A low hit rate
  on a stable codebase means queries are too varied.
- **Worker concurrency.** `SEMRAG_MAX_CONCURRENT_PARSE_WORKERS=4` is
  the default. Raise it on machines with more cores.
- **DB pool.** `SEMRAG_DB_POOL_SIZE=5` and `SEMRAG_DB_MAX_OVERFLOW=10`
  is the default. For higher concurrency, raise both.

---

## 39. Frequently asked questions

**Q: What's the difference between `semrag ask` and the `/answer` HTTP
endpoint?**
A: `semrag ask` runs the full pipeline locally (CLI process opens its
own DB connection). The HTTP endpoint runs the same pipeline in the API
process. Results are identical.

**Q: Can I index multiple projects?**
A: Yes. Use a different `--project-id` per project. Queries are scoped
by `project_id`.

**Q: Can I re-index the same project?**
A: Yes. `semrag index` is idempotent: unchanged files are skipped,
modified files are re-chunked, deleted files are pruned. The IR graph
is updated synchronously inside the indexing transaction.

**Q: Does the LLM see my code?**
A: Yes — the agent receives the retrieved chunks as context. If you
need air-gapped operation, use `SEMRAG_EMBEDDING_PROVIDER=tfidf` and
point `SEMRAG_LLM_BASE_URL` at a local server. The provider factory is
in `src/semrag/reasoning/llm/factory.py`.

**Q: Where do I see the queries the system has handled?**
A: Each query produces an OpenTelemetry trace (when enabled) and a
row in the metrics counters. The Prometheus dashboards (`/metrics`) show
aggregate throughput, latency, and per-stage timings.

**Q: How do I rotate an API key?**
A: `POST /v1/api-keys` creates a new one, `DELETE /v1/api-keys/{id}`
revokes the old one. See `src/semrag/api/routers/api_keys.py` for the
exact routes (and the live API docs at `http://localhost:8000/docs`).

**Q: How do I run the test suite?**
A: `make test` or `pytest tests/unit`. Postgres-dependent tests skip
gracefully if no DB is reachable. See [tests/README.md](../tests/README.md).

**Q: Where are the runbooks?**
A: `docs/operations/`. One per failure mode (database, embedding, LLM,
indexing, jobs, latency, memory, rollback) plus an escalation matrix.

---

## 40. Where to go next

You have a working SemRAG install. From here:

- **[README.md](../README.md)** — the one-page tour.
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** — long-form system design.
- **[DEPLOYMENT.md](../DEPLOYMENT.md)** — Docker, Compose, Kubernetes, Helm.
- **[ENGINEERING_PLAYBOOK.md](./ENGINEERING_PLAYBOOK.md)** — how to add a
  parser, provider, agent tool, or metric.
- **[src/semrag/README.md](../src/semrag/README.md)** — package map.
- **[docs/operations/](./operations/)** — on-call runbooks.
- **[RELEASE_PROCESS.md](../RELEASE_PROCESS.md)** — release procedure
  (only if you are cutting a release).
- **[CHANGELOG.md](../CHANGELOG.md)** — what shipped in each release.

For interactive API docs while the server is running, open
`http://localhost:8000/docs` (Swagger) or `http://localhost:8000/redoc`.

---

## Fresh Install Checklist

Tick each line as you complete it. If you cannot tick one, the
"Common failure" rows above tell you where to look.

- [ ] Python 3.11 or 3.12 installed
- [ ] Git installed
- [ ] Repository cloned (`git clone <repo-url> semrag`)
- [ ] Working directory is `semrag/`
- [ ] Virtual environment created (`python -m venv .venv`)
- [ ] Virtual environment activated (prompt prefixed with `(.venv)`)
- [ ] `pip install -r requirements-dev.txt` succeeded
- [ ] `pip install -e .` succeeded
- [ ] `semrag --help` works
- [ ] `.env` created from `.env.example`
- [ ] Postgres 16 + pgvector reachable (`psql ... -c "SELECT 1"` works)
- [ ] Ollama running, `nomic-embed-text` and `qwen2.5:7b` pulled
- [ ] `alembic upgrade head` succeeded
- [ ] `alembic current` shows the latest revision
- [ ] `python -c "from semrag.core.settings import get_settings; get_settings()"` succeeds
- [ ] `uvicorn semrag.api.app:app --port 8000` starts cleanly
- [ ] `curl http://localhost:8000/livez` returns `{"status":"alive"}`
- [ ] `curl http://localhost:8000/readyz` returns `{"status":"ready",...}`
- [ ] `curl http://localhost:8000/version` returns build metadata
- [ ] `curl http://localhost:8000/metrics` returns Prometheus text

---

## First Successful Run

You have a working SemRAG install when **all** of these are true:

- [ ] `curl -s http://localhost:8000/livez` returns `{"status":"alive"}`
- [ ] `curl -s http://localhost:8000/readyz` returns HTTP 200 and
  `"status": "ready"` with every component reporting `"ok"`
- [ ] `curl -s http://localhost:8000/version` returns a JSON object
  containing `version`, `git_commit`, `python_version`, `environment`
- [ ] `semrag index --project-id <myproj> --root <abs/path>` reports
  `Indexed N chunks across M files ...`
- [ ] `psql ... -c "SELECT count(*) FROM chunks WHERE project_id='<myproj>'"`
  returns a positive integer
- [ ] `semrag ask --project-id <myproj> --question "..."` returns a
  non-empty answer with at least one `[Source: ...]` citation
- [ ] `curl -X POST http://localhost:8000/v1/projects/<myproj>/answer ...`
  returns a JSON object with `answer`, `citations`, `sources`,
  `iterations`, `tools_used`
- [ ] `curl -s http://localhost:8000/metrics | grep semrag_` shows at
  least one `semrag_*` counter or gauge

If all of those are true, you are ready to use SemRAG. Move to
[DEPLOYMENT.md](../DEPLOYMENT.md) for production deployment, or
[ENGINEERING_PLAYBOOK.md](./ENGINEERING_PLAYBOOK.md) if you want to
extend the platform.

---

## Troubleshooting Matrix

| Problem | Likely Cause | How to Verify | Fix | Related Documentation |
|---------|--------------|---------------|-----|------------------------|
| `ModuleNotFoundError: semrag` | Virtualenv not active or `pip install -e .` not run | `which python` should show `.venv/bin/python` | `source .venv/bin/activate && pip install -e .` | Step 9 |
| `pip install` fails on `psycopg2-binary` | Old pip | `pip --version` | `pip install --upgrade pip` then retry | Step 9 |
| `connection refused` on `localhost:5432` | Postgres not running | `pg_isready -h localhost` | `sudo service postgresql start` or `docker start semrag-postgres` | Step 11 |
| `pgvector extension not installed` | Wrong Postgres image | `psql -c "SELECT extname FROM pg_extension WHERE extname='vector'"` | Use `pgvector/pgvector:pg16` image | Step 12 |
| `connection refused` on `localhost:11434` | Ollama not running | `curl http://localhost:11434/api/tags` | `sudo systemctl start ollama` | Step 14 |
| `model 'nomic-embed-text' not found` | Model not pulled | `ollama list` | `ollama pull nomic-embed-text && ollama pull qwen2.5:7b` | Step 14 |
| `alembic upgrade head` fails | DB connectivity | `psql "$SEMRAG_DATABASE_URL" -c "SELECT 1"` | Restore DB connection, then retry | Step 16 |
| `/readyz` reports `database: "FATAL: ..."` | Credentials wrong | `psql "$SEMRAG_DATABASE_URL" -c "SELECT 1"` | Fix `SEMRAG_DATABASE_URL` | Step 32 |
| `Default development API key 'dev-local-key' is not allowed in production` | Prod env with dev key | `echo $SEMRAG_ENVIRONMENT` | Set `SEMRAG_API_KEYS` to a real key | Step 10, §30 |
| `Embedding dimension mismatch: configured 768, server returned N` | Wrong `SEMRAG_EMBEDDING_DIMENSION` | `curl -d '{"model":"nomic-embed-text","input":["test"]}' http://localhost:11434/v1/embeddings` | Match `SEMRAG_EMBEDDING_DIMENSION` to the model | Step 15, §33 |
| `(Agent hit its iteration limit without a final answer.)` | Query too narrow | `semrag ask --debug --question "..."` | Rephrase or raise `SEMRAG_LLM_MAX_ITERATIONS` | Step 23, §34 |
| `Failed to reach LLM endpoint` | Ollama down or wrong model | `curl http://localhost:11434/api/tags` | `ollama serve`, then `ollama pull <model>` | Step 14, §34 |
| `Project root does not exist` | Wrong path passed to `index` | `ls -la /path/to/project` | Use an absolute path on the *machine running the CLI* | Step 21 |
| `/readyz` returns 503 | One of the dependencies is down | `curl -s http://localhost:8000/readyz \| jq` | Inspect the `checks` object; each component has a status string | Step 20, [docs/operations/database-outage.md](./operations/database-outage.md) |
| `Job store backend "memory" is not safe in production` | Multi-replica with in-memory store | `echo $SEMRAG_JOB_STORE_BACKEND` | Set `SEMRAG_JOB_STORE_BACKEND=database` | Step 19 |
| `from_cache: false` for every query | In-memory cache, no Redis | `echo $SEMRAG_QUERY_CACHE_BACKEND` | Set `SEMRAG_QUERY_CACHE_BACKEND=redis` and run Redis | Step 13 |
| `Rate limit exceeded` (HTTP 429) | Too many requests per minute | `curl -v ...` shows `Retry-After` | Raise `SEMRAG_RATE_LIMIT_PER_MINUTE` or back off | Step 25, [docs/operations/llm-outage.md](./operations/llm-outage.md) |
| `context window exceeded` | `query + chunks > SEMRAG_LLM_CONTEXT_WINDOW` | `curl -s http://localhost:8000/readyz \| jq` for settings | Lower `SEMRAG_FINAL_TOP_K` or raise `SEMRAG_LLM_CONTEXT_WINDOW` | Step 23, §34 |
| `embedding provider dir not writable` | `SEMRAG_EMBEDDING_PROVIDER_DIR` doesn't exist | `ls -la /tmp/semrag_embedding_providers` | `mkdir -p` the dir or change the env var | Step 15 |
| `docker compose up` — port 8000 in use | Another process on 8000 | `lsof -i :8000` | Set `SEMRAG_API_PORT=8080` in `.env` | Step 36 |
