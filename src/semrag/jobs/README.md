# `src/semrag/jobs` — Background Indexing Job System

> **The seam that lets the HTTP API return immediately on
> `POST /v1/projects/{id}/index`.** This README explains what the
> jobs package owns, how the state machine works, and how the worker,
> executor, and store compose. It is `jobs`-specific; the end-to-end
> indexing walkthrough lives in
> [developer_guide.md §4](../../../docs/developer_guide.md#4-end-to-end-indexing-lifecycle)
> and the HTTP entry points are in
> [src/semrag/api/README.md](../api/README.md). For "where do I
> change X?" use [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md).

Intended audience: **Developers** and **Maintainers** who touch job
dispatch, the executor seam, the worker loop, or the durable job
store.

---

## 1. Purpose of the package

The `jobs` package exists so the HTTP API can return a `202 Accepted`
on `POST /v1/projects/{id}/index` instead of running the full
scan → parse → IR → chunk → embed → persist pipeline synchronously
on the request thread. Without it, indexing a large repo would tie
up a request thread for minutes and would be silently lost on any
process restart.

The package owns four things:

1. **The job model** — `Job` + `JobState` (`QUEUED`, `RUNNING`,
   `COMPLETED`, `FAILED`, `CANCELLED`).
2. **The job store** — a `JobStore` ABC with two implementations:
   `InMemoryJobStore` (single-replica, dev/test) and
   `PostgresJobStore` (durable, default).
3. **The job manager** — `JobManager` exposes create / update /
   fetch / cancel / retry on top of any `JobStore`.
4. **The worker function** — `indexing_worker(...)` is the actual
   entry point a polling worker calls to run the indexing pipeline.

The `JobExecutor` ABC (`ThreadPoolJobExecutor` + `ExecutorFactory`)
exists as an infrastructure seam and is currently used by the
`/readyz` health check (the process-wide executor's liveness), not
by the worker dispatch — see [Executor model](#6-executor-model).

> **Why polling and not a push queue?** A push-based queue (Redis /
> RabbitMQ / SQS) would be the production ideal, but the worker
> module's docstring makes the constraint explicit: reuse the
> Postgres store we already have. Two workers polling at the same
> instant cannot both succeed because the in-flight set is per-
> process. Cross-replica durability flows from the persistent
> `PostgresJobStore` — see [Job persistence](#8-job-persistence).

---

## 2. Package structure

```
src/semrag/jobs/
├── __init__.py                  package marker (intentionally empty)
├── models.py                    Job, JobState, _isoformat — the
│                                 dataclass shape and the 5-value enum
├── store.py                     JobStore ABC + InMemoryJobStore
├── postgres_store.py            PostgresJobStore + IndexingJobRow
│                                 (SQLAlchemy ORM) + _row_to_job
├── manager.py                   JobManager: create / get / update /
│                                 delete / list / cancel / retry
├── executor.py                  JobExecutor ABC + ThreadPoolJobExecutor
│                                 + ExecutorFactory.create_executor(settings)
├── pipeline.py                  PipelineStage enum with relative weights
│                                 + ProgressCalculator.calculate_progress(...)
└── worker.py                    indexing_worker(...) — the actual
                                  pipeline driver; the polling
                                  WorkerService lives outside this
                                  package in src/semrag/worker.py
```

Seven files, each one concern. `__init__.py` is intentionally empty;
nothing is re-exported at the package level. Callers import directly
from `semrag.jobs.manager`, `semrag.jobs.worker`, etc.

---

## 3. Responsibilities of every module

### `models.py` — the dataclass and the state enum

`Job` is a `@dataclass` with the fields the store, the HTTP API,
and the worker all need:

- Identity: `job_id` (UUIDv4 string), `project_id`, `root_path`.
- Lifecycle: `state` (`JobState`), `progress` (0–100), `current_stage`
  (string name of the active `PipelineStage`).
- Timing: `started_at`, `completed_at`, `duration` (seconds).
- Diagnostics: `error` (string), `statistics` (free-form dict).
- Retry: `retry_count` (int).

`Job.to_dict()` is the wire shape — keys are explicitly named
(`state` is lower-cased, datetime fields are ISO-8601 strings via
`_isoformat`). The HTTP `routers/projects.py` calls this directly
for `GET /jobs/{job_id}`.

`JobState` is a 5-value `enum` (`QUEUED`, `RUNNING`, `COMPLETED`,
`FAILED`, `CANCELLED`). The transition logic lives in `manager.py`
and the worker; the enum itself is pure data.

### `store.py` — the in-memory backend and the ABC

`JobStore` is the abstract base class. Its five methods are
`create_job`, `get_job`, `update_job`, `delete_job`, `list_jobs`.
The `update_job` signature uses `**kwargs` so callers can update
arbitrary fields without a method explosion — the implementation
loops over kwargs and skips ones the dataclass doesn't have.

`InMemoryJobStore` is the single-process implementation. It uses a
`threading.RLock` so concurrent updates from the worker thread and
the request thread are safe. There is no separate process-safety
test in this package; the in-memory store is single-replica only
and the production code path uses `PostgresJobStore`.

### `postgres_store.py` — the durable backend

`PostgresJobStore` is the production default. The `IndexingJobRow`
ORM mirrors the `Job` dataclass field-for-field, with two
additions:

- `statistics` is a JSONB column (`JSON().with_variant(JSONB(),
  "postgresql")`) because the dict carries counters, timings, and
  throughput values that change shape with the indexing pipeline.
- `created_at` (added at the row level) is the queueing time. It
  is NULL on the dataclass — the worker doesn't read it back, but
  the row column is useful for oldest-job queries and admin
  inspection.

`PostgresJobStore` opens a fresh session per call (`self._session_factory()`
context manager). Every method commits on success; the row is
written in one transaction. There is no
`SELECT ... FOR UPDATE SKIP LOCKED` claim transition — see
[Worker lifecycle](#7-worker-lifecycle) for what actually enforces
"one runner per job".

`_row_to_job` is the one place that converts an `IndexingJobRow`
back to a `Job` dataclass. The conversion is lossy in one
direction: `created_at` lives only on the row, not on the dataclass.

### `manager.py` — the job operations surface

`JobManager` is the only entry point application code should use
to operate on jobs. It holds a `JobStore` and delegates every
method. The two methods that are not pure delegations:

- **`create_job(project_id, root_path) -> (Job, created)`** — if
  the store already has a `QUEUED` or `RUNNING` job for the same
  project, return it with `created=False`. Otherwise mint a new
  `job_id = uuid.uuid4()`, persist it, and increment
  `semrag_indexing_jobs_queued_total{project_id}`. The de-dupe
  is what implements the "one runner per project" invariant at
  the API boundary.

- **`cancel_job(job_id) -> bool`** — only fires when the job is
  in `QUEUED` or `RUNNING`. The transition is direct: `state =
  CANCELLED`. The worker observes the state on its next polling
  check and writes `completed_at` / `duration` (see
  [Worker lifecycle](#7-worker-lifecycle)).

- **`retry_job(job_id, max_retries) -> bool`** — increments
  `retry_count` and resets `state → QUEUED`, `progress → 0`,
  `current_stage → ""`, `started_at / completed_at / duration /
  error → None`. **Statistics are deliberately not reset** — the
  comment in the code is explicit about this: "we will leave
  statistics as is (they will be overwritten by the new run)".
  This avoids surfacing a partially-blended iteration as a clean
  slate.

> `retry_job` is defined but **not called from any production
> path today**. The router surface exposes only cancel, not retry.
> It is part of the public API and intended for follow-up use
> (operator-driven retries via a future admin endpoint).

### `executor.py` — the executor seam

`JobExecutor` is the ABC. `ThreadPoolJobExecutor` is the only
implementation; it wraps a `concurrent.futures.ThreadPoolExecutor`
with a `_shutdown` flag and a `submit(job_id, fn, on_complete)`
method that runs `fn` and calls `on_complete` (no future passed
to the callback). `ExecutorFactory.create_executor(settings)`
returns either `ThreadPoolJobExecutor` (when settings expose
`job_executor_type = "threadpool"`, the default) or falls through
to threadpool for unknown types.

The executor is **not** the dispatch mechanism for the worker.
The actual dispatch is `WorkerService` in `src/semrag/worker.py`
which polls for `QUEUED` jobs and calls `indexing_worker(...)`
directly on the worker thread. The executor is wired into the
process for the `/readyz` health check and as the future home of
alternative dispatch backends (Celery, RQ, …).

### `pipeline.py` — the stage weights and progress math

`PipelineStage` is a 9-value enum with stage end-weights:

| Stage        | Weight |
| ------------ | ------ |
| `QUEUED`     | 0.0    |
| `SCANNING`   | 0.05   |
| `PARSING`    | 0.20   |
| `IR`         | 0.45   |
| `CHUNKING`   | 0.65   |
| `EMBEDDING`  | 0.90   |
| `PERSISTING` | 1.0    |
| `COMPLETED`  | 1.0    |
| `FAILED`     | 1.0    |
| `CANCELLED`  | 1.0    |

`ProgressCalculator.calculate_progress(completed_stages,
current_stage, current_stage_progress)` returns `int(progress *
100)` between 0 and 100. The implementation uses internal
stage-range dictionaries (`SCANNING = (0.0, 0.05)`, etc.) which
match `weight()` for the end weight. The progress formula is
`start_weight + current_stage_progress * (end_weight - start_weight)`.

The worker is responsible for calling `calculate_progress` with
the running `completed_stages` list and the current stage — see
[Worker lifecycle](#7-worker-lifecycle).

### `worker.py` — the indexing pipeline driver

`indexing_worker(job_id, project_id, root_path, job_manager,
settings, session_factory, embedding_provider_factory)` is the
function a polling worker calls to run one job. It is the same
pipeline the service layer's `IngestionService.index()` runs, but
with two extras:

- **Stage-by-stage progress reporting** via `_update_progress` and
  `_update_job_statistics`. After each stage, the worker writes
  `progress` (0–100) and `current_stage` (the lowercase stage
  name) back to the store.
- **Cancellation polling** between stages — see
  [Retry & cancellation](#10-retry--cancellation).

The worker opens **one** `session_factory()` and runs the entire
pipeline inside it. The persist stage is the single commit
boundary; everything before it is `flush()`-only so a failure
anywhere rolls back the whole transaction cleanly. The outer
`try/except Exception` calls `_report_failure` so the failure path
explicitly writes `state=FAILED`, `error`, `completed_at`,
`duration`, increments the `INDEXING_JOBS_FAILED` counter, and
emits a structured ERROR log.

The worker also writes the bookkeeping Prometheus counters:
`INDEXING_JOBS_COMPLETED`, `INDEXING_DURATION`,
`INDEXING_FILES_PROCESSED`, `INDEXING_CHUNKS_GENERATED`,
`INDEXING_CHUNKS_EMBEDDED`. See [core/metrics.py]
(../core/README.md#6-metrics) for the metric inventory.

---

## 4. Job lifecycle

Every job goes through the same five-phase arc:

```
Client                Router              JobManager            Store                Worker
  │  POST /index  ──▶ │                    │                     │                    │
  │                   │ JobManager.create  │                     │                    │
  │                   │ ──────────────────▶│                     │                    │
  │                   │                    │ create_job (or      │                    │
  │                   │                    │ return existing)    │                    │
  │                   │                    │ ──────────────────▶ │                    │
  │                   │                    │ ◀── (Job, created)  │                    │
  │                   │  202 / 200         │                     │                    │
  │  ◀──────────────  │                    │                     │                    │
  │                   │                    │                     │                    │
  │  GET /jobs/{id}   │                    │                     │                    │
  │  ──────────────▶  │ JobManager.get_job │                     │                    │
  │                   │ ──────────────────▶│                     │                    │
  │                   │                    │ ──────────────────▶ │                    │
  │                   │                    │ ◀── Job             │                    │
  │                   │  Job.to_dict()     │                     │                    │
  │  ◀──────────────  │                    │                     │                    │
  │                   │                    │                     │    poll            │
  │                   │                    │                     │ ◀────────────────  │
  │                   │                    │                     │ ── Job             │
  │                   │                    │                     │ INDEXING_JOBS_     │
  │                   │                    │                     │ QUEUED.inc         │
  │                   │                    │                     │                    │
  │                   │                    │                     │ state=RUNNING      │
  │                   │                    │                     │ ──────────────────▶│
  │                   │                    │                     │                    │ run pipeline
  │                   │                    │                     │ (progress/stage    │
  │                   │                    │                     │  updates)          │
  │                   │                    │                     │ ◀─────────────────│
  │                   │                    │                     │ state=COMPLETED    │
  │                   │                    │                     │ ──────────────────▶│
  │                   │                    │                     │    run_completed   │
```

Concrete timestamps of interest:

- **t=0** — `POST /v1/projects/{id}/index` reaches the router.
  `JobManager.create_job` either returns the existing
  `QUEUED`/`RUNNING` job for the same project or persists a new
  `QUEUED` job. The router returns `202` (new) or `200`
  (existing).
- **t=polled** — the worker thread reads `list_jobs()`, finds the
  `QUEUED` job, flips it to `RUNNING` via `update_job`, and calls
  `indexing_worker(...)`.
- **t=stage boundaries** — six checkpoints (SCANNING, PARSING, IR,
  CHUNKING, EMBEDDING, PERSISTING). At each one the worker writes
  `progress` and `current_stage`, then checks the job's state for
  `CANCELLED`. See [Retry & cancellation](#10-retry--cancellation).
- **t=commit** — the single transaction boundary at the end of
  `ingest.persist`. On success, the worker writes
  `state=COMPLETED`, `completed_at`, `duration`.
- **t=failure** — any exception inside the worker reaches the
  outer `try/except`, which calls `_report_failure` to write
  `state=FAILED`, `error`, `completed_at`, `duration`.

---

## 5. State transitions

The state machine, verified against `models.py:JobState` and the
worker:

```
QUEUED ──claim──▶ RUNNING ──ok─────▶ COMPLETED
                     │
                     ├─error─────▶ FAILED
                     └─cancel────▶ CANCELLED
QUEUED ──cancel─────────────▶ CANCELLED   (router-side, before worker claim)
FAILED  ──retry_job──▶ QUEUED            (manager-side, see §10)
```

What each transition does:

| Transition               | Where it happens                                          | What is written                                                                          |
| ------------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `QUEUED → RUNNING`       | `WorkerService.claim_next_queued` (in `src/semrag/worker.py`) | `state=RUNNING`, `started_at=now`                                                       |
| `RUNNING → COMPLETED`    | `indexing_worker`, after the `PERSISTING` stage          | `state=COMPLETED`, `completed_at=now`, `duration=elapsed`                                |
| `RUNNING → FAILED`       | `_report_failure` (outer `except Exception`)              | `state=FAILED`, `error=str(exc)`, `completed_at=now`, `duration=elapsed`                 |
| `RUNNING → CANCELLED`    | `indexing_worker`, at each of 6 stage boundaries          | `state=CANCELLED`, `completed_at=now`, `duration=elapsed`                               |
| `QUEUED → CANCELLED`     | `JobManager.cancel_job` via `POST /jobs/{id}` router     | `state=CANCELLED` (no `completed_at`/`duration` — the worker fills those in on next poll, see below) |
| `FAILED → QUEUED` (retry)| `JobManager.retry_job` (not currently called from any router) | `state=QUEUED`, `progress=0`, `current_stage=""`, `started_at/completed_at/duration/error=None`, `retry_count++` |

> **The `CANCELLED` write is two-phase.** The router sets
> `state=CANCELLED` immediately so the user's `DELETE` returns
> success. The worker then sees the state on its next polling
> check and writes `completed_at` / `duration`. Between these two
> writes the job is in `CANCELLED` state but has no
> `completed_at` — a `GET /jobs/{id}` returned in that window will
> show `state=cancelled` with `completed_at=null`. This is
> intentional; the alternative (re-checking the state in the
> router) is wasted work because the worker always gets there
> eventually.

The transition is **not** atomic at the store level — neither
`InMemoryJobStore` nor `PostgresJobStore` uses
`SELECT ... FOR UPDATE SKIP LOCKED` or any other row-locking
construct. The at-most-one-runner invariant is enforced by two
separate mechanisms:

1. `WorkerService._inflight: set[str]` in `src/semrag/worker.py`
   tracks per-process job-ids; under the `with self._inflight_lock`
   guard, the worker checks both the store's state and the
   in-flight set before claiming.
2. `JobManager.create_job` returns the existing `QUEUED`/`RUNNING`
   job for the same project, so a second `POST /index` for the
   same project cannot create a parallel job.

> **The worker docstring describes a `SELECT ... FOR UPDATE SKIP
> LOCKED` claim transition.** The actual implementation in
> `WorkerService.claim_next_queued` is **not** atomic at the
> store level; it iterates the store's `list_jobs()` and uses the
> in-memory `_inflight` set. The class docstring is aspirational;
> the code is what runs. The de-duplication at `create_job`
> closes the gap for the common case (same project).

---

## 6. Executor model

The `JobExecutor` ABC and `ThreadPoolJobExecutor` exist as the
infrastructure seam for "how jobs are dispatched". The current
implementation is a `concurrent.futures.ThreadPoolExecutor` with
the project's shutdown-on-init semantics; `ExecutorFactory` maps
settings to a concrete instance.

`ExecutorFactory.create_executor(settings)` reads two settings:

- `settings.job_executor_type` (default `"threadpool"`).
- `settings.job_executor_max_workers` (default `4`).

Only `threadpool` is implemented today; the factory's `else` branch
still returns `ThreadPoolJobExecutor` so an unknown value logs
nothing and silently falls back to the only implementation.

The `_shutdown` flag is the only state the executor holds. After
`shutdown(wait=True/False)`, `submit(...)` raises `RuntimeError`
because the executor is shutdown. `get_job_executor()` in
`api/dependencies.py` lazily re-creates the executor after shutdown
so tests that create multiple `TestClient` lifespans don't see a
perpetually shut-down executor.

> **The executor is not the dispatch mechanism.** A grep across
> `src/semrag/` for `executor.submit(...)` finds only the
> implementation call inside `ThreadPoolJobExecutor.submit` itself.
> The actual job dispatch is `WorkerService.run_forever` in
> `src/semrag/worker.py`, which polls the store and calls
> `indexing_worker(...)` on the worker thread. The executor is
> exposed via `get_job_executor()` for the `/readyz` health check
> (asserting "executor exists and is not shut down") and is the
> planned replacement seam for push-based queues.

---

## 7. Worker lifecycle

The polling worker is `WorkerService` in `src/semrag/worker.py`
(not in `src/semrag/jobs/`). The lifecycle is:

```text
start
  │
  ▼
build_worker_service(settings)
  │ reads job_store_backend → PostgresJobStore(session_factory)
  │                                  else → InMemoryJobStore()
  │
  ▼
WorkerService(job_store, session_factory, settings).run_forever()
  │
  │  loop until stop() is set:
  │    job = claim_next_queued()         # see below
  │    if job is None: wait poll_interval
  │    _run_job(job)                     # see below
  │
  ▼
shutdown
  │
  ▼
worker.stop() → run_forever() exits
```

### `claim_next_queued()`

Iterates `self._job_store.list_jobs()`, picks the first
`JobState.QUEUED` job whose `job_id` is not in `_inflight`, and
flips it to `RUNNING` via `store.update_job(job_id,
state=JobState.RUNNING)`. The `_inflight` set is the per-process
guard against double-claiming. The `with self._inflight_lock`
guard makes the check-and-add atomic *within one process*.

> **Cross-process behavior.** Two `WorkerService` instances in two
> different processes can both call `claim_next_queued` at the
> same instant. The store's `update_job` is the only
> serialization point — both calls succeed, both add the job_id
> to their own `_inflight` set, and **both** call
> `indexing_worker(...)` on the same job.
> [Common mistakes](#14-common-mistakes) covers the mitigations.

### `_run_job(job)`

Calls `indexing_worker(...)` from `jobs/worker.py` with the
canonical argument list. The worker owns the job's lifecycle
from `RUNNING` to one of `COMPLETED`, `FAILED`, or `CANCELLED`.
Exception handling is at the `_run_job` level so the worker
process does not crash on a single bad job:

```python
try:
    indexing_worker(...)
except Exception as exc:
    logger.error("worker crashed while running job", ...)
    metrics_module.INDEXING_JOBS_FAILED.labels(project_id=...).inc()
finally:
    self._inflight.discard(job.job_id)
```

The `_inflight` discard in `finally` is what makes the worker
resilient to soft crashes — the next tick re-claims the same job,
sees it in `RUNNING`, and the de-duplication at `create_job`
keeps a second `POST /index` from spawning a parallel job.

### Graceful shutdown

`WorkerService.stop()` sets a `threading.Event`; the poll loop
sleeps on the event with `wait(poll_interval)` so a SIGTERM is
responsive within one poll interval (default
`Settings.worker_poll_interval_seconds = 1.0`). In-flight jobs
finish on the worker thread before the process exits.

---

## 8. Job persistence

### The row model

`IndexingJobRow` mirrors `Job` field-for-field with one
addition: `created_at` (the queueing time, default
`datetime.utcnow`). The column is `DateTime(timezone=False)` —
Postgres stores it as UTC-naive; the conversion back to a
`datetime` is done implicitly by SQLAlchemy. The `state` column
is `String(16)` storing the enum's `name` (e.g. `"QUEUED"`), and
`_row_to_job` converts back via `JobState[name.upper()]`.

The `statistics` field is `JSON().with_variant(JSONB(),
"postgresql")` — a JSONB column on Postgres, JSON on other
dialects. The free-form dict carries counters, timing, and
throughput values that change shape with the indexing pipeline.

### In-memory vs Postgres — the durable default

| Backend               | Lifetime                      | Cross-replica? | Cross-restart? | Production default? |
| --------------------- | ----------------------------- | -------------- | -------------- | ------------------- |
| `InMemoryJobStore`    | single process lifetime       | **No**         | **No**         | ❌ (dev/test only) |
| `PostgresJobStore`    | persistent in `indexing_jobs`| ✓              | ✓              | ✓                   |

The selection is via `Settings.job_store_backend`:

- `database` → `PostgresJobStore(session_factory)`.
- `memory` → `InMemoryJobStore()`.

`api/dependencies.py` builds the store at module load with a
`reset_job_store_for_tests()` hook that re-reads settings and
rebuilds. Production never hits the reset path.

> **Production default is `database`.** A fresh production deploy
> that boots into `memory` would silently lose queued jobs on
> every restart and duplicate them across replicas —
> [startup_validation.py](../api/health.py) is the boot-time
> config validation that flags this. The single-replica in-memory
> mode is still available for dev/test by setting
> `SEMRAG_JOB_STORE_BACKEND=memory`.

### The single transaction boundary

`indexing_worker` opens **one** session for the entire pipeline
and commits **once** at the end of `PERSISTING`:

```text
session = session_factory()
try:
    # ... scan, parse, IR, chunk, embed (all flush() only)
    # persist stage
    IRGraphRepository(session).save(ir_graph)
    session.flush()
    if change_set.deleted_paths:
        PgvectorStore(session).delete_by_paths(...)
    if vector_chunks:
        PgvectorStore(session).upsert_chunks_bulk(vector_chunks)
        session.flush()
    session.commit()        ← single commit
except Exception:
    session.rollback()       ← explicit rollback
    raise
finally:
    session.close()
```

A failure between `flush()` and `commit()` rolls back the entire
run; partial state never persists. A failure before any flush
(a bad root path, a scan exception) leaves the previous index in
place.

The job's `state` write is a separate transaction:

- `state=RUNNING` → `update_job` (separate commit).
- `state=COMPLETED` / `state=FAILED` / `state=CANCELLED` →
  `update_job` (separate commit).

These are not part of the pipeline transaction because the
`JobStore` is a separate connection from the worker session —
mixing them would require either a single session for both
purposes (scope leak) or a two-phase commit (overkill).

---

## 9. Progress tracking

Progress is computed from the `PipelineStage` enum weights and
the worker's idea of which stages have completed. The progression
is:

| Stage transition   | Weight range     | Cumulative progress at end |
| ------------------ | ---------------- | -------------------------- |
| QUEUED → SCANNING  | 0.00 → 0.05      | 5%                        |
| SCANNING → PARSING | 0.05 → 0.20      | 20%                       |
| PARSING → IR       | 0.20 → 0.45      | 45%                       |
| IR → CHUNKING      | 0.45 → 0.65      | 65%                       |
| CHUNKING → EMBEDDING | 0.65 → 0.90    | 90%                       |
| EMBEDDING → PERSISTING | 0.90 → 1.00  | 100%                      |
| PERSISTING → COMPLETED | 1.00 → 1.00  | 100%                      |

The worker calls `_update_progress(stage, current_stage_progress)`
after every stage, which:

1. Calls `progress_calc.calculate_progress(completed_stages,
   current_stage, current_stage_progress)` to get a 0–100 int.
2. Calls `_update_job_progress(job_manager, job_id, progress,
   stage.name.lower())` to write the integer and the lowercase
   stage name back to the store.

`statistics` are merged incrementally via `_update_job_statistics`:

```python
updated_stats = {**job.statistics, **new_stats}
job_manager.update_job(job_id, statistics=updated_stats)
```

The worker writes per-stage counters:

- `files_scanned` (after SCANNING).
- `files_parsed` (after PARSING).
- `chunks_generated` (after CHUNKING).
- `chunks_embedded` (after EMBEDDING).
- `chunks_persisted`, `added`, `modified`, `deleted` (after
  PERSISTING).
- `elapsed_time`, `estimated_remaining_time`, `throughput` (on
  completion).

Operators see the live progress in `GET /jobs/{id}` via
`Job.to_dict()`. The structured log line at each stage includes
the same numbers (`Scanning completed, files_scanned=N`, etc.) so
the live progress is also visible via the logs.

---

## 10. Retry & cancellation

### Retry

`JobManager.retry_job(job_id, max_retries)` is the public retry
surface. It:

1. Refuses if the job is not in `FAILED` or `QUEUED`.
2. Refuses if `job.retry_count >= max_retries`.
3. Increments `retry_count`.
4. Resets `state → QUEUED`, `progress → 0`,
   `current_stage → ""`, `started_at / completed_at / duration /
   error → None`.
5. **Does not reset `statistics`** — the run will overwrite the
   keys it cares about, and stale counters from a failed run
   remain visible until then.

> **No code path currently calls `retry_job`.** The HTTP
> surface exposes `POST /jobs/{id}/cancel` only. The method
> exists in the manager for operator-driven retries via a future
> admin endpoint.

### Cancellation

A job is cancelled one of two ways:

1. **Via the API** — `DELETE /v1/projects/{id}/jobs/{id}` →
   `JobManager.cancel_job(job_id)` → `update_job(state=CANCELLED)`.
   The router returns `200 {"status": "cancellation requested"}`
   once the state is flipped.
2. **From the worker** — at each of the six stage boundaries
   (SCANNING, PARSING, IR, CHUNKING, EMBEDDING, PERSISTING), the
   worker re-reads the job's state and exits early if it is
   `CANCELLED`. The cleanup writes `completed_at` and `duration`.

> **Mid-stage cancellation is not supported.** The worker
> completes the current stage before observing the cancelled
> flag. Implementing mid-stage cancel would break invariants in
> chunking / embedding batches (a partial chunk flush would leave
> the vector index in an inconsistent state). Pretend
> cancellation is polling-based; it is the documented contract.

The router-side guard rejects cancellation when the job is not in
`QUEUED` or `RUNNING` (returns `400` with the current state in
the message). Once the worker has flipped the state to
`COMPLETED` / `FAILED` / `CANCELLED`, the cancel endpoint returns
`400` — the job is no longer cancellable.

---

## 11. Failure handling

The worker is structured to bound failure propagation:

```text
indexing_worker()
  try:
    state → RUNNING
    pipeline (scanning → parsing → ir → chunking → embedding → persisting)
    session.commit()                     ← single commit boundary
    state → COMPLETED
    metrics → INDEXING_JOBS_COMPLETED.inc(), INDEXING_DURATION.observe(...)
  except Exception:
    session.rollback()
    raise
  finally:
    session.close()

# Outer layer in WorkerService._run_job:
try:
    indexing_worker(...)
except Exception as exc:
    metrics → INDEXING_JOBS_FAILED.inc()
    logger.error("worker crashed while running job", ...)
finally:
    _inflight.discard(job.job_id)
```

### What gets recorded on failure

`_report_failure(job_manager, job_id, error, start_time)` writes:

- `state=FAILED`.
- `error=str(exc)`.
- `completed_at=time.time()`.
- `duration=completed_at - start_time`.
- Increments `INDEXING_JOBS_FAILED.labels(project_id=...).inc()`.
- Observes `INDEXING_DURATION.labels(project_id=...).observe(...)`.
- Emits a structured ERROR log with `job_id`, `project_id`,
  `error`, `elapsed_time`.

The store-level commit on `update_job` is independent of the
worker's session — `update_job` opens its own connection, so the
failure-state write is durable even if the worker's session was
rolled back.

### What does NOT auto-retry

The worker does not retry on failure. A failed job stays in
`FAILED` state until an operator calls `retry_job` (currently
no router surface for this). The recovery model is "operator
investigates, then explicitly retries" — there is no silent
backoff loop.

### What survives an exception

- The session is rolled back, so the in-flight transaction's
  partial writes are gone.
- The job is in `FAILED` state with the error message recorded.
- The in-flight set discard happens in `finally`, so the next
  poll tick re-claims the same job-id.
- The `JobManager.create_job` de-duplication still applies to a
  new `POST /index` for the same project — the user gets the
  existing `FAILED` job back and can read the error from
  `GET /jobs/{id}`.

---

## 12. Extension points

The package exposes a small, well-defined surface for extension.

| Need                              | Where to change                                                                 | Notes                                                                                     |
| --------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Add a new `JobStore` backend      | `jobs/store.py` (ABC) + a new `MyJobStore(JobStore)` module                    | Register the backend in `api/dependencies.py:_build_job_store(settings)`. Update `startup_validation.py` if a new production-safe backend is added. |
| Add a new `JobExecutor`           | `jobs/executor.py` (ABC) + a new `MyJobExecutor(JobExecutor)` module            | Register the executor in `ExecutorFactory.create_executor(settings)`. Add the new `job_executor_type` to the settings model. |
| Add a new pipeline stage          | `jobs/pipeline.py:PipelineStage` enum + `ProgressCalculator` weights + steps in `indexing_worker` | The stage weight must be both in `weight()` and `stage_ranges`. The ordering of `stage_ranges` start/end weights must be monotonically increasing. |
| Add a new statistics field        | `indexing_worker` — call `_update_job_statistics(job_manager, job_id, {...})` at the relevant stage | The dict is merged in place; existing keys survive. No schema migration needed. |
| Add a new job state               | `jobs/models.py:JobState` enum + transition logic in `manager.py`/`worker.py`  | Update `Job.to_dict()` if the new state needs a different wire format. Update `routers/projects.py` cancel logic. |
| Add a new admin-visible action    | `manager.py` (method) + `routers/projects.py` (route) + `schemas.py` (response) | Mirror the existing `cancel_job` pattern. |

What is **not** an extension point:

- **The `Job` dataclass shape.** Fields are wire-format-stable —
  `Job.to_dict()` is the public response shape. Renaming a field
  is a breaking change.
- **The at-most-one-runner invariant.** `create_job`'s
  `QUEUED`/`RUNNING` de-duplication is the contract the rest of
  the system relies on. Subclassing `JobManager` to bypass it
  will produce duplicate concurrent jobs.
- **The single-commit-per-pipeline transaction.** Splitting the
  commit boundary would let partial state survive a failure.
- **The polling model.** A push-based queue is a future
  replacement, but introducing one today without a feature flag
  would change the deployment profile in ways the existing
  operation runbooks do not cover.

---

## 13. Common debugging paths

| Symptom                                                  | Where to look                                                                                |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Job stuck in QUEUED**                                  | `WorkerService` is not running, or the job is in another worker's `_inflight` set. Check `/readyz` `job_executor` check; check worker process logs. |
| **Job stays in RUNNING after a worker crash**            | The in-flight set was discarded by the worker `finally`, but the store still shows `RUNNING`. The next claim tick will not re-claim it (state is not `QUEUED`); an operator must update the state to `QUEUED` or `FAILED` directly. |
| **Two jobs running for the same project**                | `JobManager.create_job`'s de-duplication was bypassed (e.g. direct `store.create_job(...)` from a non-manager callsite). The manager is the only sanctioned enqueue path. |
| **`POST /jobs/{id}` cancel returns 400**                  | The job is in `COMPLETED` / `FAILED` / `CANCELLED` already. Cancellation is only valid for `QUEUED` / `RUNNING`. The router returns the current state in the message. |
| **Cancellation never takes effect**                      | The worker reads the state at each stage boundary. A long-running stage (e.g. embedding 50k chunks) will not observe the cancellation until the stage finishes. The cancel is honored at the next boundary, not mid-stage. |
| **Job fails but `error` is empty**                       | The exception was raised during `session.flush()` and the rollback path lost the original message. Check the worker process stderr for the traceback. |
| **`/jobs/{id}` returns `state=cancelled` with `completed_at=null`** | The router set `state=CANCELLED` but the worker has not yet polled. The worker fills in `completed_at` / `duration` on the next stage boundary. |
| **Pipeline hangs on a bad embed call**                   | Each stage has its own progress write; check `current_stage` in `GET /jobs/{id}` to see which stage the worker is in. The structured log line at `Setting completed` (each stage) shows the per-stage counter. |
| **Resume a failed job**                                  | `JobManager.retry_job(job_id, max_retries)` — currently no HTTP path; use the Python API or wire a new admin endpoint. |
| **Stats survive a retry**                                | `retry_job` deliberately does not reset `statistics`. The next run will overwrite the keys it produces (`files_scanned`, `chunks_generated`, etc.), but old keys from a failed run linger. |
| **No progress updates observed in `GET /jobs/{id}`**     | The store is `InMemoryJobStore` and the worker is in a different process. The in-memory store is single-process; production must use `PostgresJobStore`. |
| **Worker poll loop is too aggressive**                    | `Settings.worker_poll_interval_seconds` (default 1.0s). Lower numbers react faster but increase DB load. |
| **Job statistics overflow the JSONB column**             | The dict is free-form; there is no schema. If a stage adds a counter with a huge string value, JSONB will accept it. Watch for keys that grow unbounded. |

For more debugging material, see
[developer_guide.md §14](../../../docs/developer_guide.md#14-debugging-workflow)
and the operations runbooks
[docs/operations/job-backlog.md](../../../docs/operations/job-backlog.md) +
[docs/operations/indexing-failure.md](../../../docs/operations/indexing-failure.md).

---

## 14. Common mistakes

These are the mistakes the codebase has already made once or is
set up to make; each explains *why* it is wrong, not just *what*
to do.

- **Using `ThreadPoolJobExecutor.submit(...)` to dispatch a job
  from the router.** The executor is built and exposed, but the
  worker dispatch is the polling `WorkerService`. Submitting a job
  directly through the executor would double-run it once the
  worker poll re-claims it. The executor is the future replacement
  seam, not the current dispatch path.
- **Reaching into `JobStore` directly instead of going through
  `JobManager`.** `create_job` does the at-most-one-runner check;
  raw `store.create_job(...)` does not. The API routers and the
  worker both go through the manager.
- **Assuming the `QUEUED → RUNNING` transition is atomic across
  replicas.** It is not. The worker docstring describes
  `SELECT ... FOR UPDATE SKIP LOCKED`, but the actual code uses
  a per-process `_inflight` set. Two replicas in two processes can
  both observe the job as `QUEUED` and both transition it to
  `RUNNING`. The current mitigations are:
  1. `create_job`'s `QUEUED`/`RUNNING` de-duplication (closes the
     common case of two `POST /index` for the same project).
  2. The single `root_path` arg makes the work idempotent in
     practice (the second run sees the same manifest and the
     `change_set` is empty).
  For stronger guarantees, replace the polling worker with a
  push-based queue (a future migration).
- **Catching a broad exception inside `indexing_worker` and
  silently returning.** The outer `try/except Exception` and
  `_report_failure` are the place to record the failure. An
  inner `try/except: pass` would lose the `FAILED` state, the
  `error` string, and the `INDEXING_JOBS_FAILED` counter
  increment. Catch only what you can recover from.
- **Calling `session.commit()` somewhere inside the pipeline.**
  The single commit boundary is at the end of `PERSISTING`. A
  mid-pipeline commit would let partial state survive a failure
  and break the worker's atomicity guarantee.
- **Holding the `JobManager` or `JobStore` longer than a request
  needs.** Both are constructed by `api/dependencies.py` at
  module load and shared. The router reads from the manager, the
  worker writes to the store. Either piece reading and writing
  concurrently is fine because both stores are thread-safe; the
  mistake is holding a reference past the request boundary (e.g.
  on a singleton) and expecting the next request to see the
  latest state.
- **Resetting `statistics` in `retry_job`.** The comment in
  `manager.py:retry_job` is explicit: "we do not reset
  statistics". The dict is the merge target for `_update_job_statistics`
  and the next run's per-stage writes will overwrite the keys it
  cares about. Resetting the whole dict would lose context from
  prior failed runs.
- **Adding a stage to `PipelineStage` without updating both
  `weight()` and the internal `stage_ranges` dict in
  `ProgressCalculator`.** The two tables must stay in sync or
  the formula `start_weight + progress * (end - start)` will
  produce wrong progress percentages. The `int(progress * 100)`
  cast means small stages compound errors visibly.
- **Treating the worker thread as infinitely patient.** A
  long-running stage (large embedding, slow Postgres) blocks the
  `_inflight` slot for the duration. If the worker only has one
  thread, no other jobs can run. The thread-pool size is
  `Settings.job_executor_max_workers` (default 4) — note that the
  executor does **not** currently cap worker concurrency, only
  executor concurrency; the actual worker thread is one per
  `WorkerService` process.
- **Reading `job.state` from the worker after a `get_job` call
  and assuming the state is current.** The worker reads state
  inside the cancel-poll branches; it does not cache state
  across stages. If a stage's logic needs to react to a state
  change, re-read the job with `job_manager.get_job(job_id)`.
- **Adding a new field to `Job` without updating `IndexingJobRow`
  and `_row_to_job`.** The Postgres round-trip is lossy both
  ways. A new field is invisible to HTTP clients until
  `Job.to_dict()` is updated; a new field is invisible to the
  worker until `_row_to_job` is updated.

---

## 15. Architectural boundaries

Four boundaries keep the jobs package composable.

### 1. `jobs` does not own the indexing pipeline

`indexing_worker` calls into `ingestion` (Scanner, parser registry,
IRBuilder, chunk_project, embed_chunks) and `storage` (manifest
store, IR repository, pgvector store). The jobs package owns the
lifecycle (state, progress, cancellation, retry) and the pipeline
driver (the worker function and its session), but the actual
indexing primitives belong to ingestion. If a new pipeline stage
is added, the addition is in `ingestion/` and the worker simply
calls it.

### 2. `jobs` does not own the HTTP surface

Routers in `api/routers/projects.py` call `JobManager` methods.
The manager is the only place the package's API is exposed; the
router does not import from `store.py` or `worker.py`. This is
what lets the manager's contract (de-duplication, cancel guards,
retry semantics) be tested independently of the wire format.

### 3. `jobs` does not own the worker process

The polling worker is `WorkerService` in `src/semrag/worker.py`,
not in `src/semrag/jobs/`. The split is intentional:

- `jobs/worker.py` is the indexing pipeline driver — the function
  a worker calls to run one job.
- `src/semrag/worker.py` is the worker *process* — the polling
  loop, the in-flight guard, the signal handling, and the
  process-level startup (`build_worker_service`).

A future push-based worker would live alongside `WorkerService`
and call into `jobs/worker.py:indexing_worker` the same way.

### 4. `jobs` does not own process-lifecycle singletons

The process-wide `JobManager`, `JobStore`, and `JobExecutor` are
constructed in `api/dependencies.py` (or `src/semrag/worker.py`
for the worker process). The jobs package defines the *types* and
the *factories*; the *instances* live where the process's
lifecycle owns them. This is what lets the worker process use a
real `PostgresJobStore` while a test uses the in-memory variant
without changing the package's code.

---

## 16. Related documentation

- [src/semrag/README.md](../README.md) — package map and the
  dependency graph showing `jobs` as the API's way of running
  services asynchronously.
- [src/semrag/services/README.md](../services/README.md) — the
  `IngestionService.index(...)` and `QueryService.ask(...)` that
  the worker composes. The worker calls into the same lower-level
  packages; it does not call `IngestionService` directly (its
  pipeline is inlined for finer-grained progress reporting).
- [src/semrag/api/README.md](../api/README.md) — the routers
  (`/v1/projects/{id}/index`, `/jobs/{id}` DELETE) and the
  dependency injection (`get_job_manager`, `get_job_executor`).
- [developer_guide.md](../../../docs/developer_guide.md) — end-to-end
  indexing lifecycle (§4), background jobs (§7), debugging
  workflow (§14), and the architectural rules that constrain
  where new code can go.
- [CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md) — file-by-file
  ownership and "where do I change X?" tables.
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) — long-form architecture
  with data-flow diagrams.
- [src/semrag/worker.py](../worker.py) — the `WorkerService`
  polling loop that calls `indexing_worker(...)`.
- [docs/operations/job-backlog.md](../../../docs/operations/job-backlog.md)
  and
  [docs/operations/indexing-failure.md](../../../docs/operations/indexing-failure.md)
  — on-call runbooks keyed to the worker behavior documented
  here.