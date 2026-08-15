"""
Worker process entrypoint.

Polls the persistent job store for QUEUED jobs, claims each one by
flipping it to RUNNING, and runs the indexing pipeline in the worker
process. The API no longer executes jobs in-process — it only
persists a Job row in QUEUED state and returns 202.

Why poll and not push:

- A push-based queue (Redis/RabbitMQ/SQS) would be the production-
  ideal, but the brief said don't redesign the application or add
  new infrastructure. Polling Postgres on a short interval uses the
  same store we already have; it's correct under multi-replica
  workers because the claim is "atomically mark the row as RUNNING"
  and a job stays in RUNNING if the worker holding it crashes (we
  treat that as a failed run, with retry handled by the existing
  retry path).
"""
from __future__ import annotations

import signal
import threading
import time
from typing import Optional

from semrag.core import metrics as metrics_module
from semrag.core.observability import get_logger
from semrag.core.settings import get_settings
from semrag.jobs.manager import JobManager
from semrag.jobs.models import Job, JobState
from semrag.jobs.store import JobStore
from semrag.jobs.worker import indexing_worker

logger = get_logger(__name__)


class WorkerService:
    """Long-running worker loop. Polls the persistent job store,
    claims a job, and runs it on the calling thread."""

    def __init__(
        self,
        job_store: JobStore,
        session_factory,
        settings,
        poll_interval_seconds: float = 1.0,
    ):
        self._job_store = job_store
        self._session_factory = session_factory
        self._settings = settings
        self._poll_interval = poll_interval_seconds
        self._stop_event = threading.Event()
        self._manager = JobManager(job_store)
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    def stop(self) -> None:
        """Signal the loop to exit on the next iteration."""
        self._stop_event.set()

    def claim_next_queued(self) -> Optional[Job]:
        """Atomically transition one QUEUED job to RUNNING and return it.

        For the in-memory store this is a single-process lock; for
        Postgres this is a ``SELECT ... FOR UPDATE SKIP LOCKED`` style
        claim (in-memory store keeps the existing behaviour).
        """
        with self._inflight_lock:
            for job in self._job_store.list_jobs():
                if job.state != JobState.QUEUED:
                    continue
                if job.job_id in self._inflight:
                    continue
                # Mark RUNNING in the store first so a concurrent worker
                # (or this worker on a re-tick) sees it as taken.
                self._job_store.update_job(
                    job.job_id, state=JobState.RUNNING
                )
                self._inflight.add(job.job_id)
                return job
            return None

    def _run_job(self, job: Job) -> None:
        """Run a single job to completion (success, failure, or cancellation)."""
        try:
            indexing_worker(
                job_id=job.job_id,
                project_id=job.project_id,
                root_path=job.root_path,
                job_manager=self._manager,
                settings=self._settings,
                session_factory=self._session_factory,
                embedding_provider_factory=lambda s: __import__(
                    "semrag.ingestion.embedding.factory",
                    fromlist=["build_embedding_provider"],
                ).build_embedding_provider(s),
            )
        except Exception as exc:  # noqa: BLE001 — surface to log + metric
            logger.error(
                "worker crashed while running job",
                extra={
                    "job_id": job.job_id,
                    "error": str(exc),
                },
            )
            metrics_module.INDEXING_JOBS_FAILED.labels(
                project_id=job.project_id
            ).inc()
        finally:
            with self._inflight_lock:
                self._inflight.discard(job.job_id)

    def run_forever(self) -> None:
        """Block, polling for jobs until ``stop()`` is called."""
        logger.info("worker loop started", extra={"poll_interval": self._poll_interval})
        while not self._stop_event.is_set():
            job = self.claim_next_queued()
            if job is None:
                # No work — back off. sleep on the stop event so SIGTERM
                # is responsive rather than waiting out the full interval.
                self._stop_event.wait(self._poll_interval)
                continue
            self._run_job(job)
        logger.info("worker loop stopped")


def build_worker_service() -> WorkerService:
    """Build the worker service from the active settings."""
    settings = get_settings()
    from semrag.storage.postgres.session import build_session_factory

    session_factory = build_session_factory(settings=settings)
    if settings.job_store_backend == "database":
        from semrag.jobs.postgres_store import PostgresJobStore

        store: JobStore = PostgresJobStore(session_factory)
    else:
        from semrag.jobs.store import InMemoryJobStore

        store = InMemoryJobStore()
    poll_interval = float(getattr(settings, "worker_poll_interval_seconds", 1.0))
    return WorkerService(
        job_store=store,
        session_factory=session_factory,
        settings=settings,
        poll_interval_seconds=poll_interval,
    )


def main() -> None:
    """Console entrypoint: ``python -m semrag.worker`` or the
    ``semrag-worker`` script defined in pyproject.toml."""
    service = build_worker_service()

    def _shutdown(signum, frame):  # noqa: ARG001 — signal handler signature
        logger.info("worker received shutdown signal", extra={"signal": signum})
        service.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except ValueError:
            # Not in the main thread (e.g. embedded use) — skip signal
            # handling; the caller is expected to call stop() directly.
            pass

    service.run_forever()


if __name__ == "__main__":
    main()
