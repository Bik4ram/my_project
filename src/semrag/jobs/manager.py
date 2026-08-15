"""
Job management layer.

Wave 3: every freshly-queued job increments the
``semrag_indexing_jobs_queued_total{project_id}`` Prometheus
counter so queue depth and ingestion rate are observable from
Prometheus without scraping application logs.
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

from semrag.core import metrics as metrics_module
from semrag.jobs.models import Job, JobState
from semrag.jobs.store import JobStore


class JobManager:
    """Manages job lifecycle."""

    def __init__(self, store: JobStore):
        self._store = store

    def create_job(self, project_id: str, root_path: str) -> Tuple[Job, bool]:
        """Create a new queued indexing job, or return existing running/queued job for the same project.
        Returns a tuple (job, created) where created is True if a new job was created.
        """
        # Check for existing job for the same project that is queued or running
        existing_job = self._get_existing_job(project_id)
        if existing_job is not None:
            return existing_job, False

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            project_id=project_id,
            root_path=root_path,
            state=JobState.QUEUED,
        )
        self._store.create_job(job)
        metrics_module.INDEXING_JOBS_QUEUED.labels(project_id=project_id).inc()
        return job, True

    def _get_existing_job(self, project_id: str) -> Optional[Job]:
        """Return an existing queued or running job for the project, if any."""
        jobs = self._store.list_jobs(project_id=project_id)
        for job in jobs:
            if job.state in (JobState.QUEUED, JobState.RUNNING):
                return job
        return None

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        return self._store.get_job(job_id)

    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update a job's fields."""
        return self._store.update_job(job_id, **kwargs)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        return self._store.delete_job(job_id)

    def list_jobs(self, project_id: Optional[str] = None) -> list[Job]:
        """List jobs, optionally filtered by project_id."""
        return self._store.list_jobs(project_id)

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a job. Returns True if job exists and is cancellable."""
        job = self.get_job(job_id)
        if job is None:
            return False
        # Only allow cancellation of queued or running jobs
        if job.state in (JobState.QUEUED, JobState.RUNNING):
            return self.update_job(job_id, state=JobState.CANCELLED)
        return False

    def retry_job(self, job_id: str, max_retries: int) -> bool:
        """
        Retry a failed (or previously retried) job if retry count is less
        than max_retries. Returns True if the job was re-enqueued, False
        otherwise.
        """
        job = self.get_job(job_id)
        if job is None:
            return False
        if job.state not in (JobState.FAILED, JobState.QUEUED):
            return False
        if job.retry_count >= max_retries:
            return False
        # Increment retry count and reset state to QUEUED
        job.retry_count += 1
        # Reset job state and fields for retry
        self.update_job(
            job_id,
            state=JobState.QUEUED,
            progress=0,
            current_stage="",
            started_at=None,
            completed_at=None,
            duration=None,
            error=None,
        )
        # Note: we do not reset statistics? We might want to keep the statistics from the failed run?
        # For simplicity, we'll leave statistics as is (they will be overwritten by the new run).
        return True