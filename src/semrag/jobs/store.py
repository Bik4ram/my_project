"""
Job storage abstraction.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional

from semrag.jobs.models import Job, JobState


class JobStore(ABC):
    """Abstract base class for job storage."""

    @abstractmethod
    def create_job(self, job: Job) -> None:
        """Create a new job."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID."""
        pass

    @abstractmethod
    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update a job's fields. Returns True if job exists."""
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        """Delete a job. Returns True if job existed."""
        pass

    @abstractmethod
    def list_jobs(self, project_id: Optional[str] = None) -> list[Job]:
        """List jobs, optionally filtered by project_id."""
        pass


class InMemoryJobStore(JobStore):
    """Simple in-memory job store for development and testing."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return True

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def list_jobs(self, project_id: Optional[str] = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
            if project_id is not None:
                jobs = [j for j in jobs if j.project_id == project_id]
            return jobs