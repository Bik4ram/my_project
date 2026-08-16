"""
Job execution abstraction and factory.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from semrag.core.settings import Settings
from semrag.jobs.models import Job
from semrag.jobs.store import JobStore


class JobExecutor(ABC):
    """Abstract base class for job execution."""

    @abstractmethod
    def submit(
        self,
        job_id: str,
        fn: Callable[[], None],
        on_complete: Optional[Callable[[Future], None]] = None,
    ) -> None:
        """Submit a job for execution."""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        pass


class ThreadPoolJobExecutor(JobExecutor):
    """Executes jobs using a thread pool."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(
        self,
        job_id: str,
        fn: Callable[[], None],
        on_complete: Optional[Callable[[Future], None]] = None,
    ) -> None:
        """Submit a job for execution."""
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")

        def wrapper():
            try:
                fn()
            finally:
                if on_complete:
                    on_complete()

        self._executor.submit(wrapper)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)


class ExecutorFactory:
    """Factory for creating job executors based on settings."""

    @staticmethod
    def create_executor(settings: Settings) -> JobExecutor:
        """Create a job executor based on the settings.

        Currently only supports threadpool executor.
        Future implementations can support other backends like Celery, RQ, etc.
        """
        # For now, we only have threadpool executor.
        # We can read the executor type and settings from the settings object.
        # We'll assume settings has an attribute `job_executor_type` and `job_executor_max_workers`.
        # If not, we'll use defaults.
        executor_type = getattr(settings, 'job_executor_type', 'threadpool')
        max_workers = getattr(settings, 'job_executor_max_workers', 4)

        if executor_type == 'threadpool':
            return ThreadPoolJobExecutor(max_workers=max_workers)
        else:
            # Default to threadpool
            return ThreadPoolJobExecutor(max_workers=max_workers)