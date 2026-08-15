"""
Job model and state definitions for asynchronous indexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any


def _isoformat(value: Optional[datetime | float]) -> Optional[str]:
    """Render a datetime or epoch float as an ISO 8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # epoch seconds (float)
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


class JobState(Enum):
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Job:
    """Represents an indexing job."""
    job_id: str
    project_id: str
    state: JobState = JobState.QUEUED
    progress: int = 0  # 0-100
    current_stage: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None  # in seconds
    error: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    # Input parameters
    root_path: str = ""
    # Retry mechanism
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "state": self.state.name.lower(),
            "progress": self.progress,
            "current_stage": self.current_stage,
            "started_at": _isoformat(self.started_at),
            "completed_at": _isoformat(self.completed_at),
            "duration": self.duration,
            "error": self.error,
            "statistics": self.statistics,
            "retry_count": self.retry_count,
        }