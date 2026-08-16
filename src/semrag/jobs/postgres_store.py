"""
Postgres-backed job store.

Persist indexing jobs across replicas and process restarts. The Job
dataclass is the in-memory representation; the SQLAlchemy row model
below mirrors each field, with the free-form statistics dict living
in a JSONB column because it carries counters, timing, and
throughput values that all change shape with the indexing pipeline.

Note: per-job concurrency still lives in the JobManager (single
worker per job, never two). The store only needs to keep the
state durable; the application enforces the "one runner per job"
invariant by checking the state before submitting on each
endpoint hit.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from semrag.jobs.models import Job, JobState
from semrag.jobs.store import JobStore


class _JobBase(DeclarativeBase):
    pass


class IndexingJobRow(_JobBase):
    __tablename__ = "indexing_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    started_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=False))
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=False))
    duration: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(String)
    statistics: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
    )
    root_path: Mapped[str] = mapped_column(String, nullable=False, default="")
    rtl_root: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tb_root: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    docs_root: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # created_at is the queueing time; useful for oldest-job queries and
    # feels orthogonal to started_at (which records when a worker picked
    # the job up — a QUEUED job can have started_at = NULL).
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=dt.datetime.utcnow
    )


def _row_to_job(row: IndexingJobRow) -> Job:
    return Job(
        job_id=row.job_id,
        project_id=row.project_id,
        state=JobState[row.state.upper()],
        progress=row.progress,
        current_stage=row.current_stage or "",
        started_at=row.started_at,
        completed_at=row.completed_at,
        duration=row.duration,
        error=row.error,
        statistics=dict(row.statistics or {}),
        root_path=row.root_path or "",
        rtl_root=row.rtl_root,
        tb_root=row.tb_root,
        docs_root=row.docs_root,
        retry_count=row.retry_count,
    )


def _coerce_datetime(value):
    """Coerce an epoch-seconds float (as the job worker writes) into a
    naive UTC ``datetime`` so it can be stored in the DateTime columns.
    ``Job`` tolerates both forms (see ``_isoformat`` in models.py), but
    Postgres' timestamp column does not."""
    if value is None or isinstance(value, dt.datetime):
        return value
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).replace(tzinfo=None)
    return value


class PostgresJobStore(JobStore):
    """Persistent job store backed by Postgres."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def create_job(self, job: Job) -> None:
        with self._session_factory() as session:
            existing = session.execute(
                select(IndexingJobRow).where(IndexingJobRow.job_id == job.job_id)
            ).scalar_one_or_none()
            if existing is not None:
                # JobManager already prevents re-enqueueing the same job_id,
                # but a defensive no-op here keeps the store honest if
                # something else reaches this path.
                return
            session.add(
                IndexingJobRow(
                    job_id=job.job_id,
                    project_id=job.project_id,
                    state=job.state.name,
                    progress=job.progress,
                    current_stage=job.current_stage,
                    started_at=_coerce_datetime(job.started_at),
                    completed_at=_coerce_datetime(job.completed_at),
                    duration=job.duration,
                    error=job.error,
                    statistics=dict(job.statistics or {}),
                    root_path=job.root_path,
                    rtl_root=job.rtl_root,
                    tb_root=job.tb_root,
                    docs_root=job.docs_root,
                    retry_count=job.retry_count,
                )
            )
            session.commit()

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._session_factory() as session:
            row = session.execute(
                select(IndexingJobRow).where(IndexingJobRow.job_id == job_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return _row_to_job(row)

    def update_job(self, job_id: str, **kwargs) -> bool:
        with self._session_factory() as session:
            row = session.execute(
                select(IndexingJobRow).where(IndexingJobRow.job_id == job_id)
            ).scalar_one_or_none()
            if row is None:
                return False
            for key, value in kwargs.items():
                if key == "state" and isinstance(value, JobState):
                    value = value.name
                if key in ("started_at", "completed_at"):
                    value = _coerce_datetime(value)
                if hasattr(row, key):
                    setattr(row, key, value)
            session.commit()
            return True

    def delete_job(self, job_id: str) -> bool:
        with self._session_factory() as session:
            row = session.execute(
                select(IndexingJobRow).where(IndexingJobRow.job_id == job_id)
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list_jobs(self, project_id: Optional[str] = None) -> list[Job]:
        with self._session_factory() as session:
            stmt = select(IndexingJobRow)
            if project_id is not None:
                stmt = stmt.where(IndexingJobRow.project_id == project_id)
            rows = session.execute(stmt).scalars().all()
            return [_row_to_job(r) for r in rows]
