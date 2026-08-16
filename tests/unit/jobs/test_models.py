"""
Unit tests for job models and store.
"""
from __future__ import annotations

import time
from uuid import UUID

import pytest

from semrag.jobs.models import Job, JobState
from semrag.jobs.store import InMemoryJobStore


def test_job_creation():
    job = Job(
        job_id="test-id",
        project_id="test-project",
        root_path="/tmp",
    )
    assert job.job_id == "test-id"
    assert job.project_id == "test-project"
    assert job.state == JobState.QUEUED
    assert job.progress == 0
    assert job.current_stage == ""
    assert job.started_at is None
    assert job.completed_at is None
    assert job.duration is None
    assert job.error is None
    assert job.statistics == {}


def test_job_to_dict():
    job = Job(
        job_id="test-id",
        project_id="test-project",
        state=JobState.RUNNING,
        progress=50,
        current_stage="Scanning",
    )
    job.started_at = time.time()
    job.completed_at = job.started_at + 10
    job.duration = 10.0
    job.error = None
    job.statistics = {"file_count": 10}

    data = job.to_dict()
    assert data["job_id"] == "test-id"
    assert data["project_id"] == "test-project"
    assert data["state"] == "running"
    assert data["progress"] == 50
    assert data["current_stage"] == "Scanning"
    assert data["started_at"] is not None
    assert data["completed_at"] is not None
    assert data["duration"] == 10.0
    assert data["error"] is None
    assert data["statistics"] == {"file_count": 10}


def test_in_memory_job_store():
    store = InMemoryJobStore()
    job = Job(job_id="1", project_id="p1", root_path="/tmp")

    # Create job
    store.create_job(job)
    assert store.get_job("1") == job

    # Get non-existent job
    assert store.get_job("2") is None

    # Update job
    store.update_job("1", state=JobState.RUNNING, progress=50)
    updated = store.get_job("1")
    assert updated.state == JobState.RUNNING
    assert updated.progress == 50

    # Delete job
    assert store.delete_job("1") is True
    assert store.get_job("1") is None
    assert store.delete_job("1") is False  # already deleted

    # List jobs
    job2 = Job(job_id="2", project_id="p1", root_path="/tmp")
    job3 = Job(job_id="3", project_id="p2", root_path="/tmp")
    store.create_job(job2)
    store.create_job(job3)
    assert len(store.list_jobs()) == 2
    assert len(store.list_jobs(project_id="p1")) == 1
    assert len(store.list_jobs(project_id="p2")) == 1
    assert len(store.list_jobs(project_id="p3")) == 0