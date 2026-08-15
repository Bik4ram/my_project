"""
Unit tests for JobManager.
"""
from __future__ import annotations

import time

import pytest

from semrag.jobs.models import Job, JobState
from semrag.jobs.manager import JobManager
from semrag.jobs.store import InMemoryJobStore


def test_job_manager_create_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, created = manager.create_job(project_id="test-project", root_path="/tmp")
    assert job.job_id is not None
    assert created is True  # Should be a new job
    assert job.project_id == "test-project"
    assert job.state == JobState.QUEUED
    assert store.get_job(job.job_id) == job

    # Call again for the same project; should return the existing job
    job2, created2 = manager.create_job(project_id="test-project", root_path="/tmp")
    assert created2 is False  # Should not be a new job
    assert job2.job_id == job.job_id  # Same job
    assert job2.project_id == "test-project"
    assert job2.state == JobState.QUEUED


def test_job_manager_get_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    fetched = manager.get_job(job.job_id)
    assert fetched == job
    assert manager.get_job("non-existent") is None


def test_job_manager_update_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    assert manager.update_job(job.job_id, state=JobState.RUNNING, progress=50)
    updated = store.get_job(job.job_id)
    assert updated.state == JobState.RUNNING
    assert updated.progress == 50
    assert not manager.update_job("non-existent", state=JobState.RUNNING)


def test_job_manager_delete_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    assert manager.delete_job(job.job_id)
    assert store.get_job(job.job_id) is None
    assert not manager.delete_job(job.job_id)  # already deleted


def test_job_manager_cancel_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    # Queued job can be cancelled
    assert manager.cancel_job(job.job_id)
    assert store.get_job(job.job_id).state == JobState.CANCELLED
    # Run the job to running state
    job2, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    store.update_job(job2.job_id, state=JobState.RUNNING)
    assert manager.cancel_job(job2.job_id)
    assert store.get_job(job2.job_id).state == JobState.CANCELLED
    # Completed job cannot be cancelled
    job3, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    store.update_job(job3.job_id, state=JobState.COMPLETED)
    assert not manager.cancel_job(job3.job_id)
    assert store.get_job(job3.job_id).state == JobState.COMPLETED


def test_job_manager_retry_job():
    store = InMemoryJobStore()
    manager = JobManager(store)
    job, _ = manager.create_job(project_id="test-project", root_path="/tmp")
    # Fail the job
    store.update_job(job.job_id, state=JobState.FAILED, error="test error")
    # Retry with max_retries=1 should succeed
    assert manager.retry_job(job.job_id, max_retries=1)
    retry_job = store.get_job(job.job_id)
    assert retry_job.state == JobState.QUEUED
    assert retry_job.retry_count == 1
    # Retry again with max_retries=1 should fail (already reached max)
    assert not manager.retry_job(job.job_id, max_retries=1)
    # Retry with max_retries=2 should succeed
    assert manager.retry_job(job.job_id, max_retries=2)
    retry_job2 = store.get_job(job.job_id)
    assert retry_job2.retry_count == 2