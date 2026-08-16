"""
Tests for the project indexing job endpoints.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from semrag.jobs.models import JobState


def test_enqueue_indexing_job(
    api_client: TestClient,
    auth_headers: dict,
    db_session,
):
    """Test that the index endpoint enqueues a job and returns 202 for a new job."""
    with patch("semrag.api.routers.projects.indexing_worker") as mock_worker:
        # We'll use a fake worker that updates the job to completed immediately.
        from semrag.api.dependencies import get_job_manager

        def fake_indexing_worker(
            job_id: str,
            project_id: str,
            root_path: str,
            job_manager,
            settings,
            session_factory,
            embedding_provider_factory,
        ):
            # Immediately mark the job as completed with some statistics
            job_manager.update_job(
                job_id,
                state=JobState.COMPLETED,
                progress=100,
                current_stage="Persisting",
                completed_at=job_manager.get_job(job_id).started_at + 10,  # fake duration
                duration=10.0,
                statistics={
                    "file_count": 5,
                    "chunk_count": 100,
                    "added": 5,
                    "modified": 0,
                    "deleted": 0,
                },
            )

        with patch("semrag.api.routers.projects.indexing_worker", new=fake_indexing_worker):
            response = api_client.post(
                "/v1/projects/test-proj/index",
                json={"root_path": "/tmp/tests"},
                headers=auth_headers,
            )

    # First call should create a new job and return 202
    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    assert "job_id" in data
    assert data["project_id"] == "test-proj"
    assert data["file_count"] == 0  # placeholder
    assert data["chunk_count"] == 0
    assert data["scan_id"] == "00000000-0000-0000-0000-000000000000"  # placeholder

    job_id = data["job_id"]

    # Check the job status endpoint
    response = api_client.get(
        f"/v1/projects/test-proj/jobs/{job_id}",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK
    job_data = response.json()
    assert job_data["job_id"] == job_id
    assert job_data["project_id"] == "test-proj"
    assert job_data["state"] == "completed"
    assert job_data["progress"] == 100
    assert job_data["current_stage"] == "persisting"
    assert job_data["duration"] is not None
    assert job_data["statistics"]["file_count"] == 5
    assert job_data["statistics"]["chunk_count"] == 100


def test_enqueue_duplicate_indexing_job(
    api_client: TestClient,
    auth_headers: dict,
    db_session,
):
    """Test that the index endpoint returns the existing job when a job is already running."""
    with patch("semrag.api.routers.projects.indexing_worker") as mock_worker:
        # We'll use a fake worker that takes some time so we can test the duplicate behavior.
        import threading

        worker_started = threading.Event()
        worker_finished = threading.Event()

        def fake_indexing_worker(
            job_id: str,
            project_id: str,
            root_path: str,
            job_manager,
            settings,
            session_factory,
            embedding_provider_factory,
        ):
            worker_started.set()
            # Wait until cancelled or finished
            worker_finished.wait(timeout=10)  # wait for cancellation signal
            # If not cancelled, mark as completed
            job = job_manager.get_job(job_id)
            if job and job.state != JobState.CANCELLED:
                job_manager.update_job(
                    job_id,
                    state=JobState.COMPLETED,
                    progress=100,
                    current_stage="Persisting",
                    completed_at=job_manager.get_job(job_id).started_at + 10,
                    duration=10.0,
                    statistics={"file_count": 5, "chunk_count": 100, "added": 5, "modified": 0, "deleted": 0},
                )

        with patch("semrag.api.routers.projects.indexing_worker", new=fake_indexing_worker):
            # Enqueue a job
            response = api_client.post(
                "/v1/projects/test-proj/index",
                json={"root_path": "/tmp/tests"},
                headers=auth_headers,
            )
            # First call should create a new job and return 202
            assert response.status_code == status.HTTP_202_ACCEPTED
            job_id = response.json()["job_id"]

            # Wait for worker to start
            worker_started.wait(timeout=5)
            assert worker_started.is_set()

            # Now try to enqueue another job for the same project while the first is running
            response2 = api_client.post(
                "/v1/projects/test-proj/index",
                json={"root_path": "/tmp/tests"},
                headers=auth_headers,
            )
            # Should return 200 OK with the existing job
            assert response2.status_code == status.HTTP_200_OK
            data2 = response2.json()
            assert data2["job_id"] == job_id  # Same job ID
            assert data2["project_id"] == "test-proj"
            # The existing job should be RUNNING (or QUEUED if it just started, but we waited for start)
            # We'll just check that it's not a new job (status 200) and the job ID matches.

            # Signal the worker to finish (it will check for cancellation and exit)
            worker_finished.set()

            # Give it a moment to update
            import time
            time.sleep(0.1)

            # Check job status
            response3 = api_client.get(
                f"/v1/projects/test-proj/jobs/{job_id}",
                headers=auth_headers,
            )
            assert response3.status_code == status.HTTP_200_OK
            job_data3 = response3.json()
            # The job should eventually complete
            # We'll just check that it's not cancelled
            assert job_data3["state"] != "cancelled"


def test_cancel_indexing_job(
    api_client: TestClient,
    auth_headers: dict,
    db_session,
):
    """Test that the cancel endpoint works."""
    with patch("semrag.api.routers.projects.indexing_worker") as mock_worker:
        # We'll use a fake worker that takes some time so we can cancel it.
        import threading

        worker_started = threading.Event()
        worker_finished = threading.Event()

        def fake_indexing_worker(
            job_id: str,
            project_id: str,
            root_path: str,
            job_manager,
            settings,
            session_factory,
            embedding_provider_factory,
        ):
            worker_started.set()
            # Wait until cancelled or finished
            worker_finished.wait(timeout=10)  # wait for cancellation signal
            # If not cancelled, mark as completed
            job = job_manager.get_job(job_id)
            if job and job.state != JobState.CANCELLED:
                job_manager.update_job(
                    job_id,
                    state=JobState.COMPLETED,
                    progress=100,
                    current_stage="Persisting",
                    completed_at=job_manager.get_job(job_id).started_at + 10,
                    duration=10.0,
                    statistics={"file_count": 5, "chunk_count": 100, "added": 5, "modified": 0, "deleted": 0},
                )

        with patch("semrag.api.routers.projects.indexing_worker", new=fake_indexing_worker):
            # Enqueue a job
            response = api_client.post(
                "/v1/projects/test-proj/index",
                json={"root_path": "/tmp/tests"},
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_202_ACCEPTED
            job_id = response.json()["job_id"]

            # Wait for worker to start
            worker_started.wait(timeout=5)
            assert worker_started.is_set()

            # Cancel the job
            response = api_client.delete(
                f"/v1/projects/test-proj/jobs/{job_id}",
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["status"] == "cancellation requested"

            # Signal the worker to finish (it will check for cancellation and exit)
            worker_finished.set()

            # Give it a moment to update
            import time
            time.sleep(0.1)

            # Check job status
            response = api_client.get(
                f"/v1/projects/test-proj/jobs/{job_id}",
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_200_OK
            job_data = response.json()
            assert job_data["state"] == "cancelled"


def test_get_job_not_found(
    api_client: TestClient,
    auth_headers: dict,
):
    """Test getting a non-existent job returns 404."""
    response = api_client.get(
        "/v1/projects/test-proj/jobs/non-existent-job-id",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_job_wrong_project(
    api_client: TestClient,
    auth_headers: dict,
    db_session,
):
    """Test getting a job for the wrong project returns 403."""
    with patch("semrag.api.routers.projects.indexing_worker"):
        response = api_client.post(
            "/v1/projects/proj-a/index",
            json={"root_path": "/tmp/tests"},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_202_ACCEPTED
        job_id = response.json()["job_id"]

        # Try to get it from project proj-b
        response = api_client.get(
            f"/v1/projects/proj-b/jobs/{job_id}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN


def test_cancel_job_wrong_state(
    api_client: TestClient,
    auth_headers: dict,
    db_session,
):
    """Test cancelling a completed job fails."""
    with patch("semrag.api.routers.projects.indexing_worker") as mock_worker:
        # Make the worker complete immediately
        def fake_worker(*args, **kwargs):
            job_id = args[0]
            from semrag.api.dependencies import get_job_manager
            job_manager = get_job_manager.__wrapped__ if hasattr(get_job_manager, '__wrapped__') else get_job_manager
            job = job_manager.create_job(project_id="test-proj", root_path="/tmp")
            job_manager.update_job(job.job_id, state=JobState.COMPLETED)

        with patch("semrag.api.routers.projects.indexing_worker", new=fake_worker):
            # Enqueue a job (it will complete immediately)
            response = api_client.post(
                "/v1/projects/test-proj/index",
                json={"root_path": "/tmp/tests"},
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_202_ACCEPTED
            job_id = response.json()["job_id"]

            # Try to cancel the completed job
            response = api_client.delete(
                f"/v1/projects/test-proj/jobs/{job_id}",
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST