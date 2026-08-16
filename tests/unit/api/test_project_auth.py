"""
Tests for project authorization.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from semrag.api.auth import require_api_key
from semrag.api.api_key import APIKey, InMemoryAPIKeyStore
from semrag.api.project_auth import get_project_authorizer


def test_project_authorizer():
    """Test the project authorizer dependency."""
    app = FastAPI()

    # Override the API key store to return a known key
    store = InMemoryAPIKeyStore()
    # Key that is allowed for project "proj1" but not "proj2"
    allowed_key = APIKey(
        key="allowed-key",
        name="allowed",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        allowed_projects=["proj1"],
    )
    # Key that is allowed for all projects (empty allowed_projects)
    all_key = APIKey(
        key="all-key",
        name="all",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        allowed_projects=[],
    )
    # Key that is not allowed for any project (if we set allowed_projects to a list that doesn't include the project)
    restricted_key = APIKey(
        key="restricted-key",
        name="restricted",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        allowed_projects=["proj3"],
    )
    asyncio = __import__('asyncio')
    asyncio.run(store.create_key(allowed_key))
    asyncio.run(store.create_key(all_key))
    asyncio.run(store.create_key(restricted_key))

    # We need to override the get_api_key_store dependency to return our store
    from semrag.api.api_key import get_api_key_store
    import semrag.api.auth
    original_get_api_key_store = semrag.api.auth.get_api_key_store
    semrag.api.auth.get_api_key_store = lambda: store

    @app.get("/projects/{project_id}/test")
    async def test_route(
        project_id: str,
        api_key: APIKey = Depends(get_project_authorizer()),
    ):
        return {"project_id": project_id, "key_name": api_key.name}

    client = TestClient(app)

    # Test with a key that is allowed for the project
    response = client.get(
        "/projects/proj1/test",
        headers={"Authorization": "Bearer allowed-key"},
    )
    assert response.status_code == 200
    assert response.json() == {"project_id": "proj1", "key_name": "allowed"}

    # Test with a key that is allowed for all projects
    response = client.get(
        "/projects/proj2/test",
        headers={"Authorization": "Bearer all-key"},
    )
    assert response.status_code == 200
    assert response.json() == {"project_id": "proj2", "key_name": "all"}

    # Test with a key that is not allowed for the project
    response = client.get(
        "/projects/proj1/test",
        headers={"Authorization": "Bearer restricted-key"},
    )
    assert response.status_code == 403
    assert "not authorized" in response.json()["detail"]

    # Test with an invalid key
    response = client.get(
        "/projects/proj1/test",
        headers={"Authorization": "Bearer invalid-key"},
    )
    assert response.status_code == 401

    # Restore the original function
    semrag.api.auth.get_api_key_store = original_get_api_key_store