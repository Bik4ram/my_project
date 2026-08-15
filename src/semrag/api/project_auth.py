"""
Project authorization dependencies.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Path, Request, status

from semrag.api.auth import require_api_key
from semrag.api.api_key import APIKey


def get_project_authorizer():
    """Dependency factory that returns a dependency to check project authorization."""
    async def project_authorizer(request: Request, api_key = Depends(require_api_key)):
        project_id = request.path_params['project_id']
        if not api_key.is_allowed_for_project(project_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key not authorized for project '{project_id}'",
            )
        return api_key
    return project_authorizer