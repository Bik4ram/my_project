"""A2 — grounded coding agent endpoints.

POST /v1/projects/{project_id}/code runs the bounded, deterministic
coding workflow (intent gate → retrieve → read → plan → modify →
validate → bounded repair → verify → index freshness → A1-grounded
summary) and returns the structured result.

The endpoint is deliberately the ONLY new surface: it composes the
existing service (retriever, validation runner, job manager) behind the
``CodingAgent`` state machine. No new retrieval, no second validation
runner, no concurrency.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.api.auth import APIKey
from semrag.api.dependencies import get_db_session, get_job_manager, get_query_service
from semrag.api.project_auth import get_project_authorizer
from semrag.api.rate_limit import enforce_rate_limit
from semrag.api.schemas import (
    CodeChangeSummary,
    CodeRequest,
    CodeResponse,
    CodeValidationSummary,
    GroundingSummary,
)
from semrag.core.settings import Settings, get_settings
from semrag.reasoning.coding import CodingAgent, CodingRequest
from semrag.services.services import QueryService
from semrag.storage.postgres.models import Project

router = APIRouter(prefix="/v1/projects", tags=["code"])


def map_to_code_response(result, project_id: str, request_id: str) -> CodeResponse:
    """Map a CodingResult to the API response schema."""
    grounding = None
    if result.grounding:
        grounding = GroundingSummary(
            status=result.grounding.get("status", "grounded"),
            unsupported_claims=result.grounding.get("unsupported_claims", []),
            revision_attempted=bool(result.grounding.get("revision_attempted")),
            revised=bool(result.grounding.get("revised")),
        )
    return CodeResponse(
        project_id=project_id,
        intent=result.intent.value if result.intent else None,
        status=result.final_status,
        final_answer=result.final_answer,
        grounding=grounding,
        changes=[
            CodeChangeSummary(
                op=c.op.value,
                file_path=c.file_path,
                applied=c.applied,
                verified=c.verified,
                error=c.error,
            )
            for c in result.changes
        ],
        validation=[
            CodeValidationSummary(
                target=v.target,
                status=v.status,
                exit_code=v.exit_code,
                command=v.command,
                blocked_reason=v.blocked_reason,
                duration_seconds=v.duration_seconds,
            )
            for v in result.validation_attempts
        ],
        repair_attempts=result.repair_attempts,
        files_inspected=result.files_inspected,
        files_modified=result.files_modified,
        validation_target=result.validation_target,
        needs_clarification=result.needs_clarification,
        error=result.error,
        request_id=request_id,
    )


@router.post("/{project_id}/code", response_model=CodeResponse)
def run_coding_agent(
    project_id: str,
    request: CodeRequest,
    request_id: str = Depends(lambda: str(uuid.uuid4())),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db_session),
    service: QueryService = Depends(get_query_service),
    job_manager: Any = Depends(get_job_manager),
) -> CodeResponse:
    """Run the grounded coding agent workflow for a project."""
    if not settings.coding_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Coding agent is disabled (coding_enabled=false).",
        )

    project_row = db.execute(
        select(Project).where(Project.project_id == project_id)
    ).scalar_one_or_none()
    if project_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        )

    toolset = service.build_coding_toolset(
        project_id,
        project_row.root_path,
        job_manager,
    )
    agent = CodingAgent(service._llm, toolset, settings, project_id=project_id)
    coding_request = CodingRequest(
        project_id=project_id,
        request=request.request,
        max_repair_attempts=request.max_repair_attempts,
        validation_target=request.validation_target,
    )
    try:
        result = agent.run(coding_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run coding agent: {e}",
        ) from e

    return map_to_code_response(result, project_id, request_id)