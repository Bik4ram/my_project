"""Answer endpoints.

POST /v1/projects/{project_id}/answer returns the final answer to a question
using the full RAG pipeline.

POST /v1/projects/{project_id}/answer/stream returns a streaming response
with events for retrieval, agent progress, and the final answer.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from semrag.api.auth import require_api_key
from semrag.api.dependencies import get_db_session, get_query_service
from semrag.api.project_auth import get_project_authorizer
from semrag.api.rate_limit import enforce_rate_limit
from semrag.core.settings import get_settings
from semrag.services.services import QueryService, DetailedQueryAnswer
from semrag.api.schemas import AnswerResponse, GroundingSummary, QueryRequest, CitationSummary
from semrag.ingestion.chunking.models import Chunk
from semrag.retrieval.models import RetrievalResult


router = APIRouter(prefix="/v1/projects", tags=["answer"])


def map_to_answer_response(detailed_answer: DetailedQueryAnswer, request_id: str) -> AnswerResponse:
    """Map a DetailedQueryAnswer to an AnswerResponse."""
    # Build citations from citation_analysis
    citation_summary = None
    if detailed_answer.citation_analysis:
        citation_summary = CitationSummary(
            valid_count=detailed_answer.citation_analysis.get("valid_count", 0),
            fabricated_count=detailed_answer.citation_analysis.get("fabricated_count", 0),
            fabrication_rate=detailed_answer.citation_analysis.get("fabrication_rate", 0.0),
            cited_paths=detailed_answer.citation_analysis.get("cited_paths", []),
        )
    else:
        # If no citation analysis (e.g., no final answer), create an empty one
        citation_summary = CitationSummary(
            valid_count=0,
            fabricated_count=0,
            fabrication_rate=0.0,
            cited_paths=[],
        )

    # Convert included_chunks (list of Chunk) to sources (list of RetrievalResult)
    sources = []
    for chunk in detailed_answer.included_chunks or []:
        sources.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                project_id=chunk.project_id,
                relative_path=chunk.relative_path,
                chunk_type=chunk.chunk_type,
                text=chunk.text,
                source_node_id=chunk.source_node_id,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                fused_score=0.0,  # We don't have the fused score in Chunk; we can set to 0 or compute?
                signal_scores={},  # We don't have signal scores in Chunk
            )
        )

    # Extract tools used from agent_messages
    tools_used = set()
    for msg in detailed_answer.agent_messages or []:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                if isinstance(tool_call, dict) and "function" in tool_call:
                    func = tool_call["function"]
                    if isinstance(func, dict) and "name" in func:
                        tools_used.add(func["name"])
                # Alternatively, the tool call might be in a different format
                # We'll assume the above structure based on the AgentLoop's messages.
    tools_used = sorted(list(tools_used))

    # Retrieval stats
    retrieval = {
        "considered_chunks_count": detailed_answer.considered_chunks_count,
        "included_chunks_count": len(detailed_answer.included_chunks or []),
    }

    # Timings (we don't have timings in the DetailedQueryAnswer yet)
    timings = {}

    # Metadata
    metadata = {
        "embedding_provider": get_settings().embedding_provider,
        "embedding_model": get_settings().embedding_model,
        "llm_provider": get_settings().llm_provider,
        "llm_model": get_settings().llm_model,
        "request_id": request_id,
    }

    # A1 grounding gate result (additive; None when not produced)
    grounding = None
    if detailed_answer.grounding:
        grounding = GroundingSummary(
            status=detailed_answer.grounding.get("status", "grounded"),
            unsupported_claims=detailed_answer.grounding.get("unsupported_claims", []),
            revision_attempted=bool(detailed_answer.grounding.get("revision_attempted")),
            revised=bool(detailed_answer.grounding.get("revised")),
        )

    return AnswerResponse(
        answer=detailed_answer.final_answer or "",
        citations=citation_summary,
        sources=sources,
        iterations=detailed_answer.iterations_used,
        tools_used=tools_used,
        retrieval=retrieval,
        timings=timings,
        metadata=metadata,
        grounding=grounding,
    )


@router.post("/{project_id}/answer", response_model=AnswerResponse)
def answer_question(
    project_id: str,
    request: QueryRequest,
    request_id: str = Depends(lambda: str(uuid.uuid4())),
    api_key = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    settings: Settings = Depends(get_settings),
    service: QueryService = Depends(get_query_service),
) -> AnswerResponse:
    """Execute the full RAG pipeline and return the final answer with metadata."""
    try:
        detailed_answer = service.ask(project_id, request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute query: {e}",
        ) from e

    return map_to_answer_response(detailed_answer, request_id)


@router.post("/{project_id}/answer/stream")
async def answer_question_stream(
    project_id: str,
    request: QueryRequest,
    request_id: str = Depends(lambda: str(uuid.uuid4())),
    api_key = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    settings: Settings = Depends(get_settings),
    service: QueryService = Depends(get_query_service),
) -> EventSourceResponse:
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Emit an agent_start event to indicate the agent loop is about to start
            yield {"event": "agent_start", "data": json.dumps({})}
            # Execute the full RAG pipeline (this blocks until done)
            detailed_answer = service.ask(project_id, request.query)
            # Emit an agent_end event with iteration info
            yield {
                "event": "agent_end",
                "data": json.dumps(
                    {
                        "iterations_used": detailed_answer.iterations_used,
                        "hit_iteration_limit": detailed_answer.hit_iteration_limit,
                    }
                ),
            }
            # Map to AnswerResponse and yield as the final answer event
            answer_response = map_to_answer_response(detailed_answer, request_id)
            yield {"event": "answer", "data": answer_response.model_dump_json()}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"detail": f"Failed to execute query: {e}"})}

    return EventSourceResponse(event_generator())