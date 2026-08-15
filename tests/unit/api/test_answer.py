"""
Tests for the answer endpoint.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sse_starlette.sse import EventSourceResponse

from semrag.api.schemas import AnswerResponse, CitationSummary
from semrag.services.services import DetailedQueryAnswer
from semrag.ingestion.chunking.models import Chunk


def test_answer_endpoint_success(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer endpoint returns a successful response."""
    # Mock the DetailedQueryAnswer that the service would return
    mock_detailed_answer = DetailedQueryAnswer(
        final_answer="This is the answer.",
        iterations_used=2,
        hit_iteration_limit=False,
        considered_chunks_count=10,
        messages=[{"role": "assistant", "content": "Hello"}],
        debug=None,
        included_chunks=[
            Chunk(
                chunk_id="chunk1",
                project_id="proj1",
                relative_path="path/to/file.txt",
                chunk_type="text",
                text="This is a chunk.",
                source_node_id="node1",
                line_start=1,
                line_end=10,
            )
        ],
        agent_messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {
                            "name": "search_codebase",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            }
        ],
        citation_analysis={
            "valid_count": 1,
            "fabricated_count": 0,
            "fabrication_rate": 0.0,
            "cited_paths": ["path/to/file.txt"],
        },
    )

    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.return_value = mock_detailed_answer
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/proj1/answer",
            json={"query": "What is the answer?"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == "This is the answer."
    assert data["iterations"] == 2
    assert data["tools_used"] == ["search_codebase"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "chunk1"
    assert data["citations"]["valid_count"] == 1
    assert data["metadata"]["request_id"] is not None


def test_answer_endpoint_includes_grounding_summary(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """A1: the grounding gate summary is additive on the response."""
    mock_detailed_answer = DetailedQueryAnswer(
        final_answer="I couldn't verify enough of this explanation against the "
        "indexed source. The retrieved evidence does not support: i2c_master_test.",
        iterations_used=2,
        hit_iteration_limit=False,
        considered_chunks_count=10,
        messages=[{"role": "assistant", "content": "Hello"}],
        debug=None,
        included_chunks=[],
        agent_messages=[],
        citation_analysis={
            "valid_count": 0,
            "fabricated_count": 0,
            "fabrication_rate": 0.0,
            "cited_paths": [],
        },
        grounding={
            "status": "unsupported",
            "unsupported_claims": ["i2c_master_test starts the read sequence."],
            "revision_attempted": True,
            "revised": False,
        },
    )

    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.return_value = mock_detailed_answer
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/proj1/answer",
            json={"query": "What starts the read sequence?"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["grounding"]["status"] == "unsupported"
    assert data["grounding"]["unsupported_claims"] == [
        "i2c_master_test starts the read sequence."
    ]
    assert data["grounding"]["revision_attempted"] is True
    assert data["grounding"]["revised"] is False


def test_answer_endpoint_streaming_success(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer streaming endpoint returns events."""
    mock_detailed_answer = DetailedQueryAnswer(
        final_answer="Streamed answer.",
        iterations_used=1,
        hit_iteration_limit=False,
        considered_chunks_count=5,
        messages=[],
        debug=None,
        included_chunks=[],
        agent_messages=[],
        citation_analysis=None,
    )

    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.return_value = mock_detailed_answer
        mock_service_class.return_value = mock_service

        with api_client.stream(
            "POST",
            "/v1/projects/proj1/answer/stream",
            json={"query": "test"},
            headers=auth_headers,
        ) as response:
            assert response.status_code == status.HTTP_200_OK
            events = []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    events.append((event_type, data))

            # We expect at least agent_start, agent_end, and answer events
            event_types = [e[0] for e in events]
            assert "agent_start" in event_types
            assert "agent_end" in event_types
            assert "answer" in event_types

            # Find the answer event
            answer_event = next((data for event_type, data in events if event_type == "answer"), None)
            assert answer_event is not None
            assert answer_event["answer"] == "Streamed answer."
            assert answer_event["metadata"]["request_id"] is not None


def test_answer_endpoint_invalid_project(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer endpoint handles invalid projects."""
    # We'll mock the service to raise an exception that indicates an invalid project
    # For simplicity, we'll assume the service raises a ValueError for invalid project
    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.side_effect = ValueError("Invalid project")
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/invalid/answer",
            json={"query": "test"},
            headers=auth_headers,
        )

    # We expect a 400 or 500 error; the exact status depends on how we handle it
    # In our current implementation, we catch all exceptions and return 500
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_answer_endpoint_timeout(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer endpoint handles timeouts."""
    # We'll mock the service to raise a TimeoutError
    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.side_effect = TimeoutError("Request timed out")
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/proj1/answer",
            json={"query": "test"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_answer_endpoint_tool_failure(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer endpoint handles tool failures."""
    # We'll mock the service to raise an exception during tool execution
    # This would be caught in the service layer and turned into a 500
    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.side_effect = RuntimeError("Tool execution failed")
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/proj1/answer",
            json={"query": "test"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_answer_endpoint_empty_retrieval(
    api_client: TestClient,
    auth_headers: dict,
) -> None:
    """Test that the answer endpoint handles empty retrieval."""
    mock_detailed_answer = DetailedQueryAnswer(
        final_answer="No information found.",
        iterations_used=0,
        hit_iteration_limit=False,
        considered_chunks_count=0,
        messages=[],
        debug=None,
        included_chunks=[],
        agent_messages=[],
        citation_analysis=None,
    )

    with patch("semrag.api.routers.answer.QueryService") as mock_service_class:
        mock_service = Mock()
        mock_service.ask.return_value = mock_detailed_answer
        mock_service_class.return_value = mock_service

        response = api_client.post(
            "/v1/projects/proj1/answer",
            json={"query": "obscure query"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == "No information found."
    assert data["iterations"] == 0
    assert len(data["sources"]) == 0
    assert data["retrieval"]["considered_chunks_count"] == 0
    assert data["retrieval"]["included_chunks_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__])