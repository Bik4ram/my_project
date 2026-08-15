"""
Tests the OpenAICompatibleEmbeddingProvider's own logic (retry/backoff,
out-of-order response handling, dimension validation, error surfacing)
against mocked HTTP responses — genuinely valuable and genuinely
different from Part 7's fully-skipped Neo4j tests, because here the
protocol (plain JSON over HTTP) is simple enough that controlling the
response precisely tests real branches in this module's own code.

What this does NOT verify: that a real Ollama/vLLM/llama.cpp server's
actual responses match the shape assumed here. That requires a live
server, which (same as Part 7) isn't reachable in this sandbox. The
`test_live_server_smoke_test` at the bottom of this file is the one that
would catch a real-world contract mismatch, and it skips without one.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from semrag.ingestion.embedding.openai_compatible_provider import (
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)


def _mock_response(data: list[dict], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": data}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    return resp


def test_successful_embedding_request():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", dimension=3
    )
    mock_data = [
        {"embedding": [0.1, 0.2, 0.3], "index": 0},
        {"embedding": [0.4, 0.5, 0.6], "index": 1},
    ]
    with patch("requests.post", return_value=_mock_response(mock_data)) as mock_post:
        result = provider.embed_texts(["text one", "text two"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "nomic-embed-text"
    assert call_kwargs["json"]["input"] == ["text one", "text two"]


def test_out_of_order_response_is_resorted_by_index():
    provider = OpenAICompatibleEmbeddingProvider(base_url="http://localhost:11434", model="m", dimension=2)
    # Server returns index 1 before index 0 — client must not trust
    # positional order.
    mock_data = [
        {"embedding": [9.0, 9.0], "index": 1},
        {"embedding": [1.0, 1.0], "index": 0},
    ]
    with patch("requests.post", return_value=_mock_response(mock_data)):
        result = provider.embed_texts(["first", "second"])

    assert result == [[1.0, 1.0], [9.0, 9.0]]


def test_dimension_mismatch_raises_clear_error():
    provider = OpenAICompatibleEmbeddingProvider(base_url="http://localhost:11434", model="m", dimension=768)
    mock_data = [{"embedding": [0.1, 0.2], "index": 0}]  # only 2 dims, not 768
    with patch("requests.post", return_value=_mock_response(mock_data)):
        with pytest.raises(EmbeddingProviderError, match="dimension mismatch"):
            provider.embed_texts(["text"])


def test_retries_on_transient_failure_then_succeeds():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434", model="m", dimension=1, max_retries=3
    )
    success_response = _mock_response([{"embedding": [1.0], "index": 0}])

    with patch("requests.post", side_effect=[requests.ConnectionError("refused"), success_response]) as mock_post:
        with patch("time.sleep"):  # don't actually wait during tests
            result = provider.embed_texts(["text"])

    assert result == [[1.0]]
    assert mock_post.call_count == 2


def test_exhausting_retries_raises_provider_error():
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:11434", model="m", dimension=1, max_retries=2
    )
    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        with patch("time.sleep"):
            with pytest.raises(EmbeddingProviderError, match="Failed to reach embedding endpoint"):
                provider.embed_texts(["text"])


def test_empty_input_returns_empty_without_a_request():
    provider = OpenAICompatibleEmbeddingProvider(base_url="http://localhost:11434", model="m", dimension=1)
    with patch("requests.post") as mock_post:
        result = provider.embed_texts([])
    assert result == []
    mock_post.assert_not_called()


NEO4J_STYLE_LIVE_URL = os.environ.get("SEMRAG_TEST_EMBEDDING_SERVER_URL")


@pytest.mark.skipif(
    not NEO4J_STYLE_LIVE_URL,
    reason="No live OpenAI-compatible embedding server configured "
    "(set SEMRAG_TEST_EMBEDDING_SERVER_URL, e.g. to a local Ollama instance) "
    "— see this part's design doc, Section 1.",
)
def test_live_server_smoke_test():
    """The one test in this file that actually verifies real-world
    compatibility, not just this module's own logic. Skips gracefully
    without a configured server, same pattern as Postgres/Neo4j fixtures
    elsewhere in this series."""
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=NEO4J_STYLE_LIVE_URL, model=os.environ.get("SEMRAG_TEST_EMBEDDING_MODEL", "nomic-embed-text"),
        dimension=int(os.environ.get("SEMRAG_TEST_EMBEDDING_DIM", "768")),
    )
    result = provider.embed_texts(["a simple test sentence"])
    assert len(result) == 1
    assert len(result[0]) == provider.dimension
