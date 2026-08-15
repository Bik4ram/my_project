"""Tests for the OllamaEmbeddingProvider's context-truncation logic.

The provider must not fail an entire indexing job when ollama rejects a
single over-long prompt with ``500 "the input length exceeds the context
length"`` — it should retry with a progressively shorter prefix instead.
HTTP is mocked; no live server is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from semrag.ingestion.embedding.ollama_provider import (
    EmbeddingProviderError,
    OllamaEmbeddingProvider,
)

CONTEXT_ERROR = '{"error":"the input length exceeds the context length"}'


def _mock_ok(embedding: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"embedding": [0.1, 0.2, 0.3]}'
    resp.json.return_value = {"embedding": embedding}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_context_error() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 500
    resp.text = CONTEXT_ERROR
    resp.raise_for_status.side_effect = requests.HTTPError("500 error")
    return resp


def _provider(**kwargs):
    defaults = {"base_url": "http://localhost:11434", "model": "nomic-embed-text", "dimension": 3}
    defaults.update(kwargs)
    return OllamaEmbeddingProvider(**defaults)


def test_successful_embedding_request():
    with patch("requests.post", return_value=_mock_ok([0.1, 0.2, 0.3])) as mock_post:
        result = _provider().embed_texts(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert mock_post.call_args.kwargs["json"] == {"model": "nomic-embed-text", "prompt": "hello"}


def test_context_exceeded_truncates_and_retries():
    provider = _provider()
    long_text = "x" * 4096
    expected_truncated = long_text[: len(long_text) // 2]

    with patch(
        "requests.post",
        side_effect=[_mock_context_error(), _mock_ok([0.1, 0.2, 0.3])],
    ) as mock_post:
        result = provider.embed_texts([long_text])

    assert result == [[0.1, 0.2, 0.3]]
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[1].kwargs["json"]["prompt"] == expected_truncated


def test_context_exceeded_at_floor_raises():
    provider = _provider(min_input_chars=128, max_retries=1)

    with patch("requests.post", return_value=_mock_context_error()) as mock_post:
        with pytest.raises(EmbeddingProviderError, match="failed after"):
            provider.embed_texts(["tiny"])

    assert mock_post.call_count == 1


def test_transient_error_retried_then_succeeds():
    provider = _provider(max_retries=3)

    with patch(
        "requests.post",
        side_effect=[requests.ConnectionError("refused"), _mock_ok([0.1, 0.2, 0.3])],
    ) as mock_post:
        with patch("time.sleep"):
            result = provider.embed_texts(["hello"])

    assert result == [[0.1, 0.2, 0.3]]
    assert mock_post.call_count == 2


def test_dimension_mismatch_raises_clear_error():
    provider = _provider(dimension=768)

    with patch("requests.post", return_value=_mock_ok([0.1, 0.2])) as mock_post:
        with pytest.raises(EmbeddingProviderError, match="dimension mismatch"):
            provider.embed_texts(["hello"])

    assert mock_post.call_count == 1


def test_empty_input_returns_empty_without_a_request():
    with patch("requests.post") as mock_post:
        result = _provider().embed_texts([])
    assert result == []
    mock_post.assert_not_called()