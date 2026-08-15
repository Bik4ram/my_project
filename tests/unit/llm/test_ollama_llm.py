"""OllamaLLM native `/api/chat` client.

Covers the differences from OpenAICompatibleLLM:
- endpoint is `/api/chat` (not `/v1/chat/completions`)
- response is `{message, done, prompt_eval_count, eval_count}` (no
  `choices`/`usage` wrapper)
- streaming is NDJSON not SSE

Uses `requests.post` mocking, same pattern as `test_openai_compatible_llm.py`.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from semrag.reasoning.llm.ollama_llm import OllamaLLM


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    return resp


def test_provider_and_model_name():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    assert llm.provider_name == "ollama"
    assert llm.model_name == "qwen2.5:7b"


def test_complete_hits_api_chat_endpoint():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    payload = {
        "message": {"role": "assistant", "content": "hi back"},
        "done": True,
        "done_reason": "stop",
    }
    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.content == "hi back"
    assert result.finish_reason == "stop"
    # Endpoint must be /api/chat, not /v1/chat/completions.
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://localhost:11434/api/chat"
    assert mock_post.call_args.kwargs["json"]["stream"] is False


def test_complete_parses_tool_calls():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    payload = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search_codebase", "arguments": {"query": "axi"}},
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
    }
    with patch("requests.post", return_value=_mock_response(payload)):
        result = llm.complete([{"role": "user", "content": "search axi"}])

    assert result.finish_reason == "stop"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_codebase"
    assert result.tool_calls[0].arguments == {"query": "axi"}


def test_complete_captures_ollama_token_usage():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    payload = {
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 42,
        "eval_count": 17,
    }
    with patch("requests.post", return_value=_mock_response(payload)):
        result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.prompt_tokens == 42
    assert result.completion_tokens == 17
    assert result.total_tokens == 59


def test_complete_without_usage_leaves_token_fields_none():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    payload = {"message": {"role": "assistant", "content": "ok"}, "done": True}
    with patch("requests.post", return_value=_mock_response(payload)):
        result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None


def test_complete_raises_on_malformed_response():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    bad_resp = MagicMock()
    bad_resp.json.side_effect = ValueError("not json")
    bad_resp.raise_for_status = MagicMock()
    with patch("requests.post", return_value=bad_resp):
        from semrag.reasoning.llm._openai_compatible_impl import LLMProviderError
        with pytest.raises(LLMProviderError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_stream_yields_ndjson_lines():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:7b")
    # Build a fake response whose iter_lines yields our NDJSON lines.
    lines = [
        b'{"message": {"content": "hello "}, "done": false}',
        b'{"message": {"content": "world"}, "done": true, "done_reason": "stop"}',
    ]
    resp = MagicMock()
    resp.iter_lines.return_value = lines
    resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=resp):
        chunks = list(llm.stream([{"role": "user", "content": "hi"}]))

    assert chunks[0].delta_content == "hello "
    assert chunks[1].delta_content == "world"
    assert chunks[1].finish_reason == "stop"
    assert resp.raise_for_status.called