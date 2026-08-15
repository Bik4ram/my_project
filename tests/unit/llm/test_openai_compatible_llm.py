from unittest.mock import MagicMock, patch

import pytest
import requests

from semrag.reasoning.llm.openai_compatible_llm import LLMProviderError, OpenAICompatibleLLM


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    return resp


def test_simple_completion():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="qwen2.5")
    payload = {"choices": [{"message": {"content": "hello there"}, "finish_reason": "stop"}]}
    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.content == "hello there"
    assert result.tool_calls == []
    assert mock_post.call_args.kwargs["json"]["stream"] is False


def test_configured_timeout_reaches_http_request():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="qwen2.5", timeout_seconds=180.0)
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        llm.complete([{"role": "user", "content": "hi"}])

    assert mock_post.call_args.kwargs["timeout"] == 180.0


def test_default_timeout_floor_is_safe_for_heavy_local_generation():
    # R4 regression: a 90 s cap made /answer fail reproducibly (HTTP 500,
    # LLM read timeout) on heavy multi-hop prompts with a local 7B model
    # (measured ~130 s end-to-end). The default must stay above that floor.
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="qwen2.5")
    assert llm._timeout >= 180.0


def test_completion_with_tool_calls():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="qwen2.5")
    payload = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search_codebase", "arguments": '{"query": "i2c"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }]
    }
    with patch("requests.post", return_value=_mock_response(payload)):
        result = llm.complete([{"role": "user", "content": "find i2c"}])

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_codebase"
    assert result.tool_calls[0].arguments == {"query": "i2c"}


def test_malformed_response_raises_clear_error():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="m")
    with patch("requests.post", return_value=_mock_response({"unexpected": "shape"})):
        with pytest.raises(LLMProviderError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_retries_then_succeeds():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="m", max_retries=3)
    success = _mock_response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    with patch("requests.post", side_effect=[requests.ConnectionError("refused"), success]):
        with patch("time.sleep"):
            result = llm.complete([{"role": "user", "content": "hi"}])
    assert result.content == "ok"


def _sse_lines(*events: str):
    for e in events:
        yield e.encode("utf-8")


def test_streaming_accumulates_content():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="m")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = _sse_lines(
        'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    )
    with patch("requests.post", return_value=mock_resp):
        chunks = list(llm.stream([{"role": "user", "content": "hi"}]))

    contents = [c.delta_content for c in chunks if c.delta_content]
    assert contents == ["Hello", " world"]
    assert chunks[-1].finish_reason == "stop"


def test_streaming_accumulates_tool_call_across_fragments():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="m")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = _sse_lines(
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", '
        '"function": {"name": "search_codebase", "arguments": ""}}]}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": "{\\"query\\""}}]}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, '
        '"function": {"arguments": ": \\"i2c\\"}"}}]}, "finish_reason": null}]}',
        'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
        "data: [DONE]",
    )
    with patch("requests.post", return_value=mock_resp):
        chunks = list(llm.stream([{"role": "user", "content": "hi"}]))

    final_chunk = chunks[-1]
    assert final_chunk.finish_reason == "tool_calls"
    assert len(final_chunk.delta_tool_calls) == 1
    assert final_chunk.delta_tool_calls[0].name == "search_codebase"
    assert final_chunk.delta_tool_calls[0].arguments == {"query": "i2c"}


def test_empty_lines_and_malformed_json_are_skipped_not_fatal():
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434", model="m")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = _sse_lines(
        "",
        "data: not valid json",
        'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}',
        "data: [DONE]",
    )
    with patch("requests.post", return_value=mock_resp):
        chunks = list(llm.stream([{"role": "user", "content": "hi"}]))

    assert [c.delta_content for c in chunks if c.delta_content] == ["ok"]
