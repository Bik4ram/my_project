"""
Deterministic in-process LLM for tests and CI.

Returns scripted responses in order regardless of the actual messages. The
first N responses are non-streaming `CompletionResult`s; if `responses`
contains tool calls the caller (typically `AgentLoop`) drives subsequent
turns. Streaming is implemented as a single chunk containing the whole
response.

Why a real (network-attached) provider isn't the test default: every CI
run that talks to a real LLM is slow, flaky, costs money, and leaks
prompt text. Tests should not exercise network code at all; provider
behavior (request shaping, retry policy, response parsing) is tested in
isolation against `responses`-faked httpx responses.
"""
from __future__ import annotations

from collections.abc import Iterator

from semrag.core.observability import get_logger
from semrag.reasoning.llm.interfaces import CompletionResult, StreamChunk, ToolCall

logger = get_logger(__name__)


class MockLLM:
    def __init__(self, model: str = "mock", responses: list[CompletionResult] | None = None):
        self._model = model
        self._responses = list(responses or [])
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def queue(self, response: CompletionResult) -> None:
        self._responses.append(response)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> CompletionResult:
        if not self._responses:
            logger.warning("mock_llm exhausted", extra={"calls": self._call_count})
            return CompletionResult(content="(mock: no scripted response)")
        self._call_count += 1
        return self._responses.pop(0)

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamChunk]:
        result = self.complete(messages, tools)
        if result.tool_calls:
            yield StreamChunk(delta_tool_calls=result.tool_calls, finish_reason="tool_calls")
        else:
            yield StreamChunk(delta_content=result.content, finish_reason="stop")