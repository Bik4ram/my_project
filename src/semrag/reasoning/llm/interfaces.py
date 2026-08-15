"""
LLM provider interfaces.

Same Protocol-based model-agnostic pattern as every prior reasoning-plane
module (EmbeddingProvider in Part 9, VectorStore/GraphStore in Parts 6-7):
nothing outside this module imports a concrete provider class directly.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class CompletionResult(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    # Optional token usage from the provider's response. Most OpenAI-
    # compatible servers report this in `usage.prompt_tokens /
    # completion_tokens / total_tokens`; some (older Ollama, mock) do not.
    # Surface it as metric when present, ignore when absent — no need
    # for a separate API.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StreamChunk(BaseModel):
    delta_content: str | None = None
    delta_tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> CompletionResult: ...

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamChunk]: ...
