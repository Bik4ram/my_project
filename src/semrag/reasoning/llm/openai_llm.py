"""
Hosted OpenAI chat completion client.

Separate class from `OpenAICompatibleLLM` for clarity: OpenAI's hosted API
is one specific OpenAI-shaped provider, and unlike Ollama or
vLLM-compatible servers it is not a server the operator runs themselves
(no `/v1` shim concerns, no Ollama-flavoured streaming format). Splitting
them keeps each class small, keeps the contract one HTTP conversation,
and keeps future providers (Anthropic, Cohere chat, etc.) easy to add
without growing either one.
"""
from __future__ import annotations

from semrag.reasoning.llm._openai_compatible_impl import OpenAICompatibleLLM


class OpenAILLM(OpenAICompatibleLLM):
    """Hosted OpenAI chat completions. Identical to the OpenAI-compatible
    shape — the base class already targets ``/v1/chat/completions``,
    joined onto the configured ``base_url`` (``https://api.openai.com``
    by convention); only the reported provider name differs. The
    streaming/non-streaming parsers inherited from the base class work
    as-is."""

    @property
    def provider_name(self) -> str:
        return "openai"
