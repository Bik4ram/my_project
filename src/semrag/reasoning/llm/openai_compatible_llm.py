"""Backward-compatible re-export. The class now lives in this module for
historical reasons; new consumers should import `OpenAICompatibleLLM`
from here (and therefore get the corrected shape — Ollama-specific code
has moved to `OllamaLLM`)."""
from semrag.reasoning.llm._openai_compatible_impl import OpenAICompatibleLLM, LLMProviderError, _post_with_retries_shared

__all__ = ["OpenAICompatibleLLM", "LLMProviderError", "_post_with_retries_shared"]
