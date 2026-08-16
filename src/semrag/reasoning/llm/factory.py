"""
LLM provider factory.

A single `build_llm(settings)` returns an `LLMProvider` configured for the
configured `settings.llm_provider`. New providers (Anthropic, Cohere,
HuggingFace TGI, etc.) plug in here as a one-line `elif` branch with a
corresponding concrete class.

The factory intentionally reads from `Settings` rather than accepting
loose kwargs: one source of truth means the CLI and the API can't drift,
and a misconfiguration (typo in `llm_provider`, model not set) fails at
startup, not in the middle of a request.

`probe_at_startup=True` does one cheap request at process start to confirm
the endpoint actually responds — catches URL typos, missing auth, and the
"Ollama 0.32.x returns 404 on /v1/chat/completions" footgun before the
first user waits for it.
"""
from __future__ import annotations

from semrag.core.observability import get_logger
from semrag.core.settings import Settings
from semrag.reasoning.llm.interfaces import LLMProvider
from semrag.reasoning.llm.ollama_llm import OllamaLLM
from semrag.reasoning.llm.openai_llm import OpenAILLM
from semrag.reasoning.llm.openai_compatible_llm import OpenAICompatibleLLM

logger = get_logger(__name__)


class LLMConfigurationError(RuntimeError):
    pass


def build_llm(settings: Settings | None = None) -> LLMProvider:
    """Build the configured LLM provider."""
    if settings is None:
        from semrag.core.settings import get_settings
        settings = get_settings()

    provider = settings.llm_provider
    common = {
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }

    if provider == "ollama":
        # Ollama accepts ``options.num_predict`` per call; we set it from
        # ``Settings.llm_num_predict`` so the latency cap is operator-
        # configurable rather than hard-coded in the client. ``think`` is
        # the top-level flag that keeps Qwen3-class reasoning models from
        # spending the whole generation budget in their thinking phase
        # (which otherwise ends in ``done_reason="length"`` and an empty
        # answer); it is Ollama-specific, so it lives on the Ollama client.
        client: LLMProvider = OllamaLLM(
            **common,
            num_predict=settings.llm_num_predict,
            think=settings.llm_ollama_think,
        )
    elif provider == "openai":
        client = OpenAILLM(**common)
    elif provider in ("openai_compatible", "vllm"):
        # vLLM exposes the OpenAI-compatible API on `/v1/chat/completions`;
        # any other OpenAI-compat server (llama.cpp's HTTP server, hosted
        # OpenAI-compat APIs) is the same shape.
        client = OpenAICompatibleLLM(**common)
    elif provider == "anthropic":
        # Anthropic's Messages API is a different shape than OpenAI's;
        # the concrete class will be added when the deployment target
        # requires it. Raising here is intentional — silent fallback to
        # an OpenAI-shaped client would produce malformed requests.
        raise LLMConfigurationError(
            "Anthropic provider not yet implemented; PRs welcome. Set "
            "SEMRAG_LLM_PROVIDER=openai_compatible with an OpenAI-compat "
            "shim in front of Claude for now."
        )
    elif provider == "mock":
        from semrag.reasoning.llm.mock_llm import MockLLM
        client = MockLLM(model=settings.llm_model)
    else:
        raise LLMConfigurationError(f"Unknown llm_provider: {provider!r}")

    if settings.llm_probe_at_startup:
        _probe(client)
    return client


def _probe(client: LLMProvider) -> None:
    """One minimal request to confirm the provider is reachable. Catches
    URL typos, missing API keys, model-not-found errors, and provider
    versions that don't expose the expected endpoint."""
    try:
        client.complete([{"role": "user", "content": "ping"}])
        logger.info("llm probe succeeded", extra={"provider": client.provider_name, "model": client.model_name})
    except Exception as exc:  # noqa: BLE001 — startup probe; let the operator see what failed
        logger.error(
            "llm probe failed",
            extra={"provider": client.provider_name, "model": client.model_name, "error": str(exc)},
        )
        raise