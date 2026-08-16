"""LLM factory — provider routing and probe."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from semrag.core.settings import Settings
from semrag.reasoning.llm.factory import (
    LLMConfigurationError,
    build_llm,
)
from semrag.reasoning.llm.mock_llm import MockLLM
from semrag.reasoning.llm.ollama_llm import OllamaLLM
from semrag.reasoning.llm.openai_compatible_llm import OpenAICompatibleLLM
from semrag.reasoning.llm.openai_llm import OpenAILLM


def _settings(**overrides) -> Settings:
    """Build a Settings instance with probe disabled so tests don't
    need a live provider."""
    base = {
        "llm_probe_at_startup": False,
        "llm_model": "any",
        "llm_base_url": "http://example.test",
    }
    base.update(overrides)
    return Settings(**base)


def test_ollama_returns_ollama_llm():
    client = build_llm(_settings(llm_provider="ollama"))
    assert isinstance(client, OllamaLLM)


def test_ollama_wires_num_predict_and_think_from_settings():
    """Regression for the qwen3 empty-answer bug: the factory must pull
    ``llm_num_predict`` and the Ollama ``think`` toggle from Settings so a
    reasoning model doesn't exhaust its generation budget inside the
    thinking phase (``done_reason="length"``, ``content=""``)."""
    settings = _settings(
        llm_provider="ollama",
        llm_model="qwen3:4b",
        llm_num_predict=2048,
        llm_ollama_think=True,
    )
    client = build_llm(settings)
    assert isinstance(client, OllamaLLM)
    assert client._num_predict == 2048
    assert client._think is True


def test_ollama_defaults_keep_thinking_enabled():
    """The default deployment must keep Qwen3 thinking ENABLED: reasoning
    then lives in ``message.thinking`` and ``content`` stays clean, which
    the coding agent's strict-JSON steps rely on. The generation-budget fix
    is ``llm_num_predict`` headroom, not disabled thinking."""
    settings = _settings(llm_provider="ollama")
    assert settings.llm_ollama_think is True
    assert settings.llm_num_predict >= 2048
    client = build_llm(settings)
    assert client._think is True


def test_openai_returns_openai_llm():
    client = build_llm(_settings(llm_provider="openai"))
    assert isinstance(client, OpenAILLM)


def test_openai_compatible_returns_base():
    client = build_llm(_settings(llm_provider="openai_compatible"))
    assert isinstance(client, OpenAICompatibleLLM)


def test_vllm_uses_openai_compatible():
    """vLLM exposes the OpenAI-compatible API; we treat it the same as
    any other OpenAI-compat server and explicitly route there."""
    client = build_llm(_settings(llm_provider="vllm"))
    assert isinstance(client, OpenAICompatibleLLM)


def test_mock_returns_mock_llm():
    client = build_llm(_settings(llm_provider="mock"))
    assert isinstance(client, MockLLM)


def test_anthropic_raises_not_implemented():
    with pytest.raises(LLMConfigurationError) as exc:
        build_llm(_settings(llm_provider="anthropic"))
    assert "not yet implemented" in str(exc.value)


def test_unknown_provider_raises():
    # Settings validates the Literal, so we bypass it by reaching the
    # factory through a stubbed Settings instance.
    settings = _settings()
    object.__setattr__(settings, "llm_provider", "nonexistent")
    with pytest.raises(LLMConfigurationError):
        build_llm(settings)


def test_probe_runs_at_startup_when_enabled():
    settings = _settings(llm_provider="mock", llm_probe_at_startup=True)
    with patch.object(MockLLM, "complete", return_value=None) as mock_complete:
        client = build_llm(settings)
    assert isinstance(client, MockLLM)
    assert mock_complete.called


def test_probe_skipped_when_disabled():
    settings = _settings(llm_provider="mock", llm_probe_at_startup=False)
    with patch.object(MockLLM, "complete", return_value=None) as mock_complete:
        client = build_llm(settings)
    assert isinstance(client, MockLLM)
    assert not mock_complete.called