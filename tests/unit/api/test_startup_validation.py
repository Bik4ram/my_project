"""
Tests for runtime configuration validation.
"""
from __future__ import annotations

import asyncio
import pytest

from semrag.api.startup_validation import (
    RuntimeConfigurationError,
    _validate_database_url,
    validate_runtime_config,
)
from semrag.core.settings import Settings


def _settings(**overrides) -> Settings:
    """Build a valid Settings instance, then apply test overrides."""
    base = Settings(
        database_url="postgresql+psycopg2://semrag:semrag@localhost:5432/semrag",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        embedding_base_url="http://localhost:11434",
        embedding_dimension=768,
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        llm_base_url="http://localhost:11434",
        environment="development",
    )
    return base.model_copy(update=overrides)


def _run_async(coro):
    """Run an async coroutine synchronously for test compatibility."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_config_passes():
    settings = _settings()
    result = _run_async(validate_runtime_config(settings))
    assert result is settings


def test_mock_llm_passes_validation():
    """mock doesn't need a base URL or model — useful for tests."""
    settings = _settings(llm_provider="mock", llm_base_url="", llm_model="")
    _run_async(validate_runtime_config(settings))  # should not raise


def test_tfidf_embedding_passes_validation():
    """tfidf doesn't need a base URL."""
    settings = _settings(
        embedding_provider="tfidf",
        embedding_base_url="",
    )
    _run_async(validate_runtime_config(settings))


# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------

def test_database_url_must_be_postgresql():
    problems: list[str] = []
    _validate_database_url("mysql://semrag@localhost:3306/semrag", problems)
    assert any("scheme" in p for p in problems)


def test_database_url_must_have_host():
    problems: list[str] = []
    _validate_database_url("postgresql+psycopg2:///semrag", problems)
    assert any("hostname" in p for p in problems)


def test_database_url_with_valid_postgresql_ok():
    problems: list[str] = []
    _validate_database_url(
        "postgresql+psycopg2://semrag:semrag@db.example.com:5432/semrag",
        problems,
    )
    assert problems == []


def test_database_url_invalid_url_raises():
    problems: list[str] = [""]
    # urlparse doesn't raise on most strings; a blank/None triggers a
    # different problem. Pass something urlparse will reject via missing
    # scheme structure.
    _validate_database_url("not-a-url-at-all", problems)
    # "not-a-url-at-all" parses with scheme="not-a-url-at-all" so the
    # scheme check fires.
    assert any("scheme" in p for p in problems)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_embedding_dimension_must_be_positive():
    settings = _settings(embedding_dimension=0)
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "embedding_dimension" in str(exc.value).lower()


def test_embedding_provider_unknown_rejected():
    settings = _settings(embedding_provider="bogus")
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


def test_embedding_ollama_requires_base_url():
    settings = _settings(embedding_base_url="")
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "embedding_base_url" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def test_llm_unknown_provider_rejected():
    settings = _settings(llm_provider="bogus")
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


def test_llm_requires_base_url_for_ollama():
    settings = _settings(llm_base_url="")
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "llm_base_url" in str(exc.value).lower()


def test_llm_requires_model():
    settings = _settings(llm_model="")
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "llm_model" in str(exc.value).lower()


def test_llm_timeout_must_be_positive():
    settings = _settings(llm_timeout_seconds=0)
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "timeout" in str(exc.value).lower()


def test_llm_negative_retries_rejected():
    settings = _settings(llm_max_retries=-1)
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


def test_llm_num_predict_zero_rejected():
    settings = _settings(llm_num_predict=0)
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_production_requires_api_keys():
    settings = _settings(environment="production", api_keys="")
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "api_keys" in str(exc.value).lower()


def test_production_with_default_dev_key_rejected_by_settings():
    """The Settings model already rejects 'dev-local-key' in production
    before validation runs — this test confirms the layered defense."""
    with pytest.raises(ValueError, match="Default development API key"):
        Settings(environment="production", api_keys="dev-local-key")


def test_development_passes_with_dev_key():
    settings = _settings(environment="development", api_keys="dev-local-key")
    _run_async(validate_runtime_config(settings))


def test_max_body_size_must_be_positive():
    settings = _settings(max_body_size=0)
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


def test_rate_limit_negative_rejected():
    settings = _settings(rate_limit_per_minute=-1)
    with pytest.raises(RuntimeConfigurationError):
        _run_async(validate_runtime_config(settings))


# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------

def test_project_roots_empty_entry_rejected():
    settings = _settings(project_roots=["", "/srv/code"])
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    assert "project_roots" in str(exc.value).lower()


def test_project_roots_must_be_absolute():
    """._validate_project_roots is called by Settings first; verify it
    raises via Settings construction."""
    with pytest.raises(ValueError, match="absolute"):
        Settings(project_roots=["relative/path"])


# ---------------------------------------------------------------------------
# Aggregated error message
# ---------------------------------------------------------------------------

def test_validation_aggregates_all_problems():
    """Multiple problems should be reported in one error so the operator
    can fix them all at once."""
    settings = _settings(
        embedding_dimension=0,
        llm_model="",
        llm_base_url="",
    )
    with pytest.raises(RuntimeConfigurationError) as exc:
        _run_async(validate_runtime_config(settings))
    message = str(exc.value)
    assert "embedding_dimension" in message.lower()
    assert "llm_model" in message.lower()
    assert "llm_base_url" in message.lower()
