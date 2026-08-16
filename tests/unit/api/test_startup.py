"""
Tests for startup validation.
"""
from __future__ import annotations

import os
import pytest

from semrag.core.settings import Settings


def test_reject_dev_keys_in_production():
    """Test that the settings reject default development keys in production."""
    # Set the environment to production and the API keys to the default dev key
    os.environ["SEMRAG_ENVIRONMENT"] = "production"
    os.environ["SEMRAG_API_KEYS"] = "dev-local-key"
    os.environ["SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION"] = "true"  # This is the default anyway

    # We expect that the settings will raise a validation error
    with pytest.raises(ValueError, match="Default development API key"):
        Settings()

    # Clean up the environment
    del os.environ["SEMRAG_ENVIRONMENT"]
    del os.environ["SEMRAG_API_KEYS"]
    if "SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION" in os.environ:
        del os.environ["SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION"]


def test_allow_dev_keys_in_production_if_configured():
    """Test that we can allow dev keys in production if we set the flag to false."""
    os.environ["SEMRAG_ENVIRONMENT"] = "production"
    os.environ["SEMRAG_API_KEYS"] = "dev-local-key"
    os.environ["SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION"] = "false"

    # This should not raise an error
    settings = Settings()
    assert settings.environment == "production"
    assert settings.api_keys == "dev-local-key"
    assert settings.reject_dev_keys_in_production is False

    # Clean up
    del os.environ["SEMRAG_ENVIRONMENT"]
    del os.environ["SEMRAG_API_KEYS"]
    del os.environ["SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION"]


def test_dev_keys_allowed_in_development():
    """Test that dev keys are allowed in development (the default)."""
    os.environ["SEMRAG_ENVIRONMENT"] = "development"
    os.environ["SEMRAG_API_KEYS"] = "dev-local-key"

    # This should not raise an error
    settings = Settings()
    assert settings.environment == "development"
    assert settings.api_keys == "dev-local-key"

    # Clean up
    del os.environ["SEMRAG_ENVIRONMENT"]
    del os.environ["SEMRAG_API_KEYS"]


def test_api_key_store_from_settings():
    """Test that the API key store is created based on settings."""
    from semrag.api.api_key import get_api_key_store, InMemoryAPIKeyStore
    os.environ["SEMRAG_API_KEY_STORE"] = "memory"
    # Reload the settings to pick up the environment variable
    from semrag.core.settings import reset_settings_cache
    reset_settings_cache()
    settings = Settings()
    store = get_api_key_store()
    assert isinstance(store, InMemoryAPIKeyStore)
    # Reset the environment
    del os.environ["SEMRAG_API_KEY_STORE"]
    reset_settings_cache()