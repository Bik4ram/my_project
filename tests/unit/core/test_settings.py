"""Settings model — the single source of truth for env-var-driven config.

These tests verify required defaults, validation, and the api_keys
normalisation. No file I/O; isolation env vars are set per test."""
from __future__ import annotations

import os

import pytest

from semrag.core.settings import Settings, get_settings, reset_settings_cache


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all SEMRAG_ env vars so Settings picks up only what the test
    sets. Tests that don't care about defaults can ignore the fixture."""
    for key in list(os.environ):
        if key.startswith("SEMRAG_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield monkeypatch
    reset_settings_cache()


def test_defaults_when_no_env_vars(clean_env):
    settings = Settings()
    assert settings.log_level == "INFO"
    assert settings.embedding_provider == "ollama"
    assert settings.llm_provider == "ollama"
    assert settings.embedding_dimension == 768
    assert settings.debug_artifact_retention == 5


def test_trace_level_acceptable(clean_env):
    settings = Settings(log_level="TRACE")
    assert settings.log_level == "TRACE"


def test_invalid_log_level_rejected(clean_env):
    with pytest.raises(ValueError):
        Settings(log_level="TRACEY")  # typo: not a valid literal


def test_api_keys_strips_and_dedups(clean_env):
    settings = Settings(api_keys="  a, b ,a , ,, c ")
    assert settings.api_keys_set == {"a", "b", "c"}
    assert settings.api_key_list() == ["a", "b", "c"]


def test_debug_writer_uses_settings_paths(clean_env):
    settings = Settings(debug_artifacts_dir="/tmp/x", debug_artifact_retention=2)
    writer = settings.debug_writer()
    assert writer.is_active is False  # INFO is the default
    assert writer._retention == 2


def test_debug_writer_active_in_debug(clean_env):
    settings = Settings(log_level="DEBUG", debug_artifacts_dir="/tmp/x")
    writer = settings.debug_writer()
    assert writer.is_active is True


def test_get_settings_cached_singleton(clean_env):
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_cache_picks_up_new_env(clean_env, monkeypatch):
    a = get_settings()
    monkeypatch.setenv("SEMRAG_LLM_MODEL", "different-model")
    b = reset_settings_cache()
    assert b.llm_model == "different-model"
    assert a is not b


def test_area_roots_default_to_none(clean_env):
    settings = Settings()
    assert settings.rtl_root is None
    assert settings.tb_root is None
    assert settings.docs_root is None
    assert settings.project_id is None


def test_area_roots_accept_absolute_paths(clean_env):
    settings = Settings(
        rtl_root="/srv/rtl",
        tb_root="/srv/tb",
        docs_root="/srv/docs",
        project_id="chip",
    )
    assert settings.rtl_root == "/srv/rtl"
    assert settings.tb_root == "/srv/tb"
    assert settings.docs_root == "/srv/docs"
    assert settings.project_id == "chip"


def test_area_roots_reject_relative_paths(clean_env):
    with pytest.raises(Exception):
        Settings(rtl_root="rtl")
    with pytest.raises(Exception):
        Settings(tb_root="./tb")
    with pytest.raises(Exception):
        Settings(docs_root="../../docs")


def test_area_roots_accept_blank_as_none(clean_env):
    settings = Settings(rtl_root="", tb_root="   ", docs_root=None)
    assert settings.rtl_root is None
    assert settings.tb_root is None
    assert settings.docs_root is None
