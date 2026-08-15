"""Tests for the secret redaction helper."""
from __future__ import annotations

from semrag.core.redaction import (
    _is_sensitive_key,
    redact_payload,
    redact_value,
)


def test_sensitive_key_detection_is_case_insensitive():
    assert _is_sensitive_key("api_key") is True
    assert _is_sensitive_key("API_KEY") is True
    assert _is_sensitive_key("Api-Key") is True
    assert _is_sensitive_key("apikey") is True
    # Substring match — a key containing "token" anywhere is redacted.
    assert _is_sensitive_key("user_token") is True
    assert _is_sensitive_key("refreshToken") is True


def test_non_sensitive_keys_are_not_flagged():
    assert _is_sensitive_key("project_id") is False
    assert _is_sensitive_key("count") is False
    assert _is_sensitive_key("status") is False
    # False positive risk: a field called "count_of_tokens" — we
    # intentionally still redact it because the substring "token"
    # indicates a credential-adjacent name. Confirmed via a test
    # so the policy is explicit.
    assert _is_sensitive_key("count_of_tokens") is True


def test_redact_value_replaces_sensitive_strings():
    assert redact_value("api_key", "sk-abc") == "***"
    assert redact_value("password", "hunter2") == "***"
    # Bearer token in a free-form string gets scrubbed, not the
    # entire string.
    redacted = redact_value("note", "Authorization: Bearer abc.def.ghi")
    assert "abc.def.ghi" not in redacted
    assert "Bearer" in redacted


def test_redact_value_scrubs_url_passwords():
    raw = "postgresql://user:hunter2@host:5432/db"
    scrubbed = redact_value("note", raw)
    assert "hunter2" not in scrubbed
    assert "host:5432/db" in scrubbed


def test_redact_value_passes_safe_strings_through():
    assert redact_value("path", "/v1/projects/p1/query") == "/v1/projects/p1/query"


def test_redact_value_passes_non_strings_through():
    assert redact_value("count", 42) == 42
    assert redact_value("flag", True) is True


def test_redact_payload_handles_nested_dict():
    payload = {
        "project_id": "p1",
        "api_key": "sk-abc",
        "credentials": {"password": "hunter2"},
        "note": "Bearer abc.def.ghi",
    }
    scrubbed = redact_payload(payload)
    assert scrubbed["api_key"] == "***"
    assert scrubbed["credentials"]["password"] == "***"
    assert "abc.def.ghi" not in scrubbed["note"]


def test_redact_payload_drops_aws_keys():
    payload = {"log": "user agent AKIAABCDEFGHIJKLMNOP saw this"}
    scrubbed = redact_payload(payload)
    assert "AKIAABCDEFGHIJKLMNOP" not in scrubbed["log"]


def test_redact_payload_drops_jwt_tokens():
    payload = {"token": "eyJabc12345.eyJdef67890.signature9999"}
    scrubbed = redact_payload(payload)
    # The key is sensitive so the whole value is replaced.
    assert scrubbed["token"] == "***"
    # And if the JWT appears as part of a less-sensitive field, it's
    # still scrubbed.
    payload2 = {"note": "saw eyJabc12345.eyJdef67890.signature9999 today"}
    scrubbed2 = redact_payload(payload2)
    assert "eyJabc12345" not in scrubbed2["note"]
    assert "signature9999" not in scrubbed2["note"]