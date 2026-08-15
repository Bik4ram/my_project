"""Secret redaction for structured logs.

Production log pipelines ingest everything written via ``logger.info``
or `extra=...`. A single unredacted API key or DB connection string
becomes a credential leak the moment a log shipper forwards the line
to a third-party SaaS. This module is the single chokepoint — every
redaction goes through ``redact_value`` / ``redact_payload`` so the
rules stay in one file.

What we redact:

- Any value whose key matches a sensitive name (case-insensitive
  substring match: ``api_key``, ``token``, ``password``,
  ``authorization``, ``secret``, ``private_key``,
  ``connection_string``).
- Bearer-style HTTP authorization headers (``Authorization: Bearer ...``).
- Postgres / MySQL connection strings — the password component of a URL
  is replaced with ``***`` so the userinfo half still pins the host/port
  for debugging.

What we deliberately do NOT redact:

- The request_id, trace_id, project_id, job_id — operators need them.
- Numeric fields, no matter the name — a ``count`` of failed logins is
  safe to log.
- URLs without credentials (they have nothing to hide).

The redactor is conservative: when in doubt, it redacts. False
positives (a payload field called ``count_of_tokens`` getting
sanitised) are preferable to false negatives.
"""
from __future__ import annotations

import re
from typing import Any

# Case-insensitive substring match — fast enough at log emission time
# (single dict per record). Order matters only for readability;
# `_is_sensitive_key` short-circuits on the first match.
_SENSITIVE_KEY_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "auth_header",
    "auth-token",
    "auth_token",
    "authtoken",
    "bearer",
    "connection_string",
    "connectionstring",
    "db_password",
    "dbpassword",
    "passphrase",
    "password",
    "private_key",
    "privatekey",
    "secret",
    "session_token",
    "sessiontoken",
    "token",
)

_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-+/=]+")
# Connection-string password component. Matches `://user:password@host`
# so the password is replaced while the rest stays useful for triage.
_URL_PASSWORD_PATTERN = re.compile(
    r"(://[^\s:@]+:)([^@\s]+)(@)"
)

# Markers in string values that betray the presence of an inline
# credential — used to scrub the string itself rather than only
# specific keys. Covers Bearer tokens, AWS keys (AKIA…), JWT
# (`eyJ...`), and Postgres-style URLs that escaped the structured
# key detection because they were passed as a free-form value.
_INLINE_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    _BEARER_PATTERN,
    _URL_PASSWORD_PATTERN,
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    # JWT — three base64url segments separated by dots, where each
    # segment is 4+ chars (matches real JWTs which have much longer
    # segments; short test fixtures use ``eyJaa.bbbb.cccc``).
    re.compile(r"\beyJ[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{3,}\b"),
)


def _is_sensitive_key(key: str) -> bool:
    """Case-insensitive substring match against the sensitive key list."""
    lower = key.lower()
    return any(marker in lower for marker in _SENSITIVE_KEY_MARKERS)


def _redact_string(value: str) -> str:
    """Scrub inline secrets from a free-form string value."""
    redacted = value
    for pattern in _INLINE_SECRET_PATTERNS:
        if pattern is _BEARER_PATTERN:
            redacted = pattern.sub(r"\1***", redacted)
        elif pattern is _URL_PASSWORD_PATTERN:
            redacted = pattern.sub(r"\1***\3", redacted)
        else:
            redacted = pattern.sub("***", redacted)
    return redacted


def _redact_scalar(value: Any) -> Any:
    """Apply string-level scrubbing if the value is a string; pass
    other scalars through unchanged."""
    if isinstance(value, str):
        return _redact_string(value)
    return value


def redact_value(key: str, value: Any) -> Any:
    """Redact a single (key, value) pair. Sensitive keys are replaced
    with the literal string ``***`` regardless of value type; other
    values get inline secret scrubbing if they're strings."""
    if _is_sensitive_key(key):
        return "***"
    return _redact_scalar(value)


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with sensitive fields redacted.

    Iterates the dict shallowly — a nested dict is itself redacted
    but not walked recursively. Production call sites pass already-
    flattened payloads (the JSON formatter does the flattening); if a
    caller passes a nested dict, the outer dict's keys are inspected
    and the inner dict is returned (or replaced with ``***``) without
    walking inside it. This is intentional: a deeply recursive
    redactor is hard to audit, and the LogRecord pipeline flattens
    before we get here.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_sensitive_key(key):
            out[key] = "***"
        elif isinstance(value, dict):
            out[key] = {k: redact_value(k, v) for k, v in value.items()}
        else:
            out[key] = _redact_scalar(value)
    return out