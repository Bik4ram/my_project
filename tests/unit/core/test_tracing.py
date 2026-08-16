"""Tests for the OpenTelemetry tracing facade."""
from __future__ import annotations

import pytest

from semrag.core import tracing


@pytest.fixture(autouse=True)
def _reset():
    """Reset the tracing flag so each test re-configures from scratch."""
    tracing.reset_tracing_state()
    yield
    tracing.reset_tracing_state()


class _FakeSettings:
    """Minimal stand-in for `semrag.core.settings.Settings`.

    The tracing facade only reads the four attributes it cares
    about; mocking the whole pydantic-settings model in tests is
    heavyweight and not necessary.
    """

    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "semrag-test"
    trace_sample_ratio: float = 1.0


def test_disabled_tracing_returns_noop():
    settings = _FakeSettings()
    tracing.configure_tracing(settings)
    tracer = tracing.tracer("test.scope")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("k", "v")  # noop
    # Trace id helpers return zeros.
    assert tracing.current_trace_id_hex() == "0" * 32
    assert tracing.current_span_id_hex() == "0" * 16


def test_idempotent_configuration():
    settings = _FakeSettings()
    tracing.configure_tracing(settings)
    # Second call is a no-op (would raise on the second tracer
    # provider install otherwise).
    tracing.configure_tracing(settings)
    assert tracing.tracer("test.scope") is not None


def test_context_helpers_noop_when_disabled():
    settings = _FakeSettings()
    tracing.configure_tracing(settings)
    headers: dict[str, str] = {"X-Test": "1"}
    # inject_context is a no-op when tracing is disabled — the
    # headers dict is returned unchanged.
    out = tracing.inject_context(headers)
    assert out is headers
    # extract_context returns an empty dict.
    assert tracing.extract_context({"traceparent": "00-aa-bb-01"}) == {}


def test_enabled_tracing_uses_real_tracer(monkeypatch):
    """When tracing_enabled is True the SDK is wired up and
    `current_trace_id_hex()` returns a non-zero value inside a span."""
    # The endpoint points at a non-existent collector; the test only
    # needs the SDK to be installed and the trace id helpers to work.
    # Silence the OTel exporter logs so the missing collector doesn't
    # spam the test output.
    import logging

    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)

    class _EnabledSettings:
        tracing_enabled = True
        otlp_endpoint = "http://localhost:4317"
        otel_service_name = "semrag-test"
        trace_sample_ratio = 1.0

    settings = _EnabledSettings()
    tracing.configure_tracing(settings)
    tracer = tracing.tracer("test.enabled")
    with tracer.start_as_current_span("op") as span:
        # Inside a real span, the trace id is non-zero.
        trace_id = tracing.current_trace_id_hex()
        span_id = tracing.current_span_id_hex()
        assert trace_id != "0" * 32
        assert span_id != "0" * 16
        # Span attributes are accepted (the real OTel API).
        span.set_attribute("http.route", "/v1/query")
        span.set_attribute("http.status_code", 200)
    # Shut the provider down so the BatchSpanProcessor doesn't keep
    # retrying to the unreachable collector after the test exits.
    from opentelemetry import trace as _otel_trace

    provider = _otel_trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()


def test_sampling_ratio_zero_still_noop_tracer_when_disabled():
    """If tracing is disabled the sample ratio doesn't matter —
    the no-op tracer is always returned."""
    settings = _FakeSettings()
    settings.trace_sample_ratio = 0.0
    tracing.configure_tracing(settings)
    tracer = tracing.tracer("test.zero")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("k", "v")
    # No active trace.
    assert tracing.current_trace_id_hex() == "0" * 32