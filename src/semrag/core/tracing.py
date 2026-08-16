"""
OpenTelemetry tracing for Wave 3.

Wraps the OpenTelemetry SDK in a thin facade so:

- The application code is decoupled from the OTel SDK version.
- Tracing can be disabled (tests, offline dev) without every import
  failing — the tracer becomes a no-op when ``Settings.tracing_enabled``
  is False or the SDK isn't installed.
- OTLP export endpoint, sampling ratio, and service name come from
  ``Settings`` and are not hardcoded.

The module exposes:

- ``configure_tracing(settings)`` — call once at startup. Idempotent.
- ``tracer(name)`` — returns a ``Tracer`` whose ``start_as_current_span``
  is a context manager that yields a ``Span`` with the same
  ``set_attribute`` / ``record_exception`` API as the OTel SDK.
- ``current_trace_id_hex()`` / ``current_span_id_hex()`` — return the
  W3C-formatted trace/span ids so the structured logging formatter
  can stamp every log record with them.
- ``inject_context(headers)`` / ``extract_context(headers)`` — propagate
  trace context across process boundaries (HTTP, HTTP→DB driver).

OpenTelemetry imports are deferred to ``configure_tracing`` so the
tests that don't enable tracing don't pull grpc / protobuf. When the
SDK is not installed at runtime we fall back to a no-op tracer.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

from semrag.core.settings import Settings

_logger = logging.getLogger(__name__)

# Defaults that match the W3C Trace Context spec — used when no span
# is active on the current context. A zeroed-out trace id is a
# distinguishable sentinel; never used for a real trace.
_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16

_configured = False


class _NoopSpan:
    """No-op replacement for OTel Span when tracing is disabled.

    Implements the subset of the OTel Span API the project actually
    uses: ``set_attribute``, ``record_exception``, ``set_status``,
    ``end``. Tests and offline dev paths fall through here.
    """

    def set_attribute(self, key: str, value: Any) -> "_NoopSpan":
        return self

    def record_exception(self, exception: BaseException) -> "_NoopSpan":
        return self

    def set_status(self, status: Any, description: str | None = None) -> "_NoopSpan":
        return self

    def end(self) -> None:
        return None

    def get_span_context(self) -> "_NoopSpanContext":
        return _NoopSpanContext()


class _NoopSpanContext:
    trace_id = 0
    span_id = 0
    is_valid = False
    trace_flags = 0


class _NoopTracer:
    """No-op tracer returned by ``tracer()`` when tracing is disabled."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def configure_tracing(settings: Settings) -> None:
    """Initialise the OpenTelemetry SDK from Settings.

    Idempotent — second and subsequent calls are no-ops. Tests that
    swap settings can clear ``_configured`` directly to force a
    re-init (we keep this internal; there's no operator-driven reason
    to re-initialise at runtime).

    Falls back to a no-op tracer when:

    - ``settings.tracing_enabled`` is False
    - The OTel SDK or OTLP exporter is not importable

    In both cases the rest of the application code keeps using
    ``tracer(...)`` and gets the no-op implementation back.
    """
    global _configured
    if _configured:
        return

    if not settings.tracing_enabled:
        _configured = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _logger.warning(
            "opentelemetry SDK not installed; tracing disabled",
        )
        _configured = True
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=_build_sampler(settings.trace_sample_ratio),
    )

    # Pick the exporter family from the URL scheme: http+ → http/protobuf,
    # otherwise → gRPC. The default OTLP port 4317 is gRPC, 4318 is http.
    try:
        exporter = _build_exporter(settings.otlp_endpoint)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as exc:  # noqa: BLE001 — exporter failures must not crash startup
        _logger.warning(
            "OTLP exporter init failed; continuing without exporter",
            extra={"endpoint": settings.otlp_endpoint, "error": str(exc)},
        )

    trace.set_tracer_provider(provider)
    _configured = True


def _build_sampler(ratio: float):
    """Build the OTel sampler for the configured ratio.

    A ratio of 1.0 = always on, 0.0 = always off. Anything in between
    uses TraceIdRatioBased; we don't reach for parent-based sampling
    here because Wave 3's trace propagation is in-process.
    """
    try:
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_OFF,
            ALWAYS_ON,
            TraceIdRatioBased,
        )
    except ImportError:
        return ALWAYS_ON
    if ratio <= 0.0:
        return ALWAYS_OFF
    if ratio >= 1.0:
        return ALWAYS_ON
    return TraceIdRatioBased(ratio)


def _build_exporter(endpoint: str):
    """Return an OTLP span exporter for the configured endpoint.

    URL scheme → exporter:
      http://  → OTLPSpanExporter over HTTP/protobuf (port 4318)
      https:// → OTLPSpanExporter over HTTP/protobuf (port 4318)
      grpc://  → OTLPSpanExporter over gRPC (port 4317)
      otherwise (no scheme) → OTLPSpanExporter over gRPC
    """
    parsed = urlparse(endpoint)
    scheme = (parsed.scheme or "").lower()
    host_port = parsed.netloc or endpoint
    if scheme in ("http", "https"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpExporter,
            )
        except ImportError:
            return None
        return HttpExporter(endpoint=endpoint)
    # gRPC path — default.
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcExporter,
        )
    except ImportError:
        return None
    return GrpcExporter(endpoint=host_port, insecure=True)


# ---------------------------------------------------------------------------
# Tracer / span accessors
# ---------------------------------------------------------------------------


def tracer(name: str) -> Any:
    """Return a tracer for the given instrumentation scope.

    Falls back to the no-op tracer when tracing is disabled or the
    SDK is unavailable. The returned object supports the
    ``start_as_current_span(name, **attrs)`` context-manager API.
    """
    try:
        from opentelemetry import trace as _otel_trace

        if not _configured:
            return _NoopTracer()
        return _otel_trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


# ---------------------------------------------------------------------------
# Trace-id helpers — used by the structured logging formatter so every
# log line carries the active trace_id / span_id without forcing
# every log call to know about OpenTelemetry.
# ---------------------------------------------------------------------------


def current_trace_id_hex() -> str:
    """Return the current trace id as a 32-char hex string, or zeros."""
    try:
        from opentelemetry import trace as _otel_trace

        if not _configured:
            return _ZERO_TRACE_ID
        span = _otel_trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is None or not getattr(ctx, "is_valid", False):
            return _ZERO_TRACE_ID
        return format(ctx.trace_id, "032x")
    except ImportError:
        return _ZERO_TRACE_ID


def current_span_id_hex() -> str:
    """Return the current span id as a 16-char hex string, or zeros."""
    try:
        from opentelemetry import trace as _otel_trace

        if not _configured:
            return _ZERO_SPAN_ID
        span = _otel_trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is None or not getattr(ctx, "is_valid", False):
            return _ZERO_SPAN_ID
        return format(ctx.span_id, "016x")
    except ImportError:
        return _ZERO_SPAN_ID


# ---------------------------------------------------------------------------
# Context propagation across boundaries (HTTP, …)
# ---------------------------------------------------------------------------


def inject_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject the active trace context into a header dict (W3C format).

    Mutates and returns ``headers``. Inactive / disabled tracing is a
    no-op; the headers are returned unchanged.
    """
    try:
        from opentelemetry import trace as _otel_trace
        from opentelemetry.propagate import inject

        if not _configured:
            return headers
        inject(headers)
        return headers
    except ImportError:
        return headers


def extract_context(headers: dict[str, str]) -> dict[str, str]:
    """Extract a trace context from a header dict into a Context.

    The returned object is the OTel ``Context`` (or a no-op stand-in
    when tracing is disabled). Pass it to ``tracer().start_as_current_span``
    as the ``context=`` argument to make the new span a child of the
    remote parent.
    """
    try:
        from opentelemetry.propagate import extract

        if not _configured:
            return {}
        return extract(headers)
    except ImportError:
        return {}


def reset_tracing_state() -> None:
    """Test helper: clear the configured flag so ``configure_tracing``
    runs again on the next call. Production never invokes this."""
    global _configured
    _configured = False