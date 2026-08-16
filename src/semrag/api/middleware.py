"""
HTTP middleware for security and observability.

Wave 3 additions:

- `MetricsMiddleware` records HTTP request count, latency, request/
  response size, and exceptions to the Prometheus registry. It runs
  BEFORE `LoggingMiddleware` so the latency it measures includes the
  cost of the structured log emission, not just the route handler.

- `LoggingMiddleware` propagates the active trace context via the
  W3C `traceparent` header so downstream HTTP clients (the LLM
  provider, the embedding provider) inherit the trace. It also
  binds the request id to the OTel current span as an attribute,
  so a `grep request_id=...` in logs maps cleanly to the matching
  trace in the tracing backend.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from semrag.core.settings import get_settings
from semrag.core.observability import get_logger
from semrag.core import metrics as metrics_module
from semrag.core import tracing as tracing_module
from semrag.audit import log_audit_event

logger = get_logger(__name__)


def _route_template(request: Request) -> str:
    """Return the FastAPI route template for `request`, falling back
    to the URL path when the request didn't match a route.

    We never label on the raw URL — that would explode the
    Prometheus series count with one series per project id. The
    template (`/v1/projects/{project_id}/query`) has a small,
    bounded cardinality.
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path  # type: ignore[no-any-return]
    return request.url.path


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and attach a UUID4 request ID to each request.

    If the inbound request already carries an `X-Request-ID` header
    (a load balancer or upstream proxy propagating it), that value
    is reused so the request id remains consistent across the
    upstream→downstream boundary. UUID4 is generated otherwise.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get("X-Request-ID")
        request_id = incoming if incoming else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TracingMiddleware(BaseHTTPMiddleware):
    """Wrap each request in an OpenTelemetry span.

    When tracing is disabled (Settings.tracing_enabled=False or the
    SDK isn't installed) the middleware is a thin pass-through —
    `tracer().start_as_current_span` returns a no-op span whose
    `end()` is a no-op. No special-case code is needed at the call
    site.

    The span carries the same low-cardinality labels the Prometheus
    middleware uses (method, route template, status code) so the
    two backends see the same request shape.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        template = _route_template(request)
        tracer = tracing_module.tracer("semrag.http")
        with tracer.start_as_current_span(
            f"{request.method} {template}",
            attributes={
                "http.method": request.method,
                "http.route": template,
                "http.request_id": getattr(request.state, "request_id", ""),
            },
        ) as span:
            try:
                response = await call_next(request)
            except Exception as exc:
                span.record_exception(exc)
                raise
            span.set_attribute("http.status_code", response.status_code)
            # Propagate trace context downstream so HTTP clients in the
            # LLM / embedding providers can pick it up.
            tracing_module.inject_context(dict(response.headers))
            return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP latency, request/response size, status, and exceptions.

    Intentionally records the metrics the Wave 3 audit calls out:

    - request count (`semrag_http_requests_total`)
    - request duration (`semrag_http_request_duration_seconds`)
    - status code (`status` label on the two counters)
    - request size (`semrag_http_request_size_bytes`)
    - response size (`semrag_http_response_size_bytes`)
    - exceptions (`semrag_http_exceptions_total`)

    Does NOT log — `LoggingMiddleware` handles logging. The two
    middlewares are deliberately separate so a future change (e.g.
    reducing log noise) only affects one of them.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        template = _route_template(request)
        method = request.method
        # Content-Length is the only header-blessed size signal. Some
        # clients omit it (chunked transfer encoding); a missing or
        # unparseable value is recorded as -1 (an explicit sentinel —
        # Prometheus' histogram skips negative observations cleanly).
        try:
            request_bytes = int(request.headers.get("content-length", "-1"))
        except ValueError:
            request_bytes = -1
        start = time.perf_counter()
        response_status = 500
        response = None
        try:
            response = await call_next(request)
            response_status = response.status_code
            return response
        except Exception as exc:
            metrics_module.inc_http_exception(method, template, exc.__class__.__name__)
            raise
        finally:
            duration_s = time.perf_counter() - start
            response_bytes = -1
            if response is not None:
                # Read response body length off the Content-Length
                # header we may have set, or compute from chunks if
                # available. Streaming responses don't carry a
                # Content-Length; record -1 to keep cardinality
                # bounded.
                cl = response.headers.get("content-length")
                if cl is not None:
                    try:
                        response_bytes = int(cl)
                    except ValueError:
                        response_bytes = -1
            metrics_module.observe_http(
                method=method,
                path=template,
                status=response_status,
                duration_s=duration_s,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests and responses with latency, and audit events."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        # Log the request with request ID, method, path, status code, and latency.
        # stdlib `logger.info` accepts user fields only via `extra=...` — passing
        # them as plain kwargs raises TypeError("Logger._log() got an unexpected
        # keyword argument") on every request. The fields stay accessible to
        # JSON / human formatters via the LogRecord's __dict__ (see
        # core/observability._JsonFormatter).
        logger.info(
            "Request completed",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency": process_time,
            },
        )

        # Audit logging for authentication and authorization failures
        # We'll check the response status code and the request path to determine the event type.
        if response.status_code == 401:
            log_audit_event(
                event_type="authentication_failure",
                request=request,
                success=False,
                error_message="Invalid or missing API key",
            )
        elif response.status_code == 403:
            log_audit_event(
                event_type="authorization_failure",
                request=request,
                success=False,
                error_message="API key not authorized for the requested project",
            )
        elif response.status_code >= 500:
            log_audit_event(
                event_type="internal_error",
                request=request,
                success=False,
                error_message="Internal server error",
            )
        elif request.url.path.startswith("/v1/projects/") and "/index" in request.url.path:
            # Project indexing endpoint
            if response.status_code < 400:
                log_audit_event(
                    event_type="project_indexing_request",
                    request=request,
                    success=True,
                )
            else:
                log_audit_event(
                    event_type="project_indexing_request",
                    request=request,
                    success=False,
                    error_message=f"Request failed with status {response.status_code}",
                )
        elif request.url.path.startswith("/v1/projects/") and "/answer" in request.url.path:
            # Answer endpoint
            if response.status_code < 400:
                log_audit_event(
                    event_type="answer_request",
                    request=request,
                    success=True,
                )
            else:
                log_audit_event(
                    event_type="answer_request",
                    request=request,
                    success=False,
                    error_message=f"Request failed with status {response.status_code}",
                )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to set security headers on responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        settings = get_settings()
        for header, value in settings.security_headers.items():
            response.headers[header] = value
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Middleware to reject requests with bodies larger than the configured limit."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
                if length > get_settings().max_body_size:
                    return Response(
                        content=f"Request body too large: {length} > {get_settings().max_body_size}",
                        status_code=413,
                    )
            except ValueError:
                pass
        return await call_next(request)


def setup_middleware(app) -> None:
    """Set up all middleware for the FastAPI application.

    Middleware order (outermost first): TrustedHost → CORS → GZip →
    MaxBodySize → SecurityHeaders → Logging → Metrics → Tracing →
    RequestID → application.

    Middleware composition in Starlette is LIFO — the LAST added
    runs FIRST on the way in, LAST on the way out. So we add them
    in reverse order to make the call sequence read top-down in
    this function.
    """
    settings = get_settings()

    # Trusted host middleware
    if settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts,
        )

    # CORS middleware
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # GZip middleware
    if settings.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware — order matters: Tracing wraps Metrics wraps Logging
    # so a span covers metric recording AND structured logging for the
    # same request.
    app.add_middleware(RequestIDMiddleware)
    if settings.metrics_enabled:
        app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(TracingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # Max body size middleware
    app.add_middleware(MaxBodySizeMiddleware)