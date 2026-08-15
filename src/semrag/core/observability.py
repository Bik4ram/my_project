"""
Structured logging and metrics primitives.

Two outputs coexist here deliberately:

1. `logging.getLogger(__name__)` is the project's ordinary logger. Every
   module uses it instead of `print()`. Format is controlled here so
   `Settings.log_format` decides whether records become JSON lines
   (production, shippable to a backend) or a human-friendly colored
   stream (development). One-place format swap.

2. `Metrics` emits one structured event per measurement (counter, timing,
   tag set) through the same logger at INFO level with a stable prefix
   (`metric=`) so a downstream log shipper can split metric events from
   ordinary log lines without parsing free-form text. Counter increments
   and latency observations both produce one such event.

Why both: `logging` gives operators the conventional log story (levels,
handlers, library interop with uvicorn/celery). Emitting metrics as log
records keeps the system deployable without a separate metrics backend
while still being parseable into Prometheus / OpenTelemetry later by any
log pipeline. When a Prometheus exporter is added, the only change here
will be swapping the metric handler from "emit a log line" to "call
prometheus_client.Counter.inc()" — the call sites stay the same.

Public surface intentionally small:

- `configure_logging(settings) -> None` — call once at startup.
- `Timer` — context manager that records wall-clock duration into a metrics
  event with a named operation and tags. Use for retrieval, embedding,
  LLM, tool execution, query timings — the audit explicitly asks for each
  of these.
- `Metrics` — `count(name, value=1, **tags)` and `observe(name, value, **tags)`
  for histograms/latencies without a Timer block.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

from semrag.core.redaction import redact_payload
from semrag.core.settings import Settings


# A dedicated logger keeps metric and event lines channeled separately so
# a log shipper can route them; the broader project logger surface stays
# in the root/uvicorn namespace.
_METRICS_LOGGER_NAME = "semrag.metrics"

# Project-specific TRACE level sits below DEBUG (5), below NOTSET (0).
# Python's stdlib logging doesn't define it; register one so that
# `Settings.log_level = "TRACE"` and `logger.log(TRACE_LEVEL, ...)` both
# work without per-call integer arithmetic at the call site.
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def is_debug_enabled(level: str) -> bool:
    return level in ("DEBUG", "TRACE")


def is_trace_enabled(level: str) -> bool:
    return level == "TRACE"


class _JsonFormatter(logging.Formatter):
    """Stable-shape JSON line per record. Keys never change without a
    major version bump — downstream log pipelines depend on the schema.

    Wave 3 additions:

    - Always includes ``trace_id`` and ``span_id`` (zeroed when no
      span is active). Operators searching logs by trace no longer
      need a side-channel to correlate log lines with an OTel trace.
    - Secrets are redacted automatically — every key in the
      ``LogRecord.__dict__`` extension goes through
      ``redact_payload`` before being serialised. API keys, tokens,
      passwords, and connection strings never leave the process.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Local import keeps the formatting path import-cheap when the
        # OTel SDK isn't installed (the helper handles a missing SDK
        # internally by returning zeroed ids).
        try:
            from semrag.core.tracing import current_span_id_hex, current_trace_id_hex
        except ImportError:
            current_trace_id_hex = lambda: "0" * 32  # noqa: E731
            current_span_id_hex = lambda: "0" * 16  # noqa: E731

        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": current_trace_id_hex(),
            "span_id": current_span_id_hex(),
        }
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                # Test JSON-serialisability before committing the field;
                # non-serialisable values fall back to repr() rather than
                # raising mid-format (which would drop the entire record).
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        payload = redact_payload(payload)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _HumanFormatter(logging.Formatter):
    DEFAULT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.DEFAULT, datefmt="%H:%M:%S")


_configured = False


def configure_logging(settings: Settings) -> None:
    """Install handlers/formatters exactly once. Idempotent — calling twice
    does not stack handlers."""
    global _configured
    if _configured:
        return

    # stdlib's setLevel can be passed an int or a level-name string. For
    # project-local levels like "TRACE" we resolve the numeric value
    # ourselves so a misconfigured deploy fails at startup.
    numeric = logging.getLevelName(settings.log_level)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {settings.log_level!r}")

    root = logging.getLogger()
    root.setLevel(numeric)

    formatter: logging.Formatter
    if settings.log_format == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _HumanFormatter()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)
    # Replace any pre-existing handlers (uvicorn installs its own) so the
    # project's chosen format always wins for our own loggers.
    handler.setLevel(numeric)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Standard library loggers stay at WARNING; the project surfaces at
    # whatever level the operator configures, but library spammy logs
    # (e.g. urllib3 at INFO) don't drown the output.
    for noisy in ("urllib3", "httpx", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Standard project idiom:
        `logger = get_logger(__name__)`
    Always returns a usable logger; configuration happens at startup."""
    return logging.getLogger(name)


def _metrics_logger() -> logging.Logger:
    return logging.getLogger(_METRICS_LOGGER_NAME)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Lightweight metrics facade. Each call emits exactly one log record
    on the metrics logger, formatted `metric=<name> value=<v> tags=<k=v>...`.
    Library-free; ship the metrics logger to whatever backend the operator
    runs (Promtail, Vector, Fluent Bit)."""

    @staticmethod
    def count(name: str, value: int | float = 1, **tags: Any) -> None:
        _emit_metric(name, value, tags)

    @staticmethod
    def observe(name: str, value: int | float, **tags: Any) -> None:
        _emit_metric(name, value, tags)

    @staticmethod
    @contextmanager
    def timer(name: str, **tags: Any) -> Iterator[dict]:
        """Record elapsed milliseconds into the metric `name` when the
        block exits. Yields a mutable dict the caller can populate with
        extra tags that aren't known up-front (e.g. hit_count)."""
        extras: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield extras
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            merged = {**tags, **extras}
            _emit_metric(name, elapsed_ms, merged)


def _emit_metric(name: str, value: int | float, tags: dict[str, Any]) -> None:
    msg = f"metric={name} value={value}"
    if tags:
        msg += " " + " ".join(f"{k}={_format_tag(v)}" for k, v in tags.items())
    _metrics_logger().info(msg, extra={"metric": name, "value": value, "tags": tags})


def _format_tag(value: Any) -> str:
    if isinstance(value, float):
        # ms latency typically; keep 3 decimals to avoid stream bloat.
        return f"{value:.3f}"
    return str(value)


# Convenience aliases — the audit calls these out by name so they are
# used consistently across modules.
def timer(name: str, **tags: Any):
    return Metrics.timer(name, **tags)


def count(name: str, value: int | float = 1, **tags: Any) -> None:
    Metrics.count(name, value, **tags)


def observe(name: str, value: int | float, **tags: Any) -> None:
    Metrics.observe(name, value, **tags)


# ---------------------------------------------------------------------------
# Per-stage structured logging
# ---------------------------------------------------------------------------

class StageStats:
    """Mutable bag the `stage` context manager merges into the finish line.
    Convention: callers populate `stage_stats("chunk_count", 12)` etc.
    throughout the block; the manager emits them as structured fields on the
    finish log line."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


@contextmanager
def stage(name: str, *, logger: logging.Logger, level: str = "INFO", **tags: Any) -> Iterator[StageStats]:
    """Emit `start` and `finish` events around a unit of work, plus an
    `elapsed_ms` counter on finish. Failure status is logged on raise but
    the exception propagates. Tags propagate to both events so log
    pipelines can filter on project_id without joining two records."""
    stats = StageStats()
    start = time.perf_counter()
    _log_stage(logger, level, "start", name, tags, stats=stats, elapsed_ms=None)
    status = "ok"
    try:
        yield stats
    except BaseException as exc:
        status = "error"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log_stage(logger, "ERROR", "finish", name, tags, stats=stats, elapsed_ms=elapsed_ms, status="error", exc=exc)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _log_stage(logger, level, "finish", name, tags, stats=stats, elapsed_ms=elapsed_ms, status=status)


def _log_stage(
    logger: logging.Logger,
    level: str,
    event: str,
    name: str,
    tags: dict[str, Any],
    *,
    stats: StageStats,
    elapsed_ms: float | None,
    status: str = "ok",
    exc: BaseException | None = None,
) -> None:
    extra: dict[str, Any] = {
        "stage": name,
        "event": event,
        "status": status,
        **{k: v for k, v in tags.items()},
    }
    if elapsed_ms is not None:
        extra["elapsed_ms"] = round(elapsed_ms, 3)
    extra.update(stats.as_dict())
    msg = f"stage={name} event={event} status={status}"
    if elapsed_ms is not None:
        msg += f" elapsed_ms={elapsed_ms:.3f}"
    for k, v in stats.as_dict().items():
        msg += f" {k}={v}"
    if exc is not None:
        msg += f" error={exc.__class__.__name__}: {exc}"
    if level == "TRACE":
        logger.log(TRACE_LEVEL, msg, extra=extra)
    elif level == "DEBUG":
        logger.debug(msg, extra=extra)
    elif level == "ERROR":
        logger.error(msg, extra=extra)
    elif level == "WARNING":
        logger.warning(msg, extra=extra)
    else:
        logger.info(msg, extra=extra)


def trace(logger: logging.Logger, msg: str, **extra: Any) -> None:
    """Convenience wrapper — keeps call sites shorter than
    `logger.log(TRACE_LEVEL, ...)`. Use the DEBUG/TRACE check at the
    call site if the message construction is expensive."""
    logger.log(TRACE_LEVEL, msg, extra=extra)


def debug(logger: logging.Logger, msg: str, **extra: Any) -> None:
    logger.debug(msg, extra=extra)
