"""Observability primitives — TRACE level, stage context manager, metrics."""
from __future__ import annotations

import json
import logging

import pytest

from semrag.core.observability import (
    TRACE_LEVEL,
    configure_logging,
    count,
    get_logger,
    is_debug_enabled,
    is_trace_enabled,
    stage,
    timer,
)


@pytest.fixture
def captured_log():
    """Attach a memory handler to the root logger and return the records."""
    records: list[logging.LogRecord] = []

    class _List(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _List()
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield records
    root.removeHandler(handler)


def test_trace_level_is_below_debug():
    assert TRACE_LEVEL < logging.DEBUG


def test_is_debug_and_trace_helpers():
    assert is_debug_enabled("DEBUG") is True
    assert is_debug_enabled("TRACE") is True
    assert is_debug_enabled("INFO") is False
    assert is_trace_enabled("TRACE") is True
    assert is_trace_enabled("DEBUG") is False
    assert is_trace_enabled("INFO") is False


def test_timer_emits_metric_record(captured_log):
    with timer("test.op", tag="x"):
        pass
    metric_records = [r for r in captured_log if getattr(r, "metric", None) == "test.op"]
    assert metric_records, "expected at least one metric record"
    assert metric_records[0].value > 0
    assert metric_records[0].tags.get("tag") == "x"


def test_count_emits_metric_record(captured_log):
    count("test.counter", 3, kind="widget")
    metric_records = [r for r in captured_log if getattr(r, "metric", None) == "test.counter"]
    assert metric_records[0].value == 3
    assert metric_records[0].tags["kind"] == "widget"


def test_stage_emits_start_and_finish(captured_log):
    log = get_logger("test.stage")
    with stage("op", logger=log, project_id="p1") as stats:
        stats["things"] = 9
    stage_records = [r for r in captured_log if getattr(r, "stage", None) == "op"]
    events = [r.event for r in stage_records]
    assert "start" in events
    assert "finish" in events
    finish = next(r for r in stage_records if r.event == "finish")
    assert finish.status == "ok"
    assert finish.elapsed_ms >= 0
    assert finish.things == 9


def test_stage_records_error_status_on_exception(captured_log):
    log = get_logger("test.stage.err")
    with pytest.raises(RuntimeError):
        with stage("op2", logger=log):
            raise RuntimeError("boom")
    stage_records = [r for r in captured_log if getattr(r, "stage", None) == "op2"]
    finish = next(r for r in stage_records if r.event == "finish")
    assert finish.status == "error"


def test_json_formatter_has_stable_schema(captured_log):
    settings = type("S", (), {"log_level": "DEBUG", "log_format": "json"})()
    configure_logging(settings)
    log = get_logger("test.json")
    log.info("hello", extra={"project_id": "p1", "stage": "x"})
    # The captured_log records are live; render through the formatter the
    # way a real handler would.
    from semrag.core.observability import _JsonFormatter
    formatted = _JsonFormatter().format(captured_log[-1])
    payload = json.loads(formatted)
    assert payload["level"] in ("DEBUG", "INFO", "ERROR", "WARNING", "CRITICAL", "TRACE")
    assert "ts" in payload
    assert "message" in payload
    assert payload["project_id"] == "p1"
    assert payload["stage"] == "x"
