"""Tests for the performance metrics helpers."""
from __future__ import annotations

import logging
import time

from semrag.core.observability import Metrics
from semrag.core.performance import bulk_throughput, ingestion_stage_timer


def _collect_metrics(fn) -> list[logging.LogRecord]:
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("semrag.metrics")
    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate
    return records


def test_bulk_throughput_emits_count_and_rate():
    records = _collect_metrics(lambda: bulk_throughput(120, 2.0, name="ingest"))
    names = [getattr(r, "metric", None) for r in records]
    assert "ingest.items" in names
    assert "ingest.throughput_per_sec" in names


def test_bulk_throughput_handles_zero_elapsed_without_division_by_zero():
    records = _collect_metrics(lambda: bulk_throughput(50, 0.0, name="ingest"))
    rate_records = [r for r in records if getattr(r, "metric", None) == "ingest.throughput_per_sec"]
    assert rate_records
    assert rate_records[0].value == 0.0


def test_ingestion_stage_timer_emits_with_stage_tag():
    captured = {}

    def runner():
        with ingestion_stage_timer("chunking", project_id="p1") as extras:
            extras["chunk_count"] = 100
            time.sleep(0.001)
        captured["extras"] = extras

    records = _collect_metrics(runner)
    finish = [r for r in records if getattr(r, "metric", None) == "ingestion.stage.chunking.duration_ms"]
    assert finish
    assert finish[0].tags.get("stage") == "chunking"
    assert finish[0].tags.get("project_id") == "p1"
    assert finish[0].tags.get("chunk_count") == 100
    assert captured["extras"]["chunk_count"] == 100
