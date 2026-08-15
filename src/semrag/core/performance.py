"""
Performance metrics helpers.

Lightweight wrappers that record per-stage ingestion/retrieval timings
and counts without requiring an external metrics backend. The
`stage_timer` context manager emits one metric per stage via the
existing `Metrics.timer` API so downstream log pipelines see them as
ordinary `metric=...` log lines.

These don't replace the existing `core.observability.Metrics` API — they
thicken it with two phase-5 conveniences:

- `bulk_throughput`: divide a unit count by elapsed seconds, so call
  sites that already have an elapsed value can report chunks/sec without
  re-instantiating a Timer.
- `ingestion_stage_timer`: same as `stage_timer` but tagged with the
  stable stage names the indexing worker uses, so dashboards can
  filter without re-typing the tag set at every call site.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from semrag.core.observability import Metrics


def bulk_throughput(unit_count: int, elapsed_seconds: float, *, name: str, **tags: Any) -> None:
    """Record ``unit_count`` items processed in ``elapsed_seconds`` as a
    throughput metric. Negative or zero elapsed is reported as 0 to
    avoid division-by-zero noise on the hot path."""
    rate = float(unit_count) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    Metrics.count(f"{name}.items", unit_count, **tags)
    Metrics.observe(f"{name}.throughput_per_sec", rate, **tags)


@contextmanager
def ingestion_stage_timer(stage: str, **tags: Any) -> Iterator[dict]:
    """Timer context that automatically tags with the ingestion stage
    name and project_id when supplied via ``tags``."""
    merged = {"stage": stage, **tags}
    with Metrics.timer(f"ingestion.stage.{stage}.duration_ms", **merged) as extras:
        yield extras
