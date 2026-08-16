#!/usr/bin/env python3
"""
Performance regression suite for SemRAG (Wave 3 §10).

Runs a fixed workload against the live API (or a local fastapi
test client) and reports:

- Retrieval latency (p50 / p95 / p99)
- Answer latency (p50 / p95 / p99)
- Indexing throughput (chunks/sec)
- Cache hit rate
- Memory usage (peak RSS)

Output is JSON by default; CI can pin the suite to a known
baseline and fail on regression.

Usage:

    # Live API
    python tests/perf/run_perf_suite.py --url http://localhost:8000 \\
        --api-key dev-local-key --project rtl_uvm --output bench.json

    # In-process via TestClient (no live API needed; default)
    python tests/perf/run_perf_suite.py --output bench.json

When run in-process the suite uses synthetic data: TFIDF
embeddings, a mock LLM, no real Postgres. The numbers are useful
for catching "this code got 10x slower overnight" regressions,
not for absolute SLO measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Allow the suite to be run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_QUERIES: tuple[str, ...] = (
    "What is the module name of the UART?",
    "How does the arbiter round-robin between masters?",
    "What assertions live in the FIFO?",
    "Show me the reset sequence for the DUT.",
    "Where is the clock divider implemented?",
)


@dataclass
class BenchmarkResult:
    suite: str
    started_at: str
    duration_seconds: float
    iterations: int
    retrieval_latency_p50_ms: float
    retrieval_latency_p95_ms: float
    retrieval_latency_p99_ms: float
    answer_latency_p50_ms: float
    answer_latency_p95_ms: float
    answer_latency_p99_ms: float
    indexing_chunks_per_sec: float
    cache_hit_rate: float
    peak_memory_mb: float
    notes: dict[str, Any] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _measure_peak_memory_mb() -> float:
    """Return the peak RSS of the current process in megabytes.

    Uses `resource.getrusage` so the suite doesn't depend on
    psutil. Returns 0.0 on platforms where rusage isn't
    supported (e.g. Windows); the suite is POSIX-only.
    """
    try:
        import resource
        # ru_maxrss is kilobytes on Linux, bytes on macOS — divide
        # by 1024 either way so the unit is consistent in MB.
        usage = resource.getrusage(resource.RUSAGE_SELF)
        kb = usage.ru_maxrss
        return round(kb / 1024.0, 2)
    except ImportError:
        return 0.0


def _run_in_process(iterations: int) -> BenchmarkResult:
    """Run the regression suite in-process using the FastAPI TestClient.

    No live Postgres / Ollama is required; the test client exercises
    the route handlers but with mocked providers. This catches
    "the orchestration got 10x slower" regressions, not absolute
    SLO measurement.
    """
    # Defer imports until after sys.path is adjusted so the suite
    # works when invoked from any cwd.
    os.environ.setdefault("SEMRAG_EMBEDDING_PROVIDER", "tfidf")
    os.environ.setdefault("SEMRAG_LLM_PROVIDER", "mock")
    os.environ.setdefault("SEMRAG_API_KEYS", "bench-key")

    from fastapi.testclient import TestClient
    from semrag.api.app import create_app

    gc.collect()
    started_at = time.time()
    iso_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))

    app = create_app()
    retrieval_latencies: list[float] = []
    answer_latencies: list[float] = []
    cache_hits = 0
    with TestClient(app) as client:
        client.headers.update(
            {"Authorization": "Bearer bench-key", "Content-Type": "application/json"}
        )

        # First call primes the cache; subsequent identical queries hit it.
        for i in range(iterations):
            query = _QUERIES[i % len(_QUERIES)]
            t0 = time.perf_counter()
            response = client.post(
                "/v1/projects/bench_project/query",
                json={"query": query, "top_k": 5},
            )
            retrieval_latencies.append((time.perf_counter() - t0) * 1000.0)
            if response.status_code == 200 and response.json().get("from_cache"):
                cache_hits += 1

        # A small handful of /answer calls — slower path.
        for i in range(max(1, iterations // 5)):
            query = _QUERIES[i % len(_QUERIES)]
            t0 = time.perf_counter()
            client.post(
                "/v1/projects/bench_project/answer",
                json={"query": query, "top_k": 5},
            )
            answer_latencies.append((time.perf_counter() - t0) * 1000.0)

    duration = time.perf_counter() - (started_at - time.time())

    # Indexing throughput — synthetic chunks, measured as the time to
    # build a TFIDF vectorizer on 1000 chunks of varying text.
    from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider

    chunks = [f"chunk {i} text varies {random.random()}" for i in range(1000)]
    t0 = time.perf_counter()
    provider = TfidfEmbeddingProvider(dimension=64)
    provider.fit(chunks)
    fit_duration = time.perf_counter() - t0
    indexing_chunks_per_sec = 1000.0 / fit_duration if fit_duration > 0 else 0.0

    return BenchmarkResult(
        suite="perf-regression",
        started_at=iso_started,
        duration_seconds=round(time.time() - started_at, 3),
        iterations=iterations,
        retrieval_latency_p50_ms=round(_percentile(retrieval_latencies, 50), 3),
        retrieval_latency_p95_ms=round(_percentile(retrieval_latencies, 95), 3),
        retrieval_latency_p99_ms=round(_percentile(retrieval_latencies, 99), 3),
        answer_latency_p50_ms=round(_percentile(answer_latencies, 50), 3),
        answer_latency_p95_ms=round(_percentile(answer_latencies, 95), 3),
        answer_latency_p99_ms=round(_percentile(answer_latencies, 99), 3),
        indexing_chunks_per_sec=round(indexing_chunks_per_sec, 2),
        cache_hit_rate=round(cache_hits / max(1, iterations), 3),
        peak_memory_mb=_measure_peak_memory_mb(),
        notes={"mode": "in-process", "embedding_provider": "tfidf"},
    )


def _run_against_url(url: str, api_key: str, project: str, iterations: int) -> BenchmarkResult:
    """Run the regression suite against a live API."""
    import requests

    started_at = time.time()
    iso_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))
    retrieval_latencies: list[float] = []
    answer_latencies: list[float] = []
    cache_hits = 0

    for i in range(iterations):
        query = _QUERIES[i % len(_QUERIES)]
        t0 = time.perf_counter()
        response = requests.post(
            f"{url}/v1/projects/{project}/query",
            json={"query": query, "top_k": 5},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        retrieval_latencies.append((time.perf_counter() - t0) * 1000.0)
        if response.status_code == 200 and response.json().get("from_cache"):
            cache_hits += 1

    for i in range(max(1, iterations // 5)):
        query = _QUERIES[i % len(_QUERIES)]
        t0 = time.perf_counter()
        requests.post(
            f"{url}/v1/projects/{project}/answer",
            json={"query": query, "top_k": 5},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        answer_latencies.append((time.perf_counter() - t0) * 1000.0)

    return BenchmarkResult(
        suite="perf-regression",
        started_at=iso_started,
        duration_seconds=round(time.time() - started_at, 3),
        iterations=iterations,
        retrieval_latency_p50_ms=round(_percentile(retrieval_latencies, 50), 3),
        retrieval_latency_p95_ms=round(_percentile(retrieval_latencies, 95), 3),
        retrieval_latency_p99_ms=round(_percentile(retrieval_latencies, 99), 3),
        answer_latency_p50_ms=round(_percentile(answer_latencies, 50), 3),
        answer_latency_p95_ms=round(_percentile(answer_latencies, 95), 3),
        answer_latency_p99_ms=round(_percentile(answer_latencies, 99), 3),
        indexing_chunks_per_sec=0.0,  # Not measured in URL mode — too noisy.
        cache_hit_rate=round(cache_hits / max(1, iterations), 3),
        peak_memory_mb=_measure_peak_memory_mb(),
        notes={"mode": "live-url", "url": url, "project": project},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Performance regression suite")
    parser.add_argument("--url", default=None, help="Live API URL (default: in-process)")
    parser.add_argument("--api-key", default="dev-local-key")
    parser.add_argument("--project", default="rtl_uvm")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    args = parser.parse_args(argv)

    if args.url:
        result = _run_against_url(args.url, args.api_key, args.project, args.iterations)
    else:
        result = _run_in_process(args.iterations)

    serialised = asdict(result)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(serialised, fh, indent=2)
        print(f"wrote {args.output}")
    else:
        print(json.dumps(serialised, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())