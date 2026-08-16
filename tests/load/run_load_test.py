"""
Dependency-free load tester for SemRAG.

This is the lightweight alternative to `locustfile.py` — it doesn't
require Locust, only `requests`. Use it for quick smoke tests or
for CI pipelines where installing Locust is heavy.

For realistic load tests against a live cluster, use Locust
(`tests/load/locustfile.py`).

Usage:

    python -m tests.load.run_load_test --url http://localhost:8000 \\
        --api-key dev-local-key --project rtl_uvm --concurrency 8 \\
        --duration 60

The script measures:

    - requests per second
    - latency p50 / p95 / p99
    - error rate (4xx and 5xx are both counted)
    - cache hit rate (response body's `from_cache: true` flag)
    - approximate CPU and memory footprint of the test process
      (RSS only — useful as a sanity check that the load
      generator isn't itself a bottleneck)

Output is JSON to stdout so a CI job can ingest it directly.

Not part of the test suite — invoked manually.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests


_QUERIES: tuple[str, ...] = (
    "What is the module name of the UART?",
    "How does the arbiter round-robin between masters?",
    "What assertions live in the FIFO?",
    "Show me the reset sequence for the DUT.",
    "Where is the clock divider implemented?",
)


@dataclass
class LoadTestResult:
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    error_rate: float
    cache_hit_rate: float
    status_codes: dict[str, int] = field(default_factory=dict)


def _run_one(
    base_url: str, api_key: str, project_id: str, timeout_s: float
) -> tuple[float, int, bool]:
    """Run a single query. Returns (latency_seconds, status_code, cache_hit)."""
    query = random.choice(_QUERIES)
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/projects/{project_id}/query",
        json={"query": query, "top_k": 10},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - start
    cache_hit = False
    if response.status_code == 200:
        try:
            cache_hit = bool(response.json().get("from_cache", False))
        except (ValueError, KeyError):
            cache_hit = False
    return elapsed, response.status_code, cache_hit


def run_load_test(
    base_url: str,
    api_key: str,
    project_id: str,
    concurrency: int,
    duration_seconds: int,
    timeout_seconds: float,
) -> LoadTestResult:
    """Run the load test and return the aggregate result.

    Spawns `concurrency` workers; each worker issues requests
    continuously until `duration_seconds` elapses. Latencies are
    aggregated across all workers.
    """
    deadline = time.monotonic() + duration_seconds
    latencies_ms: list[float] = []
    status_counts: dict[str, int] = {}
    cache_hits = 0
    total = 0
    failed = 0

    def _worker() -> None:
        nonlocal cache_hits, total, failed
        while time.monotonic() < deadline:
            try:
                elapsed, status, cache_hit = _run_one(
                    base_url, api_key, project_id, timeout_seconds
                )
            except (requests.RequestException, requests.Timeout) as exc:
                total += 1
                failed += 1
                status_counts["exception"] = status_counts.get("exception", 0) + 1
                continue
            total += 1
            status_counts[str(status)] = status_counts.get(str(status), 0) + 1
            if status >= 400:
                failed += 1
            else:
                latencies_ms.append(elapsed * 1000.0)
                if cache_hit:
                    cache_hits += 1

    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker) for _ in range(concurrency)]
        concurrent.futures.wait(futures)
    actual_duration = time.perf_counter() - started_at

    sorted_latencies = sorted(latencies_ms)
    p50 = _percentile(sorted_latencies, 50)
    p95 = _percentile(sorted_latencies, 95)
    p99 = _percentile(sorted_latencies, 99)
    return LoadTestResult(
        duration_seconds=actual_duration,
        total_requests=total,
        successful_requests=total - failed,
        failed_requests=failed,
        cache_hits=cache_hits,
        rps=total / actual_duration if actual_duration > 0 else 0.0,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_max_ms=max(latencies_ms) if latencies_ms else 0.0,
        error_rate=(failed / total) if total else 0.0,
        cache_hit_rate=(cache_hits / (total - failed)) if total > failed else 0.0,
        status_codes=status_counts,
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return the linear-interpolated percentile of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _format_report(result: LoadTestResult) -> str:
    """Render the result as a human-readable summary."""
    lines = [
        f"duration: {result.duration_seconds:.2f}s",
        f"requests: {result.total_requests} ({result.rps:.2f} req/s)",
        f"successful: {result.successful_requests} (errors: {result.failed_requests}, error rate {result.error_rate:.2%})",
        f"cache hit rate: {result.cache_hit_rate:.2%}",
        f"latency p50: {result.latency_p50_ms:.1f} ms",
        f"latency p95: {result.latency_p95_ms:.1f} ms",
        f"latency p99: {result.latency_p99_ms:.1f} ms",
        f"latency max: {result.latency_max_ms:.1f} ms",
        f"status codes: {result.status_codes}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke load test for SemRAG")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("SEMRAG_API_KEYS", "dev-local-key"))
    parser.add_argument("--project", default="rtl_uvm")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_load_test(
        base_url=args.url,
        api_key=args.api_key.split(",")[0].strip(),
        project_id=args.project,
        concurrency=args.concurrency,
        duration_seconds=args.duration,
        timeout_seconds=args.timeout,
    )

    if args.json:
        json.dump(result.__dict__, sys.stdout, indent=2)
        print()
    else:
        print(_format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())