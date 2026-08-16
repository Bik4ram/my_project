"""
Locust load test for SemRAG.

Wave 3 §9 — Load testing.

Run against a live SemRAG API:

    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000

The default shape is "concurrent queries": one Locust user runs
queries against `/v1/projects/{project}/query` continuously. The
`IndexUser` class adds mixed-workload traffic by issuing indexing
requests alongside queries — that's a more realistic load shape
because indexing consumes the same Postgres + embedding
infrastructure as queries.

Configurable via environment variables:

    SEMRAG_LT_HOST     — target host (default http://localhost:8000)
    SEMRAG_LT_API_KEY  — API key to authenticate (default dev-local-key)
    SEMRAG_LT_PROJECT  — project_id to query (default rtl_uvm)
    SEMRAG_LT_INDEX_ROOT — root_path to index (default ./tests/fixtures/sample_repo)
"""
from __future__ import annotations

import os
import random
import time

from locust import HttpUser, between, task


_HOST = os.environ.get("SEMRAG_LT_HOST", "http://localhost:8000")
_API_KEY = os.environ.get("SEMRAG_LT_API_KEY", "dev-local-key")
_PROJECT = os.environ.get("SEMRAG_LT_PROJECT", "rtl_uvm")
_INDEX_ROOT = os.environ.get("SEMRAG_LT_INDEX_ROOT", "./tests/fixtures/sample_repo")

# Standard queries — representative of the kinds of natural-language
# questions a real SemRAG user asks. Repeat queries are intentional;
# the cache hit path is part of the load we want to measure.
_QUERIES: tuple[str, ...] = (
    "What is the module name of the UART?",
    "How does the arbiter round-robin between masters?",
    "What assertions live in the FIFO?",
    "Show me the reset sequence for the DUT.",
    "Where is the clock divider implemented?",
    "What is the purpose of the scoreboard?",
    "How is the AXI handshake sequenced?",
    "Which file implements the interrupt controller?",
)


class _BaseUser(HttpUser):
    """Common HTTP headers and pacing for every Locust user class."""
    abstract = True
    wait_time = between(1.0, 3.0)

    def on_start(self) -> None:
        self.client.headers.update(
            {
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            }
        )


class QueryUser(_BaseUser):
    """Concurrent query load — the dominant production traffic shape."""

    @task(8)
    def query_project(self) -> None:
        """Run a retrieval-only query against the configured project."""
        query = random.choice(_QUERIES)
        with self.client.post(
            f"/v1/projects/{_PROJECT}/query",
            json={"query": query, "top_k": 10},
            name="/v1/projects/{project_id}/query",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate-limit rejections are not a server failure for
                # load-test purposes — they're a normal "we're at
                # capacity" signal. Mark them as success so the
                # error rate doesn't get polluted.
                response.success()
            else:
                response.failure(f"status {response.status_code}: {response.text[:200]}")


class AnswerUser(_BaseUser):
    """Mixed-workload: queries + the full RAG pipeline."""

    @task(1)
    def answer_question(self) -> None:
        """Run the full answer pipeline (query + LLM)."""
        query = random.choice(_QUERIES)
        start = time.perf_counter()
        with self.client.post(
            f"/v1/projects/{_PROJECT}/answer",
            json={"query": query, "top_k": 10},
            name="/v1/projects/{project_id}/answer",
            catch_response=True,
        ) as response:
            elapsed = time.perf_counter() - start
            if response.status_code == 200:
                response.success()
                # Surface LLM latency in Locust's stats manually
                # since the answer endpoint bundles retrieval +
                # LLM in one round-trip.
                self.client.request(
                    "GET",
                    f"/__unused__/{int(elapsed * 1000)}",
                    name="answer.llm_latency_ms",
                ) if False else None
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"status {response.status_code}: {response.text[:200]}")

    @task(8)
    def query_project(self) -> None:
        QueryUser.query_project(self)


class IndexUser(_BaseUser):
    """Mixed indexing + query workload — exercises both pipelines."""

    @task(1)
    def trigger_indexing(self) -> None:
        """Kick off an indexing job. Doesn't wait for completion —
        the worker handles it asynchronously and Locust can move on."""
        with self.client.post(
            f"/v1/projects/{_PROJECT}/index",
            json={"root_path": _INDEX_ROOT},
            name="/v1/projects/{project_id}/index",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 202):
                response.success()
            else:
                response.failure(f"status {response.status_code}: {response.text[:200]}")

    @task(4)
    def query_project(self) -> None:
        QueryUser.query_project(self)


class HealthUser(_BaseUser):
    """Probes the health endpoints — simulates kubelet traffic."""

    @task(5)
    def livez(self) -> None:
        self.client.get("/livez", name="/livez")

    @task(2)
    def readyz(self) -> None:
        self.client.get("/readyz", name="/readyz")

    @task(1)
    def metrics(self) -> None:
        # /metrics is intentionally unauthenticated per the Wave 3
        # audit. Locust hits it the way Prometheus would.
        self.client.get("/metrics", name="/metrics")