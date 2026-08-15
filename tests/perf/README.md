# Performance regression suite

Repeatable benchmarks for the Wave 3 audit. The script measures
retrieval latency, answer latency, indexing throughput, cache hit
rate, and peak memory, and emits JSON.

Two modes:

- **In-process** (default) — runs via FastAPI TestClient with TFIDF
  embeddings and a mock LLM. No live services needed. Use for
  catching regressions in CI.

  ```bash
  python tests/perf/run_perf_suite.py --output bench.json
  ```

- **Live URL** — runs against a real SemRAG API. Use for SLO
  measurement and absolute performance baselines.

  ```bash
  python tests/perf/run_perf_suite.py --url http://localhost:8000 \
      --api-key dev-local-key --project rtl_uvm --output bench.json
  ```

The JSON output is shaped for direct ingestion into a benchmarking
dashboard — top-level keys map to the Wave 3 audit's required
metrics.