# High Latency Runbook

## Symptom

- Alert: `SemRAGHighRequestLatencyP95` (severity: warning)
- Alert: `SemRAGHighLLMLatencyP95` (severity: warning)
- Alert: `SemRAGHighEmbeddingLatencyP95` (severity: warning)
- User reports of slow responses, often without errors.

## Triage

1. Open the **SemRAG — API** dashboard. Look at the latency panels.
   The breakdown by `path` and `status` will pinpoint the slow
   endpoint.
2. If HTTP latency is up but LLM and embedding are flat, the
   bottleneck is in the API itself (DB query, JSON serialization,
   etc.). Look at the **SemRAG — Database** dashboard.
3. If LLM latency is up, the upstream LLM server is the bottleneck.
   See `llm-outage.md` for upstream checks.
4. If embedding latency is up, the embedding server is the
   bottleneck. See `embedding-failure.md` for upstream checks.

## Mitigation

- **LLM is slow** — reduce per-request cost by lowering
  `SEMRAG_LLM_NUM_PREDICT` (default 800). 200-400 tokens is
  enough for most answers; the high default was set for
  exploratory queries.
- **Embedding is slow** — the embedding provider is doing more
  work than necessary. Check
  `semrag_indexing_chunks_embedded_total` — a spike in chunks
  embedded per minute suggests a large indexing job is running,
  which is a legitimate cause. If not, the embedding server may
  be unhealthy.
- **Database is slow** — check
  `histogram_quantile(0.95, semrag_db_query_duration_seconds_bucket)`
  for the slow query type. Common culprits:
  - vector search over a large index (tune
    `SEMRAG_FINAL_TOP_K` lower if you don't need full precision)
  - large Postgres transactions (commit time spikes; usually a
    long-running indexing job)
- **API is at CPU saturation** — scale the API pods up:

  ```bash
  kubectl scale deployment/semrag-api -n semrag --replicas=4
  ```

## Verification

1. P95 latency on the slow endpoint drops back to baseline (under
   2 seconds for `/query`, under 30 seconds for `/answer`).
2. Alert clears within 5-10 minutes.

## Postmortem Notes

- Capture the slow stage (HTTP / LLM / embedding / DB) and the
  mitigation that worked.
- Update the alert thresholds if the baseline has permanently
  shifted.
