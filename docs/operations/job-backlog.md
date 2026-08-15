# Job Backlog Runbook

## Symptom

- Alert: `SemRAGIndexingJobBacklog` (severity: warning)
- Jobs are queued faster than they complete.
- `semrag_indexing_jobs_queued_total` rate exceeds
  `semrag_indexing_jobs_completed_total` rate over 15+ minutes.
- Users see longer waits between submitting an indexing request and
  seeing the project become queryable.

## Triage

1. Confirm the alert in Grafana. Open the **SemRAG — Indexing**
   dashboard and look at the "Jobs queued vs completed" panel.
2. Check the worker saturation:

   ```bash
   kubectl top pods -n semrag -l component=worker
   ```

3. Look at the per-stage latency in the **SemRAG — Indexing**
   dashboard. A spike in `semrag_indexing_duration_seconds` per
   project indicates the worker is slower than usual — usually
   because the embedding server is overloaded.

## Mitigation

- **Single worker pod is at capacity** — scale up:

  ```bash
  kubectl scale deployment/semrag-api -n semrag --replicas=4
  ```

  (The worker runs in the same process as the API. Scaling the API
  also scales the worker. A future refactor will split them.)

- **Embedding server is the bottleneck** — see
  `embedding-failure.md`. Either fix the embedding server or
  switch to a smaller / faster model.

- **Spike in indexing requests** — if the backlog is from a burst
  of `POST /index` calls, the simplest mitigation is to let the
  worker drain. New requests are still queued, just slower.

- **Stuck job in the queue** — if a single job has been in the
  `RUNNING` state for hours, the worker crashed mid-job without
  marking it failed. Cancel it and the executor will pick up the
  next queued job:

  ```bash
  curl -X DELETE -H "Authorization: Bearer $API_KEY" \
    https://semrag.example.com/v1/projects/<project>/jobs/<stuck_job_id>
  ```

## Verification

1. Queue depth in Prometheus (derived from
   `semrag_indexing_jobs_queued_total - semrag_indexing_jobs_completed_total`)
   stabilises or decreases.
2. New indexing requests complete within the expected wall-clock
   time for the project size.

## Postmortem Notes

- Document the cause (worker saturation, embedding server slowdown,
  or request spike).
- If this happens regularly, consider autoscaling the API / worker
  pods on `semrag_indexing_jobs_queued_total` rate.
