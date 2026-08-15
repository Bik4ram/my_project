# Indexing Failure Runbook

## Symptom

- Alert: `SemRAGIndexingJobsFailing` (severity: warning)
- Jobs in the queue are ending in the `failed` state.
- The `/v1/projects/{project_id}/jobs/{job_id}` endpoint shows
  `state: failed` and a non-null `error` field.
- Logs contain `Indexing job failed` with an exception class and
  message.

## Triage

1. Identify the failing job(s):

   ```bash
   curl -s -H "Authorization: Bearer $API_KEY" \
     https://semrag.example.com/v1/projects/<project>/jobs | \
     jq '.jobs[] | select(.state == "failed")'
   ```

2. Look at the job's `error` field. Common patterns:

   - `NotADirectoryError` — the project root doesn't exist or isn't
     accessible from the API pod. Check the
     `SEMRAG_PROJECT_ROOTS` allowlist.
   - `EmbeddingProviderError` — see `embedding-failure.md`.
   - `psycopg2.OperationalError` — see `database-outage.md`.
   - `PermissionError` — the API pod can't read the project root.
     Common when the root is on a hostPath volume and the pod is
     running as a non-root user.

3. Look at the indexing logs in the API pod to find the stack
   trace:

   ```bash
   kubectl logs -n semrag deploy/semrag-api --tail=200 | \
     grep "Indexing job failed"
   ```

## Mitigation

- **Single job failed** — retry the job:

  ```bash
  curl -X POST -H "Authorization: Bearer $API_KEY" \
    https://semrag.example.com/v1/projects/<project>/jobs/<job_id>/retry
  ```

- **Repeated failures with the same error** — fix the underlying
  cause (see specific runbooks above) before retrying, otherwise
  the retry will fail in the same way.

- **Mass failures (multiple projects)** — usually a downstream
  outage (Postgres, embedding server). Address the underlying issue
  first; retries will continue to fail until then.

## Verification

1. A retry of the failing job runs to completion.
2. `semrag_indexing_jobs_failed_total` rate drops to baseline in
   Prometheus within one scrape interval.

## Postmortem Notes

- Note the failing stage (scan / parse / IR / chunk / embed /
  persist). Each stage has a distinct failure mode.
- If failures are concentrated in a single stage across multiple
  projects, that's almost always a downstream service (embedding
  provider for `embedding`, Postgres for `persist`).
