# Database Outage Runbook

## Symptom

- Alert: `SemRAGDatabaseUnavailable` (severity: critical)
- `/readyz` returns HTTP 503 with `checks.database` showing
  `OperationalError`, `connection refused`, or `pgvector extension not installed`.
- HTTP requests hang or return 5xx for any endpoint that touches the
  retrieval path (`/v1/projects/.../query`, `/v1/projects/.../answer`).
- Logs at WARNING/ERROR level contain `database check failed` lines.

## Triage

1. Confirm the alert by hitting `/readyz` directly on one pod:

   ```bash
   curl -s https://semrag.example.com/readyz | jq .
   ```

   Look at `checks.database` — if it says `ok`, the alert may be stale
   or the database is recovering. If it says anything else, proceed.

2. Check Postgres health:

   ```bash
   kubectl exec -n semrag deploy/postgres -- pg_isready
   kubectl logs -n semrag deploy/postgres --tail=100
   ```

3. Check connection from the API pod:

   ```bash
   kubectl exec -n semrag deploy/semrag-api -- \
     python -c "from sqlalchemy import create_engine, text; \
       e = create_engine('<your database url>'); \
       print(e.connect().execute(text('SELECT 1')).scalar())"
   ```

4. Confirm the alert isn't a network issue between pods by checking
   the `semrag_health_up{component="database"}` gauge in Prometheus.
   A value of 0 across multiple pods indicates a real outage; a value
   of 0 on a single pod may indicate that pod's network is broken.

## Mitigation

- **Postgres is down** — restart the StatefulSet:
  `kubectl rollout restart statefulset/postgres -n semrag`.
  Confirm with `kubectl get pods -n semrag` that the new pod is Ready.

- **pgvector extension missing** — common after a fresh database
  restore on a cluster that didn't run the init script. Run the
  migration:

  ```bash
  kubectl exec -n semrag deploy/postgres -- psql -U semrag -d semrag -c \
    "CREATE EXTENSION IF NOT EXISTS vector;"
  ```

- **Disk full on the Postgres data volume** — see the
  `memory-exhaustion.md` runbook. Postgres will refuse new
  connections long before disk fills; check
  `node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql"}`
  in Prometheus.

- **Postgres is fine but the API can't reach it** — check the
  `SEMRAG_DATABASE_URL` env var on the API pod. Wrong credentials or
  wrong host are common typos after a configmap rotation:

  ```bash
  kubectl exec -n semrag deploy/semrag-api -- env | grep SEMRAG_DATABASE_URL
  ```

## Verification

1. `/readyz` returns HTTP 200.
2. A test query returns 200:

   ```bash
   curl -X POST https://semrag.example.com/v1/projects/<project>/query \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "smoke test"}'
   ```

3. The `semrag_health_up{component="database"}` gauge returns to 1
   in Prometheus within one scrape interval (~15 seconds).

## Postmortem Notes

- Capture the timeline from the alert timestamp to the
  resolution. Note any errors in the postgres logs.
- Confirm that the connection pool didn't fill up while Postgres
  was unreachable. Check
  `semrag_db_pool_connections{state="active"}` in Prometheus.
