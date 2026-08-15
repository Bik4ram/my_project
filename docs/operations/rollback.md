# Rollback Guide

## When to Roll Back

Roll back a SemRAG release when:

- A new release introduces a regression that cannot be fixed
  quickly (within the alert window).
- An alert fires immediately after a rollout that wasn't firing
  on the previous version.
- A user-facing feature is broken in the new release.

When in doubt, roll back. A safe rollback is faster than a
forensic postmortem in the middle of an incident.

## How to Roll Back (Helm)

```bash
# 1. List release history.
helm history semrag -n semrag

# 2. Roll back to the previous revision.
helm rollback semrag -n semrag

# 3. Verify the rollback.
kubectl rollout status deployment/semrag-api -n semrag
curl -s https://semrag.example.com/version
```

`/version` returns the deployed version. Confirm it matches the
expected previous version.

## How to Roll Back (Kubernetes)

If you don't use Helm:

```bash
kubectl rollout undo deployment/semrag-api -n semrag
kubectl rollout status deployment/semrag-api -n semrag
```

## Rolling Back the Database Schema

The Postgres schema is managed by Alembic. **Do NOT roll back the
schema without a corresponding code rollback** — older code may
not understand the new schema columns and may crash at runtime.

If you must roll back the schema:

```bash
# 1. Roll back the application code first (Helm rollback above).
# 2. Roll back the schema.
kubectl exec -n semrag deploy/semrag-api -- alembic downgrade -1

# 3. Verify the schema matches the code's expectations.
kubectl exec -n semrag deploy/postgres -- psql -U semrag -d semrag -c "\d"
```

In practice, schema rollbacks are rarely necessary — the
Wave 1 schema migrations are designed to be backwards-compatible.

## Rolling Back Indexes

The vector and IR indexes are stored in Postgres. They are
**not** part of the application deployment — rolling back the
app does not affect them. If a new indexing algorithm produced
worse results, you can re-index:

```bash
curl -X POST -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}' \
  https://semrag.example.com/v1/projects/<project>/index
```

Re-indexing takes seconds for small projects, minutes for large
ones. The query path remains available during re-indexing; the
old chunks are deleted only after the new ones are persisted.

## Postmortem

After rolling back, write a postmortem that includes:

1. The release that was rolled back (commit SHA).
2. The alert(s) or symptom(s) that triggered the rollback.
3. The timeline from rollout to rollback.
4. The root cause (what was wrong in the release).
5. The fix plan (when will the next attempt happen, with what
   safeguards).

Share the postmortem with the team within 48 hours.
