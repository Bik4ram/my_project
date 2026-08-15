# Memory Exhaustion Runbook

## Symptom

- Alert: `SemRAGHighMemoryUsage` (severity: warning)
- Pod is OOMKilled and restarted by the kubelet.
- API pod's last few log lines show `Killed` followed by an
  exit code of 137.
- Worker jobs hang or fail with OOM-related errors.

## Triage

1. Confirm OOMKill:

   ```bash
   kubectl describe pod -n semrag <pod-name> | grep -A2 "Last State"
   ```

   Look for `Reason: OOMKilled` and a `terminated` exit code 137.

2. Check the memory footprint before the kill:

   ```bash
   kubectl top pod -n semrag <pod-name>
   ```

3. Look at the project's index size. A 1M-line codebase can
   consume 4-8 GB resident after indexing due to the embedding
   cache and the in-memory TFIDF vectorizer.

## Mitigation

- **Memory limit too tight** — raise the limit on the API/worker
  pod. The default 2 GB is enough for a small project (~10k lines)
  but inadequate for production-scale indexes.

  In `deploy/helm/semrag/values.yaml`:

  ```yaml
  resources:
    requests:
      memory: "4Gi"
    limits:
      memory: "8Gi"
  ```

  Then `helm upgrade semrag deploy/helm/semrag/`.

- **Embedding cache is too large** — the `InMemoryEmbeddingCache`
  grows without bound as new chunks are embedded. Restart the pod
  to clear it; long-term, switch to the `jsonfile` cache backend
  via `SEMRAG_EMBEDDING_CACHE_BACKEND=jsonfile`.

- **TFIDF vectorizer is huge** — TFIDF is in-memory by design.
  Switch to Ollama-based embeddings for any project with more
  than ~100k chunks.

- **Memory leak in the worker** — restart the pod. If the leak
  recurs within hours, capture a heap dump
  (`py-spy dump --pid <pid>`) and file a bug.

## Verification

1. Pod stays up for at least one full indexing cycle without
   OOMKill.
2. `node_memory_MemAvailable_bytes` on the node stays above 15%
   of total.

## Postmortem Notes

- Note the project size and the memory limit at the time of the
  kill. Sizing rules of thumb:
  - < 100k chunks: 4 GB limit
  - 100k-500k chunks: 8 GB limit
  - 500k+ chunks: 16 GB limit (and consider splitting the project)
- If the same pod OOMKills at the same memory limit, file a bug
  with the heap dump attached.
