# Embedding Failure Runbook

## Symptom

- Alert: `SemRAGEmbeddingUnavailable` (severity: warning)
- Indexing jobs are stuck in the `embedding` stage for >30 minutes
  per project (visible in the indexing dashboard).
- `/readyz` shows `checks.embedding: not configured` or
  `connection refused`.
- Logs contain `embedding request failed`, `EmbeddingProviderError`,
  or `ollama embeddings failed after N attempts`.

## Triage

1. Confirm the embedding provider is reachable:

   ```bash
   # Ollama
   curl -sS http://<embedding-host>:11434/api/embeddings \
     -d '{"model": "nomic-embed-text", "prompt": "test"}' \
     -H "Content-Type: application/json"

   # OpenAI-compatible
   curl -sS -H "Authorization: Bearer $KEY" \
     http://<embedding-host>/v1/embeddings \
     -d '{"model": "nomic-embed-text", "input": "test"}' \
     -H "Content-Type: application/json"
   ```

2. Check `SEMRAG_EMBEDDING_PROVIDER` and `SEMRAG_EMBEDDING_MODEL` on
   the API pod.

3. Check the `embedding.request.duration_ms` metric in Prometheus.
   P95 > 30s usually means the model server is overloaded.

## Mitigation

- **Embedding server OOMKilled** — `nomic-embed-text` is a 274 MB
  model; on a node with <4 GB free RAM it'll be killed. Either
  restart and add memory, or move to a smaller embedding model.
- **Model not loaded** — Ollama loads models on first request. The
  first call after a server restart takes 5-30 seconds; subsequent
  calls are fast. If you see persistent timeouts on every request,
  the model is failing to load. Check Ollama logs.
- **Embedding cache poisoning** — if the cache is returning wrong-
  dimension vectors, delete the cache directory
  (`SEMRAG_EMBEDDING_CACHE_DIR`).
- **Fallback to TFIDF** — as a last resort, switch the provider to
  `tfidf` so indexing can continue without an external embedding
  server. Set `SEMRAG_EMBEDDING_PROVIDER=tfidf` and re-run the
  failing indexing job. Note that the index built with TFIDF is not
  directly comparable to the Ollama-built index; queries will need
  to use the same provider.

## Verification

1. `/readyz` shows `embedding: ok` (or `configured`).
2. An indexing job runs to completion in the normal timeframe:

   ```bash
   curl -X POST https://semrag.example.com/v1/projects/<project>/index \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

3. `semrag_health_up{component="embedding"}` returns to 1 in
   Prometheus.

## Postmortem Notes

- Note the embedding model and provider.
- Check the indexing job's failure rate during the incident — a
  burst of failures suggests a downstream change (model upgrade,
  config rotation) rather than a hardware issue.
- If the cache was poisoned, document the cache key pattern that
  triggered it. Future versions should validate the embedding
  dimension on cache read.
