# LLM Outage Runbook

## Symptom

- Alert: `SemRAGLLMUnavailable` (severity: critical)
- `/readyz` returns HTTP 503 with `checks.llm` showing
  `not configured` or `connection refused`.
- `/v1/projects/{id}/answer` requests time out or return 500 with
  `LLMProviderError` in the response body.
- Logs contain `llm.complete.failed`, `LLM provider failed`, or HTTP
  error responses from the upstream model server.

## Triage

1. Check which LLM provider is configured:

   ```bash
   kubectl exec -n semrag deploy/semrag-api -- env | grep SEMRAG_LLM_
   ```

2. Probe the LLM endpoint directly:

   ```bash
   # Ollama
   curl -sS http://<llm-host>:11434/api/tags

   # OpenAI-compatible
   curl -sS -H "Authorization: Bearer $SEMRAG_LLM_API_KEY" \
     http://<llm-host>/v1/models
   ```

3. Check `/readyz` for the specific message:

   ```bash
   curl -s https://semrag.example.com/readyz | jq .checks.llm
   ```

   - `not configured` → `SEMRAG_LLM_BASE_URL` or `SEMRAG_LLM_MODEL` is
     empty.
   - `connection refused` → the LLM server isn't reachable from the
     API pod.
   - `configured (provider/model)` → configuration is fine but the
     actual probe call failed (check upstream).

## Mitigation

- **LLM server is down** — restart or scale the model server. If
  SemRAG is configured to use Ollama, check the Ollama pod:

  ```bash
  kubectl rollout restart statefulset/ollama -n semrag
  ```

- **OOMKilled on the LLM pod** — large models (>=13B parameters)
  routinely OOM on GPU-less hardware. Reduce `SEMRAG_LLM_NUM_PREDICT`
  to limit per-request token cost, or move to a smaller model.
  See the `memory-exhaustion.md` runbook for sizing.

- **Slow / hanging LLM responses** — many cloud-hosted models rate-
  limit aggressively. Increase `SEMRAG_LLM_MAX_RETRIES` if you have
  budget, otherwise reduce traffic with a tighter
  `SEMRAG_RATE_LIMIT_PER_MINUTE`.

- **Wrong credentials** — check `SEMRAG_LLM_API_KEY`. Anthropic /
  OpenAI keys are particularly easy to rotate by accident.

## Verification

1. `/readyz` shows `llm: ok` (or `configured` for non-mock providers).
2. The `/answer` endpoint returns a non-empty answer for a smoke
   test query within the configured timeout
   (`SEMRAG_LLM_TIMEOUT_SECONDS`):

   ```bash
   time curl -X POST https://semrag.example.com/v1/projects/<project>/answer \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "smoke test"}'
   ```

3. `semrag_health_up{component="llm"}` returns to 1 in Prometheus.

## Postmortem Notes

- Note the LLM provider / model and the failure mode (timeout, OOM,
  auth, etc.).
- Check `semrag_rag_llm_duration_seconds` for the P95 latency
  leading up to the outage — slow LLM calls usually precede an
  outright outage by minutes.
- If the LLM was slow but reachable, consider lowering
  `SEMRAG_LLM_NUM_PREDICT` and/or scaling the LLM pod horizontally.
