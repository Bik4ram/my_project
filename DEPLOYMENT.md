# Deployment

This document covers running SemRAG in every supported environment:
local development, Docker, Docker Compose, Kubernetes, and Helm.

For an end-to-end developer quickstart (venv, CLI, single API call),
see [README.md](./README.md). For the public API surface, see
[RUN_API.md](./RUN_API.md).

---

## Health & version endpoints

The API exposes three production-grade operational endpoints. All
return JSON and require no authentication.

| Endpoint   | Purpose                                              | Failure mode                              |
|------------|------------------------------------------------------|-------------------------------------------|
| `GET /livez`  | Process liveness. No I/O.                       | 200 only if the Python process is alive.   |
| `GET /readyz` | Readiness: every dependency is reachable.      | 200 with JSON `status: ready`, 503 otherwise. |
| `GET /version`| Build metadata (version, git commit, build time). | 200 always.                               |

`/readyz` checks, in order:

- **database** — Postgres reachable *and* the `vector` (pgvector) extension is installed.
- **llm** — Provider configuration is internally consistent (URL, model, timeout).
- **embedding** — Provider configuration is internally consistent.
- **job_executor** — Background job executor is initialized and not shut down.
- **cache** — Configured cache backend reports ready.
- **rate_limiter** — Configured rate-limit backend reports ready.

```bash
curl -s http://localhost:8000/livez
# {"status":"alive"}

curl -s http://localhost:8000/readyz | jq
# {
#   "status": "ready",
#   "checks": {
#     "database": "ok",
#     "llm": "configured (ollama/qwen2.5:7b)",
#     "embedding": "configured (ollama/nomic-embed-text)",
#     "job_executor": "ok",
#     "cache": "ok (memory)",
#     "rate_limiter": "ok (memory)"
#   },
#   "version": "0.1.0",
#   "timestamp": "2026-08-08T22:00:00+00:00"
# }

curl -s http://localhost:8000/version | jq
# {
#   "application": "semrag",
#   "version": "0.1.0",
#   "git_commit": "abc1234",
#   "build_timestamp": "2026-08-08T22:00:00Z",
#   "python_version": "3.12.3",
#   "environment": "production"
# }
```

The Dockerfile's `HEALTHCHECK` directive uses `/livez` (cheap, no
external dependencies). Kubernetes liveness and readiness probes use
`/livez` and `/readyz` respectively.

---

## Running locally

Plain Python with a venv. See [README.md](./README.md#installation) for
the full setup. The default Settings assume Postgres on
`localhost:5432` and Ollama on `localhost:11434`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -e .
cp .env.example .env

alembic upgrade head
make run-api
```

---

## Docker (single container)

The image is multi-stage, non-root, pinned to a specific Python base,
and includes `tini` for clean signal handling. Build with the
version-tagged helper:

```bash
make docker-build      # builds semrag:dev
make docker-run        # runs it
# OR explicitly:
docker build -t semrag:0.1.0 .
docker run --rm -p 8000:8000 --env-file .env semrag:0.1.0
```

`docker stop` sends SIGTERM; the API's lifespan drains the job executor
and disposes the SQLAlchemy engine before exiting.

---

## Docker Compose

`docker-compose.yml` ships with everything needed for a single-host
production-ish deployment: Postgres + pgvector, SemRAG API, and
profile-gated Ollama and Redis. `.env` provides configuration.

```bash
cp .env.example .env
docker compose up -d                 # API + Postgres + worker
docker compose --profile ollama up -d # add Ollama
docker compose --profile redis up -d  # add Redis

docker compose ps
docker compose logs -f api

curl http://localhost:8000/livez
curl http://localhost:8000/readyz
```

`docker-compose.override.yml` is merged automatically for local
development. It binds the API, Postgres, Redis, and Ollama ports to
loopback, mounts the source tree for hot reload, and runs uvicorn with
`--reload`. Production deployments should run `docker compose -f
docker-compose.yml up -d` without the override.

### Service profiles

| Service   | Default | How to enable                              |
|-----------|---------|--------------------------------------------|
| `postgres` | on     | always                                     |
| `api`      | on     | always                                     |
| `worker`   | on     | always (indexing; no HTTP)                 |
| `redis`    | off    | `--profile redis`                          |
| `ollama`   | off    | `--profile ollama`                         |

---

## Kubernetes

Raw manifests live under `deploy/k8s/`. Apply them in order:

```bash
kubectl apply -f deploy/k8s/
```

The deployment uses `/livez` for liveness and `/readyz` for readiness.
The startup probe is more forgiving than liveness so a single slow
first request (LLM probe) doesn't trigger a restart loop.

### Configuration overrides

The ConfigMap (`semrag-config`) holds non-secret settings. The Secret
(`semrag-secrets`) holds credentials. Override per-cluster with
Kustomize:

```yaml
# kustomization.yaml
resources:
  - deploy/k8s/

secretGenerator:
  - name: semrag-secrets
    literals:
      - POSTGRES_PASSWORD=...
      - SEMRAG_API_KEYS=key1,key2
```

Or edit the Secret in place:

```bash
kubectl create secret generic semrag-secrets -n semrag \
    --from-literal=POSTGRES_PASSWORD=... \
    --from-literal=SEMRAG_API_KEYS=key1,key2 \
    --dry-run=client -o yaml | kubectl apply -f -
```

The Deployment's `preStop` hook sleeps 5s before SIGTERM, giving
uvicorn a chance to drain in-flight requests. The
`terminationGracePeriodSeconds` is 30s, so the kubelet waits up to
that long before escalating to SIGKILL.

### Health probe timing

| Probe      | Path     | Initial delay | Period | Failure threshold |
|------------|----------|---------------|--------|-------------------|
| Liveness   | `/livez`  | 5 s          | 10 s   | 3                 |
| Readiness  | `/readyz` | 10 s         | 10 s   | 1                 |
| Startup    | `/livez`  | 5 s          | 5 s    | 18 (90 s budget)  |

### Ingress

The bundled Ingress is a template (NGINX, host `semrag.example.com`).
Set `ingressClassName` to match your controller and edit `tls:` to
reference your cert-manager-issued secret.

### Validation

```bash
kubectl apply --dry-run=client -f deploy/k8s/
# or, with kubeconform:
kubeconform -strict -summary deploy/k8s/*.yaml
```

---

## Helm

The chart lives at `deploy/helm/semrag/`. Install with sensible
defaults:

```bash
helm install semrag deploy/helm/semrag/ \
    --namespace semrag --create-namespace \
    --set config.environment=production \
    --set secret.POSTGRES_PASSWORD=... \
    --set secret.SEMRAG_API_KEYS=key1,key2
```

Override a custom image:

```bash
helm install semrag deploy/helm/semrag/ \
    --set image.repository=ghcr.io/my-org/semrag \
    --set image.tag=v0.1.0
```

Wire to an existing Secret (recommended for production):

```bash
kubectl create secret generic semrag-secrets \
    --from-literal=POSTGRES_PASSWORD=... \
    --from-literal=SEMRAG_API_KEYS=key1,key2

helm install semrag deploy/helm/semrag/ \
    --set existingSecret=semrag-secrets
```

Lint:

```bash
helm lint deploy/helm/semrag/
helm template semrag deploy/helm/semrag/ | kubeconform -strict -summary -
```

### Chart values

The defaults are tuned for a single-node staging cluster. See
`values.yaml` for the full list. Highlights:

| Key                          | Default                                     | Notes                          |
|------------------------------|---------------------------------------------|--------------------------------|
| `replicaCount`               | `2`                                         | bumped when autoscaling is off |
| `image.repository`           | `semrag`                                    | pin to a release tag in prod   |
| `image.tag`                  | `0.1.0`                                     | matches `Chart.appVersion`     |
| `resources.limits.cpu`       | `2`                                         | one uvicorn worker + LLM headroom |
| `resources.limits.memory`    | `1Gi`                                       | covers one in-flight LLM call  |
| `persistence.enabled`        | `true`                                      | required for the cache to survive restarts |
| `ingress.enabled`            | `false`                                     | flip when an ingress controller is installed |
| `autoscaling.enabled`        | `false`                                     | requires a metrics-server      |
| `config.environment`         | `production`                                | enforces `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION` |

---

## Environment variables

Full list with defaults and validation lives in
`src/semrag/core/settings.py`. The most commonly tuned for deployment:

| Variable                                  | Default                                                | Purpose                                                  |
|-------------------------------------------|--------------------------------------------------------|----------------------------------------------------------|
| `SEMRAG_ENVIRONMENT`                      | `development`                                          | `development` \| `staging` \| `production`               |
| `SEMRAG_API_KEYS`                         | `dev-local-key`                                        | comma-separated; required in production                  |
| `SEMRAG_DATABASE_URL`                     | `postgresql+psycopg2://semrag:semrag@localhost:5432/semrag` | full SQLAlchemy URL                                       |
| `SEMRAG_DB_POOL_SIZE`                     | `5`                                                    | SQLAlchemy pool size                                     |
| `SEMRAG_DB_MAX_OVERFLOW`                  | `10`                                                   | additional connections the pool can open under load      |
| `SEMRAG_DB_POOL_RECYCLE`                  | `300`                                                  | recycle connections after N seconds                      |
| `SEMRAG_LOG_LEVEL`                        | `INFO`                                                 | `TRACE` \| `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`      |
| `SEMRAG_LOG_FORMAT`                       | `human`                                                | `human` (color) or `json`                                |
| `SEMRAG_LLM_PROVIDER`                     | `ollama`                                               | see `reasoning/llm/factory.py`                            |
| `SEMRAG_LLM_MODEL`                        | `qwen2.5:7b`                                           |                                                        |
| `SEMRAG_LLM_BASE_URL`                     | `http://localhost:11434`                               |                                                        |
| `SEMRAG_LLM_PROBE_AT_STARTUP`             | `true`                                                 | fail-fast on bad LLM URL                                 |
| `SEMRAG_EMBEDDING_PROVIDER`               | `ollama`                                               |                                                        |
| `SEMRAG_EMBEDDING_MODEL`                  | `nomic-embed-text`                                     |                                                        |
| `SEMRAG_EMBEDDING_BASE_URL`               | `http://localhost:11434`                               |                                                        |
| `SEMRAG_EMBEDDING_DIMENSION`              | `768`                                                  | must match the model's output                            |
| `SEMRAG_RATE_LIMIT_PER_MINUTE`            | `60`                                                   | per-key                                                  |
| `SEMRAG_RATE_LIMIT_BACKEND`               | `memory`                                               | `memory` \| `redis`                                      |
| `SEMRAG_QUERY_CACHE_BACKEND`              | `memory`                                               | `memory` \| `redis`                                      |
| `SEMRAG_API_KEY_STORE`                    | `memory`                                               | `memory` \| `redis` \| `database`                        |
| `SEMRAG_MAX_BODY_SIZE`                    | `5242880`                                              | 5 MB                                                    |
| `SEMRAG_GIT_COMMIT`                       | unset                                                  | injected by CI for `/version`                            |
| `SEMRAG_BUILD_TIME`                       | unset                                                  | injected by CI for `/version`                            |

All runtime configuration is validated at startup. A misconfigured
deployment fails fast with an actionable error message that lists
every offending setting in one go.

---

## Production deployment checklist

Before going live, verify:

- [ ] `SEMRAG_ENVIRONMENT=production` is set.
- [ ] `SEMRAG_API_KEYS` contains at least one non-default key.
- [ ] `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION=true` (the default).
- [ ] `SEMRAG_DATABASE_URL` points at Postgres with the `pgvector` extension installed.
- [ ] `SEMRAG_LLM_BASE_URL` and `SEMRAG_EMBEDDING_BASE_URL` resolve to a healthy provider.
- [ ] `SEMRAG_LOG_FORMAT=json` so the log pipeline can ingest output.
- [ ] `SEMRAG_TRUSTED_HOSTS` is set when behind a reverse proxy that rewrites `Host`.
- [ ] `SEMRAG_CORS_ORIGINS` is set when the API is consumed from a browser.
- [ ] Secrets are stored in a Secret manager (SealedSecret, ExternalSecret, Vault, etc.) — never in `values.yaml`.
- [ ] `/readyz` returns 200 against the deployed Pod.
- [ ] `/version` returns the expected git commit and version.
- [ ] TLS is terminated at the Ingress / proxy.
- [ ] Pod `requests` aren't set so high the scheduler can't fit two replicas per node.
- [ ] PDB (PodDisruptionBudget) is applied if you ship one.
- [ ] Backup strategy is in place for the Postgres data volume.

---

## Graceful shutdown

When a pod receives SIGTERM (kubelet, `docker stop`, Ctrl-C):

1. The pod's `preStop` hook sleeps 5s, draining in-flight requests via uvicorn's graceful shutdown.
2. The FastAPI lifespan runs `shutdown_resources()`:
   - the JobExecutor drains (`wait=True`) and refuses new submissions
   - the SQLAlchemy engine's connection pool is disposed
3. tini (PID 1 in the container) reaps zombies and exits cleanly.

The kubelet waits up to `terminationGracePeriodSeconds` (30s) before
SIGKILL. Tune the `preStop` sleep and the grace period together — the
sum of both is the upper bound on in-flight request duration before a
deploy forces a hard cut.

---

## Troubleshooting

#### `/readyz` reports `database: pgvector extension not installed`

The Postgres image is missing the `vector` extension. Check the
docker-compose service:

```bash
docker compose exec postgres psql -U semrag -c '\dx'
```

If `vector` isn't listed, the image is `pgvector/pgvector:pg*` (not
plain `postgres`) — confirm the `image:` line in your compose/K8s
manifest. For an existing cluster without pgvector, install it via
`CREATE EXTENSION IF NOT EXISTS vector;` in your migration job.

#### `/readyz` reports `database: OperationalError: ...`

The API can't reach Postgres. Check:

```bash
docker compose logs postgres
kubectl -n semrag logs -l app.kubernetes.io/component=api
```

Common causes: wrong host (`postgres` vs `localhost`), wrong port
(5432), wrong credentials, network policy blocking the API pod from
the DB pod.

#### `llm probe failed` at startup

The `_probe()` call in `reasoning/llm/factory.py` runs at process
startup (when `SEMRAG_LLM_PROBE_AT_STARTUP=true`, the default). It
fails fast on a bad URL or missing model. Either fix the provider
configuration or disable the probe for environments where the LLM
isn't reachable at startup (e.g. `docker compose up` before the LLM
container is healthy).

Disable with `SEMRAG_LLM_PROBE_AT_STARTUP=false` in `.env` or the
ConfigMap.

#### Pods stuck in `CrashLoopBackOff`

```bash
kubectl -n semrag logs -l app.kubernetes.io/component=api --previous
```

The most common cause is the runtime configuration validator raising
`RuntimeConfigurationError` — the message lists every problem and
should be visible in the previous pod's logs.

#### Helm install fails with `value for ... is required`

When `existingSecret` is empty, the chart tries to render the bundled
Secret template, which requires `POSTGRES_PASSWORD` and
`SEMRAG_API_KEYS` in `secret:`. Either set those values, or set
`existingSecret: <name>` to use a Secret managed elsewhere.

#### Rolling update hangs at `maxUnavailable=0`

The new pod can't pass readiness. Check `/readyz` on the rolling pod
directly:

```bash
kubectl -n semrag port-forward <new-pod> 8000:80
curl localhost:8000/readyz
```

The likely culprit is a destination provider (LLM, embedding) that
isn't reachable from the new pod's network namespace — for example,
an init script that pulls a model pulls it on the *first* pod and
the new pod is racing against the pull.

#### Embedding cache lost on restart

`SEMRAG_EMBEDDING_CACHE_BACKEND=jsonfile` writes the cache to
`SEMRAG_EMBEDDING_CACHE_DIR` (default `./.embedding_cache`). In a
container without a mounted volume, the cache evaporates on restart.
Mount a persistent volume at the cache directory (the chart does
this when `persistence.enabled=true`).