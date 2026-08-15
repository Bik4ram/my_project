# `deploy/` — Deployment assets

Every artifact the codebase ships for running SemRAG outside
the developer's laptop. Lives next to the code so the directory's
contents and the configuration they describe cannot drift.

The split is deliberate:

- **`deploy/k8s/`** — raw Kubernetes manifests, apply in numeric order.
- **`deploy/helm/semrag/`** — the Helm chart packaging the same shape.
- **`deploy/monitoring/`** — Prometheus scrape config, recording
  rules, alertmanager routing, and Grafana provisioning.
- **`Dockerfile` + `docker-compose.yml` + `docker-compose.override.yml`**
  — single-machine containers, root-level files because Docker Compose
  expects them at the project root.

This README is the **assets map**. It explains what each file
contains, why each piece exists, and how the pieces wire together —
so an operator opening the directory for the first time can decide
which artifact to edit without reading every line. **End-to-end
operational procedures** (how to bring the stack up, what each health
endpoint returns, the production checklist, graceful shutdown,
troubleshooting recipes) live in
[`DEPLOYMENT.md`](../DEPLOYMENT.md) and the runbooks under
[`docs/operations/`](../docs/operations/). This README does not
duplicate them.

---

## 1. Purpose of `deploy/`

The package exists to give every supported environment a single,
versioned source of truth for its runtime topology:

- **Docker** — a single-host production-ish deployment (one VM,
  one host). Postgres + pgvector, the API, an opt-in Redis, an
  opt-in Ollama.
- **Docker Compose (with `override.yml`)** — the developer path:
  same stack with hot reload and source-tree bind mounts.
- **Kubernetes (`deploy/k8s/`)** — eight numbered manifests that
  apply in order without a templating engine.
- **Helm (`deploy/helm/semrag/`)** — the same K8s shape, parameterized,
  with secrets deliberately not bundled so operators can wire to an
  existing Secret manager.
- **Monitoring (`deploy/monitoring/`)** — Prometheus scrape config,
  alert rules, alertmanager routing, Grafana datasource and dashboard
  provisioning. Eight boards, one per surface area (API, retrieval,
  LLM, database, cache, rate-limit, jobs, indexing).

Configuration is **fully env-var-driven**. Both the Compose and K8s
shapes expose every operational knob as a `SEMRAG_*` environment
variable; nothing in `deploy/` reads application configuration from a
file. The ConfigMap and `values.yaml` exist only to feed the
deployment's env vars from a structured location.

---

## 2. Directory layout

```
.
├── Dockerfile                          Production image (multi-stage, non-root, tini)
├── docker-compose.yml                  Production Compose stack (api + postgres + opt-in redis/ollama)
├── docker-compose.override.yml         Local-dev overlay (mounts source, runs --reload)
│
├── deploy/
│   ├── k8s/                            Raw K8s manifests, numbered for apply order
│   │   ├── 00-namespace.yaml
│   │   ├── 01-configmap.yaml           SEMRAG_* non-secret env
│   │   ├── 02-secret.yaml              POSTGRES_PASSWORD, SEMRAG_API_KEYS, …
│   │   ├── 03-pvc.yaml                 /app/data → 1Gi PVC (cache + debug artifacts)
│   │   ├── 04-deployment.yaml          API Deployment (2 replicas, RollingUpdate)
│   │   ├── 05-service.yaml             ClusterIP Service (api semrag :80 → 8000)
│   │   ├── 06-ingress.yaml             NGINX Ingress template (TLS commented)
│   │   └── 07-worker-deployment.yaml   Worker Deployment (1 replica, no HTTP)
│   │
│   ├── helm/semrag/                    Helm chart (Chart v2, appVersion 0.1.0)
│   │   ├── Chart.yaml
│   │   ├── values.yaml                 Defaults — replicaCount, resources, probes, ingress
│   │   └── templates/                  All K8s resources as Go templates
│   │       ├── _helpers.tpl            Name / labels / selectorLabels / secret helpers
│   │       ├── configmap.yaml          Mirrors k8s/01-configmap.yaml
│   │       ├── secret.yaml             Rendered only when existingSecret is empty
│   │       ├── pvc.yaml                /app/data, gated by persistence.enabled
│   │       ├── serviceaccount.yaml     ServiceAccount for IRSA / Workload Identity
│   │       ├── deployment.yaml         API Deployment
│   │       ├── worker-deployment.yaml  Worker Deployment (`python -m semrag.worker`)
│   │       ├── service.yaml            ClusterIP Service
│   │       ├── ingress.yaml            Gated by ingress.enabled
│   │       ├── hpa.yaml                Gated by autoscaling.enabled
│   │       └── NOTES.txt               Post-install verification commands
│   │
│   └── monitoring/                     Observability surface
│       ├── prometheus/
│       │   ├── prometheus.yml          Scrape config: semrag-api, semrag-worker, self
│       │   └── rules/semrag.yml        Alerts: latency, errors, health, cache, workers, host
│       ├── alertmanager/
│       │   └── alertmanager.yml        Routing by severity; default + critical receivers
│       └── grafana/
│           ├── datasources/prometheus.yml   One read-only Prometheus datasource
│           ├── dashboards/provider.yml      File-based provisioning into folder "SemRAG"
│           └── dashboards/semrag-*.json     8 dashboards (api, retrieval, llm, database,
│                                            cache, rate-limit, jobs, indexing)
```

---

## 3. Docker assets

### `Dockerfile`

Multi-stage production image, pinned Python 3.12 on `bookworm`-slim,
non-root user (`uid=1001`), `tini` as PID 1 for clean signal
forwarding. Verified against `Dockerfile`:

- **Builder stage** (`python:3.12-slim-bookworm`, `build-essential`)
  installs `requirements.txt` to `/install`. `build-essential` is the
  fall-back compiler for native wheels (`blake3`, `pgvector`) that
  don't ship manylinux builds.
- **Runtime stage** installs only `tini` + `curl`. Copies `/install`
  from the builder, then `pyproject.toml` + `src/`, and runs
  `pip install --no-deps .` to register the package. `chown semrag`
  on `/app` before `USER semrag`.
- **`HEALTHCHECK`** calls `curl -fsS http://localhost:8000/health` —
  matches what the Compose and K8s healthchecks use for liveness.
  Note: `/health` is the alias — `/livez` and `/readyz` are the
  documented production endpoints; `/health` is the legacy alias the
  Dockerfile pinned for backwards compatibility (see `DEPLOYMENT.md`
  §"Health & version endpoints").
- **Entrypoint:** `tini -- python -m uvicorn semrag.api.app:app
  --host 0.0.0.0 --port 8000 --workers 1`. One worker is correct for
  the FastAPI lifespan / startup-validation path; horizontal scale
  lives in the Deployment, not in `uvicorn --workers`.

### `docker-compose.yml`

`name: semrag`. Four services:

| Service    | Image                            | Profile      | Purpose                                                  |
| ---------- | -------------------------------- | ------------ | -------------------------------------------------------- |
| `postgres` | `pgvector/pgvector:pg16`         | always-on    | Postgres 16 with the `vector` extension pre-installed.    |
| `redis`    | `redis:7-alpine`                 | `redis`      | Rate-limit / query-cache backend (opt-in).               |
| `ollama`   | `ollama/ollama:latest`           | `ollama`     | Local LLM + embedding provider (opt-in).                 |
| `api`      | built from the Dockerfile        | always-on    | SemRAG API on host port `${SEMRAG_API_PORT:-8000}`.      |
| `worker`   | built from the Dockerfile        | always-on    | Indexing worker; polls `semrag` Postgres job store.      |

Verified services-level details:

- **Service hosts are the bridge-network DNS.** `SEMRAG_DATABASE_URL`
  hard-codes `postgres:5432`, `SEMRAG_LLM_BASE_URL` defaults to
  `http://ollama:11434`. The published host ports are not used for
  inter-service traffic; they exist for the operator to run `psql`,
  `redis-cli`, or `curl` from the host.
- **Profile gating** keeps the production-ish default lean — just
  the API + Postgres. Redis and Ollama are activated with
  `docker compose --profile redis up` / `--profile ollama up`. The
  `depends_on: { redis: { required: false } }` lets the Compose
  parser still validate the block when redis isn't in the active
  profile.
- **Image default** is `${SEMRAG_IMAGE:-semrag:latest}` so the same
  compose file works with locally-built dev images or pinned release
  tags from a registry.
- **Healthchecks:** `postgres` runs `pg_isready` + `SELECT 1`
  (cheap; pgvector is verified by the API's `/readyz`); `redis` runs
  `redis-cli ping`; `ollama` runs `curl -fsS /api/tags` (cheaper
  than a full generate); `api` runs `curl -fsS /livez` (no
  dependency check — see the in-file comment explaining why
  `/readyz` would be wrong for `HEALTHCHECK`).
- **Resource limits** are env-overridable so the same compose scales
  between developer (1 GB API memory) and CI (4 GB worker memory).
- **`stop_grace_period: 30s`** for the API, **`60s`** for the
  worker, with `stop_signal: SIGTERM`. uvicorn drains; the FastAPI
  lifespan's `shutdown_resources()` disposes the SQLAlchemy engine.
- **`SEMRAG_API_RUNS_WORKER=false`** is set on both `api` and `worker`.
  The API does not execute jobs in-process; the worker is a separate
  container polling the persistent job store.
- **Volumes:** named volumes for `postgres-data`, `redis-data`,
  `ollama-data`. Not host-bind-mounted.

### `docker-compose.override.yml`

The auto-merged local-dev overlay:

- Drops the profile guard on `redis` and `ollama` (`profiles: []`)
  so both come up automatically with `docker compose up`.
- Adds `volumes: ./src:/app/src:ro` and
  `./pyproject.toml:/app/pyproject.toml:ro` so Python edits are
  visible after `docker compose restart api` (the image installs the
  package with `pip install --no-deps .`, so module discovery works
  without re-installing).
- Overrides `command:` to add uvicorn's `--reload` flag.
- Sets `SEMRAG_ENVIRONMENT=development`, `SEMRAG_LOG_LEVEL=DEBUG`,
  `SEMRAG_LOG_FORMAT=human` — the validator's
  `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION` is a no-op when
  `environment=development`.
- Binds service ports to `127.0.0.1` explicitly (matches the prod
  compose defaults).

**Production deploys run with the override disabled:**
`docker compose -f docker-compose.yml up -d`. The override is for
local development only.

---

## 4. Kubernetes manifests (`deploy/k8s/`)

Eight YAML files, zero-prefixed filenames so `kubectl apply -f
deploy/k8s/` applies them in the right order: namespace first, PVC
before the Deployment that mounts it, Deployment before the Service
that routes to it, Service before the Ingress that fronts it.

### `00-namespace.yaml`

`Namespace/semrag` with `app.kubernetes.io/name: semrag` and
`app.kubernetes.io/part-of: semrag`. The shared key/value pair is
what every other manifest's `labels:` uses for selector matching.

### `01-configmap.yaml`

`ConfigMap/semrag-config` — non-secret env. Holds: `SEMRAG_DATABASE_URL`
(with `$(POSTGRES_PASSWORD)` interpolated by the pod from the Secret
via `envFrom`), log level/format, LLM and embedding provider settings,
cache/rate-limit backend selection, security flags
(`environment`, `rejectDevKeysInProduction`,
`trustedHosts`, `corsOrigins`, `apiKeyStore`,
`maxBodySize`), and CI-injected `gitCommit` / `buildTime` placeholders.

Override per-cluster with Kustomize or `kubectl edit configmap
semrag-config`. The Deployment's pod annotation `checksum/config`
is wired to force a rollout on ConfigMap content change.

### `02-secret.yaml`

`Secret/semrag-secrets` template — `stringData` placeholders. **The
file's placeholders are committed on purpose** so the Secret exists
for `kubectl apply` to wire `valueFrom.secretKeyRef`; **real values
are never committed**. The Deployment reads:

| Secret key                | Used as                                                 |
| ------------------------- | ------------------------------------------------------- |
| `POSTGRES_PASSWORD`       | Interpolated into `SEMRAG_DATABASE_URL`.                |
| `SEMRAG_API_KEYS`         | Comma-separated list, validated at startup.             |
| `SEMRAG_LLM_API_KEY`      | Optional (`optional: true`). Empty for Ollama.          |
| `SEMRAG_EMBEDDING_API_KEY`| Optional (`optional: true`).                            |

The file's comment block documents both the manual path
(`kubectl create secret generic … --from-literal=…`) and the
Kustomize `secretGenerator:` path.

### `03-pvc.yaml`

`PersistentVolumeClaim/semrag-data` — 1 Gi, `ReadWriteOnce`,
`storageClassName` deliberately left blank so the cluster default
applies. The comment block lists the two bounded payloads it holds
(embedding cache for the `jsonfile` backend, debug artifacts) and
notes the size is fixed by `Settings` defaults.

### `04-deployment.yaml`

`Deployment/semrag` — the API. Verified details:

- **Two replicas, RollingUpdate with `maxUnavailable=0, maxSurge=1`.**
  Exactly one pod serves traffic while the other rolls. Soft
  pod-anti-affinity by hostname (single-node dev clusters still
  schedule both pods).
- **`securityContext`:** non-root UID/GID 1001 (matches the
  Dockerfile); `readOnlyRootFilesystem: true`,
  `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`.
- **Probes** use the operational contract documented in `DEPLOYMENT.md`:

  | Probe     | Path      | Initial delay | Period | Failure threshold |
  | --------- | --------- | ------------- | ------ | ----------------- |
  | Liveness  | `/livez`  | 5s            | 10s    | 3                 |
  | Readiness | `/readyz` | 10s           | 10s    | 1                 |
  | Startup   | `/livez`  | 5s            | 5s     | 18 (90s budget)   |

- **`lifecycle.preStop`** sleeps 5s before SIGTERM. Combined with
  `terminationGracePeriodSeconds: 30`, this gives uvicorn 30s to
  drain in-flight requests via `shutdown_resources()` (job executor
  drain, SQLAlchemy engine dispose).
- **`SEMRAG_API_RUNS_WORKER=false`** is hard-coded as a literal
  env value (not from the ConfigMap) so a typo in the ConfigMap
  cannot accidentally re-enable the in-process worker.
- **Resources:** `requests: 250m CPU / 512Mi memory`,
  `limits: 2 CPU / 1Gi memory`. Sized for one uvicorn worker with
  headroom for one in-flight LLM call.
- **`checksum/config` annotation** is templated into the Helm
  chart via `sha256sum` of the rendered ConfigMap; in the raw
  manifest it is empty (`""`) because there is no templating
  engine. Operators wire `checksum/config` via
  `kubectl create configmap … --dry-run=client … | kubectl apply -f -`
  or by adopting `kustomize` (see the comment in the manifest).

### `05-service.yaml`

`Service/semrag` — `ClusterIP`, selector over
`app.kubernetes.io/name=semrag, component=api`, port 80 → `http`
(targetPort 8000). `publishNotReadyAddresses: false` is the
default; the comment makes it explicit so a future operator knows
why a failing-readiness pod stops getting traffic.

### `06-ingress.yaml`

`Ingress/semrag` — single host (`semrag.example.com`), NGINX-style
annotations (`proxy-body-size: 8m`, `proxy-read-timeout: 120`,
`proxy-send-timeout: 120`). `tls:` is commented out — the cluster
operator supplies cert-manager-issued secrets. `ingressClassName`
should be swapped for the provider-specific value (e.g. `gce`,
`alb`) on managed clusters.

### `07-worker-deployment.yaml`

`Deployment/semrag-worker`. One replica. Verified details:

- **`command: ["python", "-m", "semrag.worker"]`** — workers don't
  serve HTTP. They poll the persistent job store and run the
  indexing pipeline via `semrag.jobs.worker.indexing_worker`.
- **No `containerPort`** — workers never receive traffic.
- **Probes** are operational only: an `exec` `livenessProbe` runs
  `test -d /proc/1` to confirm the container's PID 1 is alive. The
  kubelet just needs to know the process hasn't segfaulted.
- **`preStop`** sleeps 10s before SIGTERM (longer than the API's
  5s, because an in-flight indexing job can be long); the
  termination grace period is 60s.
- **`SEMRAG_JOB_STORE_BACKEND`** is read from the ConfigMap
  (`optional: true`) so a deployment without that key falls
  through to the `Settings` default.
- Resources are heavier than the API's: 500m CPU / 1Gi memory
  requests, 4 CPU / 4Gi limits. The comment block notes a typical
  large project pegs at 1-2 GiB during embedding, so 4Gi covers
  parser-pipeline spikes.

---

## 5. Helm chart (`deploy/helm/semrag/`)

Chart API version `v2`, type `application`, version `0.1.0`,
`appVersion: "0.1.0"`. The chart packages exactly the same shape as
`deploy/k8s/` — there is no feature in the K8s manifests that the
chart can't express, and no feature in the chart that the K8s
manifests don't have.

### When to use the chart vs. the raw manifests

- **Use `deploy/k8s/`** when you want to read every applied object
  in a `kubectl diff`. Good for GitOps with non-Helm CD pipelines.
- **Use `deploy/helm/semrag/`** when you need per-release values
  overrides, an `existingSecret` hook, or a single `helm upgrade`
  to bump the image tag. Good for production installs.

### `values.yaml` highlights

| Key                              | Default    | Purpose                                                              |
| -------------------------------- | ---------- | -------------------------------------------------------------------- |
| `replicaCount`                   | `2`        | API replicas (ignored when `autoscaling.enabled: true`).             |
| `image.repository`               | `semrag`   | Container image; pin to a release tag in production.                 |
| `image.tag`                      | `0.1.0`    | Matches `Chart.appVersion`.                                          |
| `resources.limits.cpu/memory`    | `2` / `1Gi`| API container. One uvicorn worker + one in-flight LLM call.         |
| `persistence.enabled`            | `true`     | Required for embedding cache + debug artifacts to survive restarts.  |
| `persistence.size`               | `1Gi`      | PVC size; raise if retention / TTL settings changed.                 |
| `probes.liveness.path`           | `/livez`   | Matches the API contract.                                            |
| `probes.readiness.path`          | `/readyz`  | Matches the API contract.                                            |
| `probes.startup.path`            | `/livez`   | 90-second startup budget for the LLM probe.                          |
| `terminationGracePeriodSeconds`  | `30`       | ≥ `preStop` sleep (5s). The sum is the upper bound on drain time.    |
| `ingress.enabled`                | `false`    | Flip when an ingress controller is installed.                        |
| `autoscaling.enabled`            | `false`    | Requires metrics-server.                                             |
| `config.environment`             | `production` | Drives `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION` validation.           |
| `existingSecret`                 | `""`       | Use an external Secret (recommended for prod); otherwise the bundled `templates/secret.yaml` renders. |
| `worker.replicaCount`            | `1`        | Worker replicas (ignored when `worker.autoscaling.enabled: true`).   |
| `worker.resources.limits.cpu`    | `4`        | Indexing is more memory-hungry than serving HTTP.                    |
| `worker.terminationGracePeriodSeconds` | `60` | ≥ `preStop` sleep (10s); in-flight indexing jobs need the headroom. |

### Templates

| Template                  | Notes                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `_helpers.tpl`            | `name`, `fullname`, `chart`, `labels`, `selectorLabels`, `serviceAccountName`, `secretName`.      |
| `configmap.yaml`          | Mirrors `k8s/01-configmap.yaml`. Every key becomes a `SEMRAG_*` env var via `envFrom`.            |
| `secret.yaml`             | Rendered only when `existingSecret` is empty. Use the external Secret in production.             |
| `pvc.yaml`                | Rendered only when `persistence.enabled: true`. Size and class via values.                       |
| `serviceaccount.yaml`     | Rendered only when `serviceAccount.create: true` (default). For IRSA / Workload Identity.         |
| `deployment.yaml`         | API Deployment. `checksum/config` annotation uses `sha256sum` of the rendered ConfigMap — config changes trigger a rollout. |
| `worker-deployment.yaml`  | Worker Deployment. No HTTP port. Polls the persistent job store.                                  |
| `service.yaml`            | `ClusterIP` by default; `type` via values.                                                        |
| `ingress.yaml`            | Rendered only when `ingress.enabled: true`. TLS / className / hosts / paths via values.           |
| `hpa.yaml`                | Rendered only when `autoscaling.enabled: true`. CPU + memory utilization targets.                 |
| `NOTES.txt`               | Post-install commands: `kubectl get pods`, `port-forward`, hit `/livez`, `/readyz`, `/version`.  |

### What the chart **does not** do

Three things that the raw K8s manifests also don't do:

- **Manage Secrets.** Both shapes leave Secret creation to the
  operator (raw: via `kubectl create secret`; Helm: via
  `--set existingSecret=…` or `--values` with `secret:` populated,
  which only renders `templates/secret.yaml` as a placeholder).
- **Provision databases.** Postgres is out of scope; the in-cluster
  assumption is an operator-managed DB or an external managed
  Postgres. The Compose stack is the only place a Postgres container
  ships in-tree, and only for the developer / single-host path.
- **Wire to a managed Prometheus / Grafana.** The chart deploys
  nothing in `deploy/monitoring/`; it relies on those scrapers
  pointing at the chart's `Service` (or the pod IPs) externally.

---

## 6. Monitoring assets (`deploy/monitoring/`)

The observability surface. Lives in `deploy/` so the scrape config,
the alert rules, and the dashboards live with the rest of the
deployment artifacts.

### Prometheus — `prometheus.yml`

Three scrape jobs:

- `semrag-api` — the `/metrics` endpoint exposed by `core/metrics.py`.
  15s scrape interval, 10s timeout.
- `semrag-worker` — the same `/metrics` endpoint when API + worker
  run in separate pods. The scrape config covers both shapes (same
  container in the dev compose; separate deployments in K8s / Helm).
- `prometheus` — self-scrape, so the `up{}` series is visible.

`alerting:` points at `alertmanager:9093` (the Compose service
name on the bridge network — operators on K8s swap this for the
Alertmanager Service DNS).

### Prometheus rules — `rules/semrag.yml`

Seven groups, 12 alert rules. Verified against the file:

| Group                  | Alerts                                                                  |
| ---------------------- | ----------------------------------------------------------------------- |
| `semrag-latency`       | `SemRAGHighRequestLatencyP95`, `SemRAGHighLLMLatencyP95`, `SemRAGHighEmbeddingLatencyP95` |
| `semrag-errors`        | `SemRAGHighErrorRate` (5xx > 5%), `SemRAGIndexingJobsFailing`           |
| `semrag-health`        | `SemRAG{Database,LLM,Embedding,JobExecutor}Unavailable` (mirror `/readyz`) |
| `semrag-cache`         | `SemRAGCacheHitRateLow`, `SemRAGRateLimiterSaturation`                  |
| `semrag-workers`       | `SemRAGWorkerCrash` (uptime counter resetting), `SemRAGIndexingJobBacklog` (queue > completion × 1.2 over 15 min) |
| `semrag-host`          | `SemRAGLowDiskSpace`, `SemRAGHighMemoryUsage` (require `node_exporter`) |

Severity is `info` / `warning` / `critical`. The `team: platform`
label and `severity` together drive Alertmanager routing.

### Alertmanager — `alertmanager.yml`

Three receivers (declared: `default`, `critical`; `warning` reuses
`default`). Routing rules:

- `critical`: 10s group wait, 1h repeat interval.
- `warning`: 1m group wait, 6h repeat interval.
- `default`: 30s group wait, 4h repeat interval.

An `inhibit_rule` suppresses `warning` alerts that share an
`alertname` with a `critical` alert for the same cluster. Webhook
URLs (`https://hooks.example.com/...`) are placeholders — replace
per environment.

### Grafana — `datasources/prometheus.yml` + `dashboards/`

- One read-only Prometheus datasource, pointing at
  `http://prometheus:9090` (the dev Compose service name; swap for
  managed Prometheus URL on K8s).
- Dashboard provisioning provider in `dashboards/provider.yml`
  loads every `dashboards/semrag-*.json` into folder `"SemRAG"`.
  Update interval: 30s.
- Eight boards, each scoped to a single concern:

  | Dashboard                       | What it shows                                                                |
  | ------------------------------- | ----------------------------------------------------------------------------- |
  | `semrag-api.json`               | API health, latency, error rate, request volume.                              |
  | `semrag-retrieval.json`         | Retrieval, embedding, reranker, and LLM latency.                              |
  | `semrag-llm.json`               | LLM provider latency, throughput, error rate.                                  |
  | `semrag-database.json`          | Postgres query latency, transaction count, connection pool.                    |
  | `semrag-cache.json`             | Query cache hit rate, miss rate, eviction rate.                                |
  | `semrag-rate-limit.json`        | Rate limiter accepted / rejected traffic.                                       |
  | `semrag-jobs.json`              | Background job lifecycle (queue → run → completion / failures).                 |
  | `semrag-indexing.json`          | Indexing job queue, throughput, duration, failures.                            |

### Why dashboards and rules live in `deploy/`, not in the source tree

The application emits the `semrag_*` metrics from
`src/semrag/core/metrics.py`. The Prometheus / Grafana / Alertmanager
side is **infrastructure** — the shapes it expects (service DNS
targets, alert receiver URLs, dashboard `uid`s) are environment-
specific. Putting the rules and dashboards next to the scrape config
keeps the entire observability surface in one directory that an
operator can lint / render / version-control as a unit.

---

## 7. Configuration ownership

Two rules govern where configuration lives. They are written down
here because they show up in every deploy:

### Rule 1 — application configuration lives in the source code

Every `SEMRAG_*` environment variable is declared in
`src/semrag/core/settings.py` with its default, validation, and
documentation. The deploy artifacts **only carry overrides**, never
new settings. If a value isn't in `Settings`, no deployment shape
can set it.

### Rule 2 — environment-specific values live in the deploy shape

| Value                                | Owned by                                                              |
| ------------------------------------ | --------------------------------------------------------------------- |
| Image repository + tag               | `Dockerfile` → `compose`, `values.yaml`, raw `k8s/04-deployment.yaml` |
| Resource requests / limits           | Compose `deploy.resources`; Helm `values.resources`; raw K8s `resources:` |
| Replica count                        | Compose only (single replica); Helm `replicaCount`; raw K8s `spec.replicas` |
| Secret material                      | Compose `.env` (with `env_file:`); K8s `Secret`; Helm `existingSecret` |
| Probes (path / timing / thresholds)  | Compose `healthcheck`; Helm `values.probes.*`; raw K8s probe blocks   |
| Ingress (host / class / annotations) | K8s `06-ingress.yaml`; Helm `values.ingress.*`                        |
| Scrapers / dashboards / alerts       | `deploy/monitoring/*`                                                |
| Persistent volume class + size       | Compose volume mounts; Helm `values.persistence`; raw K8s `03-pvc.yaml` |

Anything not in those rows **lives in `Settings`**, not in a deploy
artifact. Edits to a deploy artifact that introduce a new key
without putting it through `Settings` are an architectural mistake —
the validator won't know about it.

---

## 8. Deployment flow

End-to-end, in the order an operator follows:

### Local developer / single-host

```text
.env (or .env.example → cp)
    │
    ▼
docker compose up -d                 (api + postgres)
docker compose --profile ollama up   (local LLM)
    │
    ▼
http://localhost:8000/livez          (process liveness)
http://localhost:8000/readyz         (deps)
http://localhost:8000/version        (build metadata)

  docker-compose.override.yml auto-merges for local dev
  (mounts ./src, --reload, DEBUG log level).
```

### Raw Kubernetes

```text
kubectl apply -f deploy/k8s/

  00 namespace
  01 configmap (non-secret env)
  02 secret    (POSTGRES_PASSWORD, SEMRAG_API_KEYS — set via kubectl create)
  03 pvc       (/app/data, 1Gi)
  04 deployment (api, 2 replicas)
  05 service   (ClusterIP :80 → :8000)
  06 ingress   (NGINX host semrag.example.com)
  07 worker-deployment (1 replica, python -m semrag.worker)

kubectl port-forward svc/semrag -n semrag 8000:80
curl http://localhost:8000/livez
```

### Helm

```text
kubectl create secret generic semrag-secrets \
    --from-literal=POSTGRES_PASSWORD=... \
    --from-literal=SEMRAG_API_KEYS=...

helm install semrag deploy/helm/semrag/ \
    --namespace semrag --create-namespace \
    --set existingSecret=semrag-secrets \
    --set image.tag=v0.1.0

  # Secrets are NOT auto-created by the chart.
  # Image repository / tag / replicaCount / probes all via --set.
  # ConfigMap holds the SEMRAG_* non-secret env; Secret holds credentials.
```

### With monitoring

```text
Prometheus  →  scrapes /metrics  on  semrag-api:8000  and  semrag-worker:8000
           ↘  evaluates rules/semrag.yml (12 alerts)
              ↓
Alertmanager routes by severity (default / critical):
  - critical → 1h repeat, 10s group_wait
  - warning  → 6h repeat, 1m group_wait
  - default  → 4h repeat, 30s group_wait
            ↘  webhook (or PagerDuty / Slack per env)
Grafana:
  - 8 dashboards loaded from deploy/monitoring/grafana/dashboards/
  - provisioned folder "SemRAG"
  - datasource: deploy/monitoring/grafana/datasources/prometheus.yml
```

### Graceful shutdown (every shape)

1. `preStop` hook sleeps (5s API, 10s worker).
2. `tini` (PID 1) forwards SIGTERM.
3. uvicorn drains in-flight requests via `lifespan.shutdown`.
4. `shutdown_resources()` runs: `JobExecutor.shutdown(wait=True)`
   (refuses new submissions, drains in-flight jobs), then the
   SQLAlchemy engine's connection pool is disposed.
5. `tini` exits; kubelet / `docker stop` returns.

See `DEPLOYMENT.md` §"Graceful shutdown" for the full walkthrough.

---

## 9. Extension points

Every place the deployment layer is deliberately pluggable:

| Need                                          | Touch                                              | What to add                                                            |
| --------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| New deployment target (e.g. ECS / Nomad)      | New directory under `deploy/`, e.g. `deploy/ecs/`  | Maintain the env-var contract; reuse `Dockerfile`.                    |
| New Probe path                                | K8s `probes.*` block, Helm `values.probes.*`, Compose `healthcheck.test` | Repoint at any path the API exposes; do not invent new endpoints.    |
| New PVC                                       | K8s template (`templates/pvc.yaml`); Compose `volumes:` | Named volume in Compose, PVC in K8s. The mount path convention (`/app/data`) is the only contract. |
| New dashboard                                 | `deploy/monitoring/grafana/dashboards/semrag-<name>.json` | JSON model file + add to `provider.yml` if the path changes.       |
| New alert rule                                | `deploy/monitoring/prometheus/rules/semrag.yml`    | New rule under an existing or new group. Severity + `team: platform` labels drive Alertmanager routing. |
| New scrape job                                | `deploy/monitoring/prometheus/prometheus.yml`      | New `scrape_configs:` block. Reuse the 15s interval / 10s timeout.    |
| New alert receiver (PagerDuty / Slack)        | `deploy/monitoring/alertmanager/alertmanager.yml`  | New `receivers:` entry + `routes:` clause by `severity:` label.       |
| New `existingSecret` alternative (Vault / SOPS)| `deploy/helm/semrag/templates/secret.yaml`        | Template branch on a new toggle value; reuse `semrag.secretName` helper. |

Two extension points that are deliberately **not** available:

- **Hard-coding a `SEMRAG_*` env var into a deploy shape.** All
  app-level config is the validator's job (`core/settings.py`).
  A `command: ["/bin/sh", "-c", "export SEMRAG_FOO=bar"]` in a
  manifest would bypass validation.
- **A generic in-cluster Postgres.** Bringing your own Postgres is
  the production assumption. The Compose stack does ship a Postgres
  container (developer path); the K8s / Helm shapes assume the
  cluster operator has one ready, and the
  `SEMRAG_DATABASE_URL` template's host is `postgres` — change it to
  the operator-managed host.

---

## 10. Common debugging paths

The on-call runbooks under `docs/operations/` cover the user-facing
failure modes end-to-end (database outage, LLM outage, embedding
failure, indexing failure, job backlog, high latency, memory
exhaustion, rollback, escalation). The matrix below is for the
**asset-level** questions an operator opens `deploy/` with:

| Symptom                                              | Look first                                                                |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| `kubectl apply` complains about an unknown field     | Run `kubeconform -strict -summary deploy/k8s/*.yaml`; the K8s manifests follow the API versions on the cluster — older clusters may need `apps/v1` pas. |
| Helm template render fails with "value for X is required" | The chart requires `secret: { POSTGRES_PASSWORD, SEMRAG_API_KEYS }` when `existingSecret` is empty. The bundled `templates/secret.yaml` is the placeholder; the *real* Secret should come from outside. |
| Helm chart rolled out but ConfigMap change didn't trigger a restart | `checksum/config` annotation is computed from the rendered ConfigMap. Helm handles this automatically; manual `kubectl apply -f deploy/k8s/` against the raw manifests requires `kubectl rollout restart`. |
| Worker pod stuck in `CrashLoopBackOff`              | `kubectl logs -l component=worker --previous`. Often `SEMRAG_DATABASE_URL` missing from the Secret or `valueFrom.secretKeyRef` mis-keyed. |
| `/readyz` returns 503 across all replicas           | Check `01-configmap.yaml` → `SEMRAG_LLM_BASE_URL` / `SEMRAG_EMBEDDING_BASE_URL`; the API init probes the LLM at startup. Drop `SEMRAG_LLM_PROBE_AT_STARTUP=false` if the LLM is genuinely unreachable at boot. |
| Compose healthcheck for `postgres` flaps            | `pg_isready` alone isn't enough — the manifest's full test runs `pg_isready` + `SELECT 1`. If only `pg_isready` passes, the database itself isn't accepting queries (out of connections, recovery, etc.). |
| Compose up succeeds but `ollama` is "unhealthy"     | `/api/tags` returns the list of pulled models. If no model has been pulled, it's still 200 with `[]`, but a missing model is a different problem — pull one with `docker compose exec ollama ollama pull <model>`. |
| `helm upgrade` fails with "another operation in progress" | A previous `helm install` / `helm upgrade` is still reconciling. Wait for the release status to settle (`helm status semrag`), or roll back with `helm rollback`. |
| Prometheus scraping returns 0 series                 | Check `targets:` in `prometheus.yml` — the DNS `semrag-api:8000` is the Compose service name. On K8s, swap for the Service DNS (`semrag.semrag.svc.cluster.local:80` → targetPort 8000) or the pod IPs via a ServiceMonitor (an operator-managed CRD, not in the chart). |
| Alertmanager not receiving alerts                    | `prometheus.yml`'s `alerting.alertmanagers[0]` must resolve; check both directions (Prometheus → Alertmanager). The `group_by` and `group_wait` defaults are tuned for Slack-class receivers, not paging. |
| Grafana shows "Datasource not found"                | The datasource uid is `prometheus` by convention but a managed Grafana instance may have given it a different uid. Edit `dashboards/semrag-*.json` (`datasource.uid`) or override via `__inputs`. |
| A dashboard's panels show "No data"                 | Verify the metric names in `src/semrag/core/metrics.py` (the application's metric registry) match the queries. The metric names in `rules/semrag.yml` were chosen against that registry. |
| Image pull failures in K8s                          | `imagePullSecrets:` is empty by default in both raw and Helm. Private registries: `values.yaml`'s `imagePullSecrets:` list, or `spec.template.spec.imagePullSecrets` in the raw manifest. |

---

## 11. Common mistakes

1. **Committing real `SEMRAG_API_KEYS` / `POSTGRES_PASSWORD` to
   `deploy/k8s/02-secret.yaml`.** The placeholders
   (`REPLACE_ME`) are the file's contract. Real values go through
   `kubectl create secret … --from-literal=…` or Kustomize
   `secretGenerator:`.
2. **Editing `values.yaml` to manage real credentials.** The
   `secret:` block in `values.yaml` is for the bundled Secret only
   (placeholder rendering). Use `existingSecret:` and manage the
   Secret out-of-band (SealedSecret / ExternalSecret / Vault).
3. **Mounting the source tree in production.** That's the
   `docker-compose.override.yml` behavior. Production runs
   `docker compose -f docker-compose.yml up -d` only.
4. **Reordering the raw K8s manifests.** The `00-` to `07-` prefix
   is the apply order. `kubectl apply -f deploy/k8s/` relies on
   lex order; namespace must exist before the ConfigMap that
   references it.
5. **Skipping `kubectl rollout restart` after editing the raw
   ConfigMap / Secret.** The raw manifests' `checksum/config`
   annotation is empty (`""`); only the chart fills it in via
   `sha256sum`. On raw, the operator restarts manually.
6. **Setting `SEMRAG_API_RUNS_WORKER=true` in `03-pvc.yaml`'s
   ConfigMap.** The Deployment hard-codes the env var as a literal
   (`value: "false"`), not via `configMapKeyRef`. A typo in the
   ConfigMap can't accidentally re-enable the in-process worker.
   Don't try to "improve" that by moving it to the ConfigMap.
7. **Using `--workers >1` on uvicorn in the Dockerfile.** One
   uvicorn worker is correct for the FastAPI lifespan / startup-
   validation path. Horizontal scale lives in the Deployment
   (`replicaCount`), not in `--workers`.
8. **Reusing the dev compose for CI.** The override mounts
   `./src` and the source tree is the developer's working copy
   with uncommitted changes. CI builds a fresh image from
   `Dockerfile` and runs the production compose against it.
9. **Shipping a Dashboard JSON whose `datasource.uid` doesn't match
   the provisioned datasource.** Grafana provisions with a fixed uid
   (`prometheus`); the JSON files in
   `deploy/monitoring/grafana/dashboards/` reference it. Editing
   panels without updating `datasource.uid` breaks the panel.
10. **Treating `NOTES.txt` output as authoritative.** It's a
    convenience — `helm status semrag` and `kubectl get pods` are
    the real source of post-install state.
11. **Adding a `SEMRAG_*` env var to a K8s manifest without adding
    it to `Settings`.** The validator won't know about it; the
    startup probe (`/readyz`) will report misconfigurations as
    `ok` because the underlying pydantic-settings field is just
    absent. Always edit `src/semrag/core/settings.py` first.
12. **Disabling the ConfigMap rollout annotation to "make CI
    faster".** The annotation triggers a rolling restart on
    ConfigMap content change. Removing it means rolling out a new
    config requires an operator to manually `kubectl rollout
    restart`.

---

## 12. Architectural boundaries

What this directory **does not** do:

- **Define application configuration.** Every `SEMRAG_*` env var
  is declared in `Settings` (`core/settings.py`). The deploy
  shapes feed env vars; they do not own the schema.
- **Run the indexing job in-process on the API.** The API ships
  with `SEMRAG_API_RUNS_WORKER=false`; the worker is a separate
  process (Compose `worker` service, raw K8s
  `07-worker-deployment.yaml`, Helm `worker-deployment.yaml`).
- **Bundle a database in production.** Postgres is in the
  developer Compose stack only. K8s / Helm assume an operator-
  managed Postgres (or another SQLAlchemy-compatible DB).
- **Manage application secrets.** The chart renders a placeholder
  Secret only when `existingSecret` is empty. Production always
  sets `existingSecret`.
- **Render dashboards outside of Grafana.** The JSON files in
  `deploy/monitoring/grafana/dashboards/` are the source of
  truth; `provider.yml` provisions them at startup.
- **Wire a managed Prometheus / Grafana / Alertmanager.** The
  configuration in `deploy/monitoring/*` ships with the dev
  defaults (Compose service names, placeholder webhook URLs,
  `prometheus:9090`). Operators wire to managed endpoints by
  editing the YAML.
- **Provide a CI / CD pipeline.** The deploy assets are outputs
  a pipeline applies, not the pipeline itself.
- **Implement a CI / cluster-admin tool.** `kubectl`, `helm`,
  `docker` / `docker compose` are the operators; `deploy/` is
  configuration they consume.

What it **does** own:

- The multi-stage production `Dockerfile` and the
  `docker-compose.yml` shape (including the dev `override.yml`).
- The eight numbered raw Kubernetes manifests.
- The `deploy/helm/semrag/` chart (values, helpers, templates,
  `NOTES.txt`).
- The Prometheus scrape config, alert rules, alertmanager routing,
  Grafana datasource + dashboard provisioning.
- The post-install verification commands (`helm template |
  kubeconform`, `NOTES.txt`).
- The graceful-shutdown wiring (`preStop` sleeps, `terminationGracePeriodSeconds`).
- The probe paths / timings that match the API's
  health-endpoint contract.

---

## 13. Related documentation

- [`DEPLOYMENT.md`](../DEPLOYMENT.md) — end-to-end deployment
  procedures: bringing up the stack on every supported shape, the
  full environment-variable reference, the production checklist,
  and the troubleshooting recipes. This README complements, does
  not replace, `DEPLOYMENT.md`.
- [`docs/README.md`](../docs/README.md) — documentation index;
  the deployment docs sit in the "Repository" section.
- [`docs/CODEBASE_GUIDE.md`](../docs/CODEBASE_GUIDE.md) — file-level
  catalog including `deploy/`'s tree and the per-`values.yaml` key
  summary.
- [`docs/developer_guide.md`](../docs/developer_guide.md) — the
  configuration contract: how `core/settings.py` validates settings
  at startup, how env-var-driven configuration reaches the
  application code, the per-process resource budgets.
- [`docs/operations/`](../docs/operations/) — on-call runbooks
  (`database-outage.md`, `llm-outage.md`, `embedding-failure.md`,
  `indexing-failure.md`, `job-backlog.md`, `high-latency.md`,
  `memory-exhaustion.md`, `rollback.md`, `escalation.md`). The
  asset-level debugging matrix in
  [§10](#10-common-debugging-paths) is intentionally a complement
  to these — the runbooks diagnose by symptom; the matrix diagnoses
  by asset.
- [`src/semrag/api/health.py`](../src/semrag/api/health.py) — the
  `/livez`, `/readyz`, `/version` endpoints every probe in this
  directory points at.
- [`src/semrag/core/settings.py`](../src/semrag/core/settings.py) —
  the env-var schema; `DEPLOYMENT.md` §"Environment variables" is a
  less complete copy of the same field list.
- [`src/semrag/core/metrics.py`](../src/semrag/core/metrics.py) —
  the application's metric registry; the Prometheus rules and
  Grafana dashboards in `deploy/monitoring/` were authored against
  this registry.
- [`src/semrag/jobs/worker.py`](../src/semrag/jobs/worker.py) — the
  indexing worker loop; `python -m semrag.worker` is the entry
  point every `worker-*` deployment uses.
- [`Makefile`](../Makefile) — the `docker-build` / `docker-run`
  targets (image tag `${PROJECT}:dev` by default).

---

## Skipped (and why)

- **An ECS / Fargate / Cloud Run deploy shape.** The repo is
  on Kubernetes + Compose only; managed-container alternatives
  are out of scope until a deployment target is requested. The
  pattern would be: reuse `Dockerfile`, write a `deploy/<target>/`
  with the same env-var contract.
- **A managed-database operator inside the chart.** Postgres is
  the production assumption but it is **not** bundled into the
  K8s / Helm shapes. The chart documents the operator-managed
  PostgreSQL assumption in `values.yaml` and assumes the operator
  sets `SEMRAG_DATABASE_URL` to the managed DB's URL.
- **A multi-region / HA topology.** Postgres HA (Patroni /
  CloudNativePG), read replicas, Redis Sentinel / Cluster,
  cross-region failover — none are wired into any deploy shape.
  Add when the production environment requires it.
- **A GitOps template (ArgoCD / Flux).** The Helm chart is
  GitOps-friendly (`helm template` produces deterministic
  output), but no ArgoCD Application / Flux Kustomization ships
  in-tree. Add when the platform team picks a GitOps tool.
- **In-cluster Postgres backups.** The PVC holds the data; the
  Postgres container in the Compose stack does not run scheduled
  backups. Operators on K8s use Velero / pgBackRest / a managed
  snapshot mechanism against the managed DB. Not a deliverable.
- **TLS termination other than NGINX.** The Ingress template uses
  NGINX-style annotations and a commented TLS stanza. Operators
  on `gce` / `alb` swap `ingressClassName` and TLS config. A
  cert-manager `Certificate` resource is the production TLS path
  but is not in the chart.
- **Service-mesh integration (Istio / Linkerd).** The raw
  manifests don't add mesh sidecar annotations. The Service is
  `ClusterIP`, which any mesh can route to. Add when a mesh is
  installed.
- **SealedSecret / ExternalSecret / SOPS templates.** The chart
  assumes the operator picks their own Secret manager; bundled
  templates would force a tool choice on operators using a
  different one.
- **Canary / blue-green deployment strategies.** The raw manifest
  uses `RollingUpdate` with `maxUnavailable=0, maxSurge=1`. A
  canary (e.g. `Argo Rollouts`) or blue-green setup is a
  deployment-pattern change, not a chart change. Out of scope.
- **An in-cluster Prometheus / Alertmanager / Grafana.** The
  monitoring configuration assumes an external stack
  (Prometheus running elsewhere, scraping the API's `/metrics`).
  Bundling the Prometheus StatefulSet would be a heavier change;
  operators on managed Prometheus / Grafana Cloud can point their
  scrapers at the Service.
- **Database schema migrations as a Job in the chart.** Alembic
  runs separately (`alembic upgrade head`); the chart does not
  include a pre-deploy Job. The CI / operator's job is to run
  Alembic against the database *before* `helm upgrade`.

---

## Add when

- **A new deployment target.** Add a directory under `deploy/`
  named after the target (e.g. `deploy/ecs/`, `deploy/nomad/`).
  Reuse the `Dockerfile`; translate the env-var contract to the
  target's resource model.
- **A `helm test` suite.** Add `templates/tests/*.yaml` with
  `helm.sh/hook: test` Pod specs that hit `/readyz`. Add when
  the chart's behavior is worth testing in CI.
- **A `kubeconform` GitHub Action.** Run `kubeconform -strict
  -summary deploy/k8s/*.yaml` and `helm lint deploy/helm/semrag/`
  in CI. Add when the deploy surface stabilizes.
- **A migration Job.** Add `deploy/helm/semrag/templates/migration-job.yaml`
  with `helm.sh/hook: pre-upgrade` and an `alembic upgrade head`
  command. Add when Alembic is wired to run as part of the
  release pipeline rather than a separate CI step.
- **An Argo Rollouts resource.** Replace the raw Deployment's
  strategy with a `Rollout` (canary / blue-green). Add when the
  rollout strategy is worth moving beyond the simple
  RollingUpdate default.
- **A Prometheus Operator `ServiceMonitor`.** The Prometheus
  scrape config lives in plain Prometheus YAML in
  `deploy/monitoring/prometheus/prometheus.yml`. A `ServiceMonitor`
  CRD is the equivalent for the Prometheus Operator. Add when the
  Operator is the chosen scrape surface.
- **A multi-cluster / multi-region chart.** Add a `values-multi.yaml`
  or a separate `deploy/helm/semrag-region/` chart with
  per-region overrides (DB URL, ingress host, etc.). Add when the
  deployment spans more than one region.
- **PodDisruptionBudgets.** The chart doesn't ship one by
  default; PDBs are environment-specific (minAvailable thresholds
  depend on replica count and quorum requirements). Add via an
  environment-specific values overlay.
- **A NetworkPolicy.** Outgoing traffic from the API pod to the
  LLM / embedding / DB / Redis hosts is unrestricted. A
  `NetworkPolicy` with explicit egress allow-lists is the
  production hardening path; add when the platform's CNI
  enforces them.
- **A HorizontalPodAutoscaler for the worker.** `worker.autoscaling.*`
  is in `values.yaml` but no template renders it. Add a
  `templates/worker-hpa.yaml` mirroring the existing `hpa.yaml`,
  gated by `worker.autoscaling.enabled`.
