# Release Process

> How a SemRAG release ships from source to production. Concise.
> Checklist-driven. Operational. Describes only what the repository
> **currently ships**. Where a step is omitted, it is marked
> **Not implemented** — do not invent the missing step.
>
> **Audience:** release engineers, on-call operators, the maintainer
> who signs off a release.
>
> **Read alongside:** [`DEPLOYMENT.md`](./DEPLOYMENT.md),
> [`deploy/README.md`](./deploy/README.md),
> [`docs/ENGINEERING_PLAYBOOK.md`](./docs/ENGINEERING_PLAYBOOK.md),
> [`docs/operations/`](./docs/operations/),
> [`docs/developer_guide.md`](./docs/developer_guide.md),
> [`CHANGELOG.md`](./CHANGELOG.md), [`AGENTS.md`](./AGENTS.md).

---

## 1. Purpose

The operational release procedure for SemRAG: what to verify, in
what order, and how to roll back. Releases are the moments when
code, configuration, schema, image, and infrastructure must agree.

If you are not preparing a release, this document is not for you.
Use [`docs/ENGINEERING_PLAYBOOK.md`](./docs/ENGINEERING_PLAYBOOK.md)
for day-to-day engineering.

---

## 2. Release philosophy

SemRAG's release model is deliberately conservative. The primary
safety properties are **grounded answers** (citation-enforced) and
**graceful shutdown** (in-flight requests drained on SIGTERM).
Releases prioritize those two properties above feature velocity.

Five rules drive every release:

1. **Reproducibility first.** Reproducible from `git describe` +
   `pyproject.toml` (`version`) + lock files. The image is
   multi-stage with a pinned base (`python:3.12-slim-bookworm`,
   `Dockerfile:8-9`).
2. **Configuration is code.** Every `SEMRAG_*` env var is declared
   in `src/semrag/core/settings.py`. A deploy shape that bypasses
   the validator is a defect, not a feature.
3. **Configuration validates at startup.** The FastAPI lifespan
   runs `validate_runtime_config()` and fails fast with an
   actionable list of every misconfig. The release verification
   command is the same one CI runs on every PR.
4. **Graceful shutdown over fast kill.** The `preStop` sleep (5 s)
   and `terminationGracePeriodSeconds` (30 s) are deliberate;
   every release assumes traffic can be drained.
5. **Rollback beats hotfix.** Per
   [`docs/operations/rollback.md`](./docs/operations/rollback.md):
   **when in doubt, roll back.**

---

## 3. Versioning

The version is declared in three places:

- `pyproject.toml` (`version = "0.1.0"`)
- `deploy/helm/semrag/Chart.yaml` (`appVersion: "0.1.0"`)
- `deploy/helm/semrag/values.yaml` (`image.tag`)

CI injects git commit and build timestamp into `/version` via
`SEMRAG_GIT_COMMIT` / `SEMRAG_BUILD_TIME` (`DEPLOYMENT.md`
§"Health & version endpoints"). `/version` is the operational
source of truth for "what's running".

**Scheme.** Three-part `MAJOR.MINOR.PATCH` literal. **Not
implemented:** a formal SemVer / CalVer contract; automated
version bump; git tags created by the release process;
`CHANGELOG.md` linter.

**What to do for a release.** Move all three files to the same
string; write a `CHANGELOG.md` entry under a dated `##` heading
matching the existing format; record the version in `/version`
after deploy.

---

## 4. Branch expectations

**Not implemented.** The repo does not document a release branch
model. CI in `.github/workflows/ci.yml` runs on `push` and
`pull_request` to `main` only. There is no `release/*` convention,
no Git Flow, no GitHub Flow contract. **Releases ship from
`main`** — treat `main` as the always-green branch.

If you need a hotfix flow, the operational equivalent is: revert
the offending commit in `main` (the rollback runbook assumes an
image-tag rollback; reverting in `main` and re-running CI is the
source-control equivalent).

---

## 5. Pre-release checklist

- [ ] All PRs intended for the release are merged to `main`.
- [ ] `CHANGELOG.md` has a new dated `## <version>` heading.
- [ ] `pyproject.toml` `version`, `Chart.yaml` `appVersion`,
      `values.yaml` `image.tag` all match the release version.
- [ ] `git describe` returns the expected tag / short SHA.
- [ ] No `_deployment_notes:` or `TODO: cut release` markers
      remain in `CHANGELOG.md`.

---

## 6. Local verification checklist

```bash
make check            # ruff + black --check + mypy
make test             # pytest tests/unit -q
make coverage         # coverage run / report / xml

# Exercise the Settings validator against production-shaped env
SEMRAG_ENVIRONMENT=production \
SEMRAG_API_KEYS=$(openssl rand -hex 24) \
SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION=true \
SEMRAG_DATABASE_URL=postgresql+psycopg2://semrag:semrag@localhost:5432/semrag \
SEMRAG_LOG_FORMAT=json \
  python -c "from semrag.core.settings import Settings; print(Settings().model_dump_json(indent=2))"
```

The `Settings()` dump exercises the same code path that
`validate_runtime_config()` runs at process start
(`api/startup_validation.py`). If construction in production mode
raises, the validator will fail.

---

## 7. Test checklist

| Suite           | Command                                        | Purpose                                       |
| --------------- | ---------------------------------------------- | --------------------------------------------- |
| Unit            | `make test`                                    | Per-module correctness; Protocols; Settings.  |
| Coverage        | `make coverage`                                | Lines touched by the release.                 |
| Eval (optional) | `python run_benchmarks.py --tag <release-tag>` | Retrieval NDCG regression; LLM judge drift.   |
| Performance     | `python tests/perf/run_perf_suite.py`          | p50 / p95 / p99 drift on the live server.     |
| Load smoke      | `python tests/load/run_load_test.py --url …`   | Cold start; rate-limit behavior.              |

**Required before tagging:** unit + coverage green.

**CI coverage target.** **Not implemented.** CI uploads
`coverage.xml` to Codecov but does not gate on a minimum
threshold (`fail_ci_if_error: false`,
`.github/workflows/ci.yml:69`). Treat a coverage drop as a
review concern, not a CI block.

**Integration tests.** **Not implemented.** The repo ships
unit, load, and perf; no `tests/integration/` with gating.
See [`tests/README.md`](./tests/README.md).

---

## 8. Documentation checklist

- [ ] `CHANGELOG.md` has the new release under a dated
      heading.
- [ ] `DEPLOYMENT.md` env-var table matches `Settings`
      (`src/semrag/core/settings.py`).
- [ ] New env vars are reflected in `.env.example`.
- [ ] New env vars are documented in the Helm `values.yaml`
      comment block.
- [ ] New CLI flags are in `RUN_API.md` if they affect the
      CLI surface (not applicable to API-only changes).

The release process **does not gate on documentation**. The
single-source-of-truth pattern in this repo's docs means
most changes do not require a docs PR.

---

## 9. Docker validation

```bash
make docker-build IMAGE=semrag:0.1.0
docker inspect --format '{{json .Config.Labels}}' semrag:0.1.0 | jq
docker inspect --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}' semrag:0.1.0 | jq

docker run --rm --name semrag-validate -p 8000:8000 --env-file .env semrag:0.1.0 &
sleep 8
curl -fsS http://localhost:8000/livez
curl -fsS http://localhost:8000/readyz | jq
curl -fsS http://localhost:8000/version | jq
docker stop semrag-validate
```

Image invariants (`Dockerfile`):

- Multi-stage, non-root (`UID 1001`, `GID 1001`).
- tini as PID 1, SIGTERM-aware (`ENTRYPOINT ["/usr/bin/tini", "--"]`).
- `HEALTHCHECK` against `/health` (legacy alias for `/livez`).
- `WORKDIR /app`, `EXPOSE 8000`.

CI builds an image on every PR / push to `main`
(`.github/workflows/ci.yml:80-97`, `tags: semrag:ci`) but does
**not** push or tag with the release version. Pushing is a manual
step.

---

## 10. Helm validation

```bash
helm lint deploy/helm/semrag/
helm template semrag deploy/helm/semrag/ \
  --set config.environment=production \
  --set existingSecret=semrag-secrets \
  | kubeconform -strict -summary -

helm diff upgrade semrag deploy/helm/semrag/ \
  --namespace semrag \
  --values <prod-values.yaml>   # requires helm-diff plugin
```

After `helm template` you should expect:

- Two API replicas (`values.yaml:10`) with
  `maxUnavailable=0, maxSurge=1`
  (`templates/deployment.yaml:13-16`).
- Probes: `/livez` liveness, `/readyz` readiness, `/livez`
  startup with `failureThreshold: 18` (90 s startup budget).
- `preStop` sleep + `terminationGracePeriodSeconds: 30`.
- `existingSecret: ""` defaults render
  `templates/secret.yaml` with `REPLACE_ME` placeholders;
  production must override.

---

## 11. Kubernetes validation

```bash
kubeconform -strict -summary deploy/k8s/*.yaml
kubectl apply --dry-run=client -f deploy/k8s/

# Confirm ConfigMap / Secret wiring
kubectl create secret generic semrag-secrets \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -hex 16) \
  --from-literal=SEMRAG_API_KEYS=$(openssl rand -hex 24) \
  --dry-run=client -o yaml | kubectl apply -f -
```

Two known rough edges in the raw manifests:

1. `deploy/k8s/04-deployment.yaml:61` ships
   `image: REPLACE_WITH_IMAGE`. The raw manifests are
   templates; Kustomize / sed before apply.
2. `metadata.annotations.checksum/config: ""`
   (`04-deployment.yaml:39`) is intentionally empty for the
   raw manifests — Helm's annotation
   (`templates/deployment.yaml:26`) populates it
   automatically. For raw manifests, run
   `kubectl rollout restart deployment/semrag -n semrag`
   after a ConfigMap / Secret change.

The Helm chart is the recommended install path
(`DEPLOYMENT.md:140-208`). Raw manifests exist for clusters
without Helm.

---

## 12. Database migration checklist

Alembic (`alembic/`, `src/semrag/storage/postgres/`). Five
migrations ship today:

```
0190b83318c7_add_full_text_search_to_chunks.py
692caefcb795_initial_schema.py
d5b1cc8385d4f2b7_add_indexing_jobs_table.py
e28abd752a06b245_add_api_keys_table.py
f5e3cd5b5b42_add_chunks_table_for_pgvector.py
```

For every migration in the release:

- [ ] `alembic upgrade head` succeeds on a fresh DB.
- [ ] `alembic downgrade -1` succeeds (always include a working
      `downgrade()` — the rollback runbook depends on it).
- [ ] `alembic upgrade head` succeeds **again** on the
      downgraded DB.
- [ ] On a non-empty DB (the staging replica), the migration
      completes without errors and the new columns / indexes
      are populated.
- [ ] pgvector is installed (`CREATE EXTENSION IF NOT EXISTS
      vector;`). `/readyz` fails if the extension is missing.
- [ ] Embedding dimension did not change. If it did, the
      pgvector column must be re-created (`Vector(N)` has a
      fixed dimension; the dimension lives in
      `SEMRAG_EMBEDDING_DIMENSION`).

**Schema rollbacks.** Per the rollback runbook: never roll back
the schema without a corresponding code rollback
(`docs/operations/rollback.md:42-60`). Older code may not
understand the new columns and will crash at runtime.

---

## 13. Configuration review

Verify every env var that landed in this release against the
production checklist in `DEPLOYMENT.md:304-322`:

- [ ] `SEMRAG_ENVIRONMENT=production` (validated at startup).
- [ ] `SEMRAG_API_KEYS` contains at least one non-default key.
- [ ] `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION=true` (default).
- [ ] `SEMRAG_DATABASE_URL` points at Postgres with pgvector.
- [ ] `SEMRAG_LLM_BASE_URL` / `SEMRAG_EMBEDDING_BASE_URL`
      resolve to a healthy provider (probe at startup by
      default).
- [ ] `SEMRAG_LOG_FORMAT=json` (the production log pipeline
      ingests JSON).
- [ ] `SEMRAG_TRUSTED_HOSTS` set when behind a reverse proxy
      that rewrites `Host`.
- [ ] `SEMRAG_CORS_ORIGINS` set when consumed from a browser.
- [ ] Secrets live in a Secret manager (SealedSecret /
      ExternalSecret / Vault) — never in `values.yaml`.
- [ ] `SEMRAG_API_RUNS_WORKER=false` on the API replicas
      (`04-deployment.yaml:67-69`); the worker is a separate
      Deployment.

The Settings model is the source of truth; diff §6's dump
against `settings.py`.

---

## 14. Security review

- [ ] No raw credentials committed. Run `git grep -nE
      "(POSTGRES_PASSWORD|SEMRAG_API_KEYS|password|api_key)="`
      and review against `*.example`, `.env.example`, and the
      chart's `REPLACE_ME` placeholders.
- [ ] `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION=true` (the startup
      validator rejects the literal `dev-local-key` in
      production — `developer_guide.md` §8).
- [ ] API keys are hashed, not stored plaintext
      (`APIKeyStore` ABC requires hashing).
- [ ] The image runs non-root (`runAsUser: 1001`,
      `Dockerfile:46-47`).
- [ ] The container's filesystem is read-only
      (`values.yaml:49`); only `/app/data` is writable (PVC).
- [ ] All Linux capabilities dropped; `allowPrivilegeEscalation:
      false` (`values.yaml:50-52`).
- [ ] Structured-log redaction on by default
      (`semrag.core.redaction.redact_payload()`).
- [ ] TLS terminated at the Ingress / proxy.
- [ ] `/livez` and `/readyz` are unauthenticated by design
      (`DEPLOYMENT.md:14-15`); the orchestrator probes must
      not require a key.

---

## 15. Performance regression checklist

- [ ] Run `python tests/perf/run_perf_suite.py` against the
      release image. Compare against the previous release;
      flag any p95 increase > 10% on
      `/v1/projects/.../query` or `/v1/projects/.../answer`.
- [ ] Run `python run_benchmarks.py --tag <release-tag>` if
      retrieval weights changed in this release.
- [ ] Confirm the embedding dimension and provider are
      unchanged. Switching models typically resets the
      embedding cache (chunk_id is content-derived; same text
      → same id, but vectors differ).

The perf suite thresholds are operator-configurable. There is
no enforced pass / fail bar today (**Not implemented**: CI-
enforced perf thresholds).

---

## 16. Build artifacts

| Artifact                        | Producer                  | Pointer                                           |
| ------------------------------- | ------------------------- | ------------------------------------------------- |
| Source (git tag, optional)      | git                       | `<commit-sha>`                                    |
| Container image                 | `make docker-build` / CI  | `semrag:<version>` (or registry override)         |
| Helm chart (in-tree)            | The repo                  | `deploy/helm/semrag/` (`Chart.yaml:10-11`)        |
| Raw K8s manifests (in-tree)     | The repo                  | `deploy/k8s/*.yaml`                               |

**Container registry publishing.** **Not implemented.** CI
builds (`tags: semrag:ci`) but does not push; the release
engineer pushes manually after §9 passes.

**Provenance / SBOM / signed images / changelog auto-build.**
**Not implemented.**

---

## 17. Deployment order

The worker and the API share the same image but talk to the same
Postgres through different roles. Recommended order:

1. **Provision / verify infrastructure.** Postgres with pgvector,
   persistent volume, Secrets, ConfigMap, Ingress. Helm:
   `helm install` with `--set existingSecret=...`. Raw K8s: apply
   manifests in numeric order (`00-namespace.yaml` →
   `07-worker-deployment.yaml`).

2. **Run Alembic migrations** before any API pod that needs the
   new schema boots:
   ```bash
   kubectl create job semrag-migrate-$(date +%s) \
     --image=<release-image> --namespace=semrag \
     -- /bin/sh -c "alembic upgrade head"
   # Option B: run inside a transient pod.
   ```
   There is **no built-in `Job` template** for migrations today;
   one-shot pods / Jobs are scripted.

3. **Deploy the API.** Rolling update with
   `maxUnavailable=0, maxSurge=1`
   (`deploy/k8s/04-deployment.yaml:23-26`,
   `deploy/helm/semrag/templates/deployment.yaml:13-16`).
   Watch `kubectl rollout status deployment/semrag -n semrag`.

4. **Deploy the worker.** Same image; same Helm release or
   `07-worker-deployment.yaml`. Workers run indexing jobs from
   the Postgres-backed job store
   (`SEMRAG_JOB_STORE_BACKEND=database`,
   `docker-compose.yml:215-216`). The worker's K8s healthcheck
   is `test -d /proc/1` (`docker-compose.yml:227-229`) —
   readiness is process liveness, not LLM reachability.

5. **Watch `/readyz`** for both old and new pods during the roll.
   Readiness `failureThreshold: 1` demotes the pod from the
   Service on a single 503 (`04-deployment.yaml:228-235`).

---

## 18. Smoke tests

```bash
SVC=https://semrag.example.com

# 1. Liveness + readiness + version
curl -fsS $SVC/livez
curl -fsS $SVC/readyz | jq     # status: "ready"
curl -fsS $SVC/version | jq    # version matches release

# 2. Auth required (expect 401 / 403)
curl -i $SVC/v1/projects/x/index | head -1
# 3. Auth granted (expect 2xx)
curl -i -H "x-api-key: $API_KEY" $SVC/v1/projects/x/index | head -1

# 4. End-to-end answer
curl -fsS -H "x-api-key: $API_KEY" -H "content-type: application/json" \
  -d '{"question":"What does the indexer do?"}' \
  $SVC/v1/projects/x/answer | jq
# Confirm "answer" + "citations"[] + grounding.
```

If any check fails, **stop and roll back** per §20 before
investigating further. Smoke failures on a fresh release are
almost always misconfig (env vars), not code.

---

## 19. Post-release validation

- [ ] `/version` on every Pod returns the release tag (compare
      with `git describe`).
- [ ] Prometheus dashboards show expected query latency
      (`semrag_http_request_duration_seconds`) within ±10% of
      pre-release baseline.
- [ ] No `CrashLoopBackOff` in the last 30 minutes of pod history.
- [ ] Alertmanager has no firing alerts except expected.
- [ ] Indexing backlog is empty
      (`semrag_indexing_jobs_queued_total` flat).
- [ ] Embedding cache hit-rate is non-zero (cold cache is
      expected on the first hour; flat at zero is a bug).
- [ ] A canned query returns a non-empty, cited answer.
- [ ] `helm history semrag -n semrag` shows the new revision.

Dashboards: `deploy/monitoring/grafana/dashboards/`. Alerts:
`deploy/monitoring/prometheus/rules/semrag.yml`. See
[`deploy/README.md`](./deploy/README.md).

---

## 20. Rollback procedure

Full runbook:
[`docs/operations/rollback.md`](./docs/operations/rollback.md).
Summary, in execution order:

### Helm

```bash
helm history semrag -n semrag        # confirm known-good revision
helm rollback semrag -n semrag
kubectl rollout status deployment/semrag-api -n semrag
curl -s https://semrag.example.com/version | jq .version   # confirm
```

### Raw Kubernetes

```bash
kubectl rollout undo deployment/semrag-api -n semrag
kubectl rollout status deployment/semrag-api -n semrag
```

### Compose

```bash
docker compose down
git checkout <previous-good-tag>
make docker-build IMAGE=semrag:<previous-tag>
docker compose up -d
```

**Worker rollback.** Same command, same image version. Worker
state lives in Postgres; rolling back has no side effects beyond
"jobs in flight are restarted by the previous-version worker".

**Index data.** Vector and IR indexes live in Postgres; they are
not part of the application deployment
(`docs/operations/rollback.md:62-78`). A rolled-back code version
sees the same indexes. Re-index only if the indexing algorithm
itself changed.

---

## 21. Emergency rollback

When the standard rollback is too slow (e.g. the new version is
serving 500s on `/livez`):

**Disable the canary (Helm).**

```bash
helm rollback semrag -n semrag --wait --cleanup-on-fail
```

**Disable the new revision (raw K8s).**

```bash
kubectl scale deployment/semrag -n semrag --replicas=0
kubectl set image deployment/semrag semrag=<previous-image> -n semrag
kubectl rollout status deployment/semrag -n semrag
```

**Disable the whole API (last resort).**

```bash
kubectl scale deployment/semrag -n semrag --replicas=0
# Operators see a 503 from the Service. Document and escalate.
```

**Worker emergency stop.** Workers do not serve HTTP.

```bash
kubectl scale deployment/semrag-worker -n semrag --replicas=0
```

**Compose (single host).** `docker compose down` brings the whole
stack; `docker compose up -d` with the previous tag brings it back.
Use the [`memory-exhaustion`](./docs/operations/memory-exhaustion.md)
or [`high-latency`](./docs/operations/high-latency.md) runbooks if
the issue is resource-shaped rather than version-shaped.

**Always:** post-rollback, capture the failing pod's previous logs
(`kubectl logs --previous`), the failing `/readyz` body, and the
`/version` from the bad Pod. Add to the postmortem.

---

## 22. Common release mistakes

1. **Bumping the version in `pyproject.toml` only.** Helm
   `Chart.appVersion` and `values.yaml` `image.tag` will
   drift. All three must move together.
2. **Skipping `helm template | kubeconform`.** Catches YAML
   drift before it lands.
3. **Setting `POSTGRES_PASSWORD` / `SEMRAG_API_KEYS` directly
   in `values.yaml`.** Use `existingSecret:` for production.
4. **Running the API without applying migrations first.** The
   API fails `/readyz` against a DB missing the new columns.
5. **Setting `SEMRAG_API_RUNS_WORKER=true` on API replicas**
   to "save resources". The API process has one worker
   (`Dockerfile:69`); indexing in the API path is what the
   `job-backlog` runbook exists to recover from.
6. **Adding a new `SEMRAG_*` env var in `deploy/` without
   updating `core/settings.py`.** The validator won't see it;
   divergence becomes inevitable.
7. **Shipping an `image:` override without pinning
   `imagePullPolicy`.** `IfNotPresent` (`values.yaml:19`) is
   the production-safe default; `Always` re-pulls on every
   pod restart and can stall rollouts.
8. **Updating a ConfigMap without restarting the
   Deployment.** The raw manifests' `checksum/config` is
   empty by design; Helm's is filled. Without `kubectl
   rollout restart`, the new config does not take effect.
9. **Treating `/health` as the readiness contract.** The
   `Dockerfile:65-66` `HEALTHCHECK` uses `/health` (an alias
   for `/livez`); the kubelet's readiness probe uses
   `/readyz`. Different probes — document which is which.
10. **Skipping post-release validation.** The first 30 minutes
    after a release is when regressions surface. If the
    on-call leaves the watch, the next page is the next day.

---

## 23. Release sign-off checklist

The release is signed off when **all** of the following are
true. The release engineer (or designated maintainer) is the
signer.

- [ ] §5–§8 pre-release / local / test / documentation checks
      green.
- [ ] §9 Docker validation passes against the release image.
- [ ] §10 Helm validation passes with the production values.
- [ ] §11 K8s validation passes against the target cluster
      (dry-run at minimum).
- [ ] §12 migration checks pass against a non-empty DB.
- [ ] §13 configuration review matches production Secrets
      and ConfigMaps.
- [ ] §14 security review green.
- [ ] §15 performance checks within tolerance (or documented
      exceptions).
- [ ] §16 artifacts tagged and pushed (image to production
      registry; Helm chart revision noted; git tag applied
      if used).
- [ ] §17 deployment order executed; all five sub-steps
      completed.
- [ ] §18 smoke tests pass against the live Service.
- [ ] §19 post-release validation green for at least 30
      minutes.
- [ ] `CHANGELOG.md` entry published.
- [ ] Release announcement sent (Slack / email / ticket per
      team convention).

Record the **version**, **git SHA**, **release engineer**,
**sign-off date/time** in the announcement.

---

## 24. Related documentation

- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — deployment shapes, env
  vars, production checklist, troubleshooting.
- [`deploy/README.md`](./deploy/README.md) — Docker / Compose /
  K8s / Helm / monitoring asset map.
- [`docs/ENGINEERING_PLAYBOOK.md`](./docs/ENGINEERING_PLAYBOOK.md)
  — PR verification checklist (§36); cross-reference index
  (§41).
- [`docs/operations/`](./docs/operations/) — runbooks:
  [`rollback.md`](./docs/operations/rollback.md),
  [`escalation.md`](./docs/operations/escalation.md), plus
  the seven failure-mode runbooks (database, llm, embedding,
  indexing, jobs, latency, memory).
- [`docs/developer_guide.md`](./docs/developer_guide.md) —
  configuration (§10), security (§8), observability (§9),
  debugging (§14), anti-patterns (§15).
- [`docs/CODEBASE_GUIDE.md`](./docs/CODEBASE_GUIDE.md) — file
  ownership; "Where do I change X?" table.
- [`docs/README.md`](./docs/README.md) — documentation index.
- [`AGENTS.md`](./AGENTS.md) — contribution policy and
  priority order.
- [`CHANGELOG.md`](./CHANGELOG.md) — release history.

---

## Skipped (and why)

Release-process topics intentionally **not** included because
the repository does not implement them. Each is a "do not
invent" — if a future task seems to require one, escalate
rather than invent.

- **A release branch model.** §4. CI runs only on `main`;
  releases ship from `main`.
- **Formal SemVer / CalVer contract.** §3. The codebase
  uses `MAJOR.MINOR.PATCH` literals; bump rules are not
  codified.
- **Automated version bump / changelog generation.** §3, §8.
  The release engineer updates three files manually.
- **Container registry publishing in CI.** §16. CI builds
  but does not push.
- **Provenance / SBOM / signed images.** §16.
- **A migration `Job` template / Helm hook.** §17. Migrations
  are run via one-shot pods or Jobs that the release engineer
  scripts.
- **CI-enforced coverage / perf thresholds.** §7, §15.
  Codecov upload does not gate; the perf suite reports but
  does not gate.
- **An automated integration test suite.** §7. The `tests/`
  directory has unit / load / perf; no `tests/integration/`.
- **An automated release announcement.** §23. The release
  engineer announces manually per team convention.
- **Git tags created by the release process.** §5, §16. The
  release engineer tags manually if at all.

---

## Add when

Release-process topics to add **only when the corresponding
capability actually ships**. None are present today; the
absence is intentional.

- **Automated version bump from `CHANGELOG.md`.** Add when a
  tool derives `pyproject.toml` / `Chart.yaml` from the
  changelog.
- **CI-driven container publish.** Add when
  `.github/workflows/ci.yml` gains `push: true` + registry
  login.
- **Helm chart packaging and OCI registry publish.** Add
  when a `helm package` + `helm push` step lands.
- **A migration `Job` template / Helm pre-install hook.** Add
  when `deploy/k8s/` or `deploy/helm/semrag/templates/` ships
  a one-shot migration `Job`.
- **A formal SemVer / CalVer contract.** Add when
  `pyproject.toml` or `Chart.yaml` enforces a bump rule.
- **Provenance / SBOM / cosign signing.** Add when an
  attestation step is wired into CI.
- **Integration tests gating releases.** Add when
  `tests/integration/` is populated and CI gates on it.
- **A `release.yml` workflow.** Add when an automated release
  pipeline ships.
- **A release-notes linter.** Add when a tool validates
  `CHANGELOG.md` against PRs.
- **Coverage / perf thresholds.** Add when Codecov and
  `tests/perf/` gain a pass / fail gate.