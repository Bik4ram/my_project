# SemRAG Documentation

The official documentation for **SemRAG**, a production-grade Retrieval-Augmented
Generation platform for semiconductor design repositories.

This index is the entry point for every other document in the project. If you
are new to the codebase, start here.

---

## Who should read this

This documentation is written for professional software engineers. The
intended audiences are:

- **Users** — engineers who want to index a repository and ask questions of
  it through the CLI or HTTP API.
- **Developers** — engineers extending SemRAG, adding parsers, chunkers,
  embedding or LLM providers, or storage backends.
- **Contributors** — engineers preparing pull requests and following the
  project's style, testing, and review conventions.
- **Operators** — SRE / platform engineers running SemRAG in development,
  Docker, Kubernetes, or Helm, and on-call responders using the runbooks.

Pick the learning path below that matches your role, or browse by section.

---

## How the documentation is organized

The documentation is split into six logical sections. Each has a stable URL,
an obvious scope, and links into the others where relevant.

| Section         | Path                                | Purpose                                                     |
| --------------- | ----------------------------------- | ----------------------------------------------------------- |
| **Guides**      | [`docs/guides/`](./guides/)         | Task-oriented how-to material, organized by job to be done. |
| **Reference**   | [`docs/reference/`](./reference/)   | Factual lookup material: configuration, CLI, API, schema.   |
| **Architecture**| [`docs/architecture/`](./architecture/) | Design deep-dives on the retrieval, agent, prompting, and observability subsystems. |
| **Operations**  | [`docs/operations/`](./operations/) | On-call runbooks, one per failure mode.                     |
| **Contributing**| [`docs/contributing/`](./contributing/) | Code style, testing, and release process for contributors. |
| **Repository**  | `src/semrag/<package>/README.md`    | Package-level maps of the source tree.                      |

---

## How to navigate

- **I'm new** — start with [`docs/GETTING_STARTED.md`](./GETTING_STARTED.md).
  It is a hands-on, lab-manual walkthrough from a fresh `git clone`
  through indexing a project and asking the first question. Then
  return to the *User* path below for the next steps.
- **I have a question** — read the [Reference](#reference) section. It is
  written so you can jump to the relevant page directly.
- **I want to understand the system** — read [ARCHITECTURE.md](../ARCHITECTURE.md),
  then the [Architecture](#architecture) section of this index for deeper
  material.
- **I'm on call** — go straight to the [Operations](#operations) runbooks.
- **I want to contribute** — read the [Contributing](#contributing) section.

---

## Learning paths

### New User

1. [Quick Start](../README.md) — install, index a project, ask a question.
2. [Running the API](../RUN_API.md) — launch the HTTP API locally.
3. [Local Postgres setup](../SETUP_POSTGRES.md) — set up the database.
4. *Coming in a future phase:* `docs/guides/indexing-a-repository.md`.

### Developer

1. [Architecture](../ARCHITECTURE.md) — the long-form picture.
2. [Engineering Playbook](./ENGINEERING_PLAYBOOK.md) — the day-to-day
   engineer's manual: how to add a parser, a provider, an agent tool,
   a router, a metric, a migration, etc. Start here before reading
   the architecture deep-dives.
3. [Repository map](#repository-map) — where each subsystem lives in the
   source tree.
4. *Coming in a future phase:* `docs/architecture/retrieval-pipeline.md`,
   `docs/architecture/agent-loop.md`, `docs/architecture/prompting.md`.
5. *Coming in a future phase:* `docs/guides/writing-a-new-parser.md`,
   `docs/guides/adding-an-embedding-provider.md`,
   `docs/guides/adding-an-llm-provider.md`.

### Contributor

1. [Repository Guidelines](../AGENTS.md) — contribution policy.
2. [Engineering Playbook](./ENGINEERING_PLAYBOOK.md) — how to add
   parsers, providers, etc., and the PR verification checklist
   (§36). Read this before opening a PR.
3. *Coming in a future phase:* `docs/contributing/code-style.md`,
   `docs/contributing/testing.md`,
   `docs/contributing/release-process.md`.
4. [Changelog](../CHANGELOG.md) — format your entry to match
   (the canonical release-history format). The full release
   procedure is in [RELEASE_PROCESS.md](../RELEASE_PROCESS.md).

### Operator

1. [Deployment](../DEPLOYMENT.md) — every supported environment.
2. [Engineering Playbook — Production debugging](./ENGINEERING_PLAYBOOK.md#26-production-debugging)
   — and the surrounding §27–§32 chapters for incident investigation, log
   reading, debug artifacts, Docker, Kubernetes, and Helm workflows.
3. *Coming in a future phase:* `docs/operations/README.md` — on-call runbook
   index. *(See also the existing runbook index at
   [`docs/operations/README.md`](./operations/README.md) — already populated
   with runbooks.)*
4. *Coming in a future phase:* `docs/reference/metrics.md`,
   `docs/architecture/observability.md`.

---

## Section summaries

### Guides

Task-oriented how-tos. Each guide answers *how do I do X?* rather than
*what is X?*. *(Coming in a future phase.)*

> **Placeholder:** the `docs/guides/` directory and its contents will be
> written in a later phase of the documentation roadmap. Until then, the
> existing root-level recipes in [RUN_API.md](../RUN_API.md) and
> [SETUP_POSTGRES.md](../SETUP_POSTGRES.md) cover the most common tasks.

### Reference

Factual, lookup-oriented material. Each page describes one axis — every
setting, every CLI flag, every API endpoint, every data model — without
narrative.

- *Coming in a future phase:* `docs/reference/configuration.md` — the full
  `SEMRAG_*` environment variable schema.
- *Coming in a future phase:* `docs/reference/cli.md` — `run_local_qa.py`
  and the installed `semrag` CLI.
- *Coming in a future phase:* `docs/reference/api.md` — every HTTP route,
  request shape, and response shape.
- *Coming in a future phase:* `docs/reference/data-model.md` — Postgres
  schema, pgvector columns, and graph schema.
- *Coming in a future phase:* `docs/reference/storage-backends.md` — when
  to pick memory, file, database, or Redis backends.
- *Coming in a future phase:* `docs/reference/metrics.md` — Prometheus
  metric catalog.

### Architecture

Design deep-dives. These explain *why* a subsystem is shaped the way it is
and where the seams are for extension.

- *Coming in a future phase:* `docs/architecture/retrieval-pipeline.md`,
  `docs/architecture/agent-loop.md`,
  `docs/architecture/prompting.md`,
  `docs/architecture/evaluation.md`,
  `docs/architecture/observability.md`,
  `docs/architecture/security.md`.

### Operations

On-call runbooks. One page per failure mode, each describing the symptom,
the diagnostic steps, and the remediation.

The runbook index already exists at
[`docs/operations/README.md`](./operations/README.md) and covers the
following failure modes today: database outage, embedding failure,
high latency, indexing failure, job backlog, LLM outage, memory
exhaustion, and rollback. Escalation is documented at
[`docs/operations/escalation.md`](./operations/escalation.md).

### Contributing

Style, testing, and release conventions for contributors. The project
treats contributions as production work: small commits, clear function
names, type hints, and an end-to-end passing pipeline before review.

- [Repository Guidelines](../AGENTS.md) — already exists at the repo root.
- *Coming in a future phase:* `docs/contributing/code-style.md`,
  `docs/contributing/testing.md`,
  `docs/contributing/release-process.md`.

### Repository

Package-level maps that live next to the code. Each `src/semrag/<package>/README.md`
explains that package's purpose, public entry points, and the design notes
that matter when extending it.

- [`src/semrag/README.md`](../src/semrag/README.md) — currently the
  source-package map.
- [`src/semrag/api/README.md`](../src/semrag/api/README.md) — the HTTP
  API package map (routers, auth, middleware, DI, health endpoints).
- [`src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md)
  — the hybrid retrieval pipeline (stages, fusion, ranking, rerank,
  diversity).
- [`src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md)
  — the indexing pipeline (scanner, parsers, IR, chunking, embedding).
- [`src/semrag/evaluation/README.md`](../src/semrag/evaluation/README.md)
  — the eval harness and benchmark sets.
- [`src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md)
  — the LLM abstraction, agent loop, and prompting.
- [`src/semrag/storage/README.md`](../src/semrag/storage/README.md) —
  the persistence adapters (Postgres, pgvector, experimental Neo4j).
- [`src/semrag/jobs/README.md`](../src/semrag/jobs/README.md) — the
  indexing job system (store, executor, worker).
- [`src/semrag/core/README.md`](../src/semrag/core/README.md) and
  [`src/semrag/services/README.md`](../src/semrag/services/README.md) —
  settings/observability and the service entry points.
- [`tests/README.md`](../tests/README.md) — the test suite map.
- [`deploy/README.md`](../deploy/README.md) — the deployment-assets
  map (Docker, Compose, K8s, Helm, monitoring).

---

## Documentation conventions

- **Plain English, professional tone.** No marketing prose, no emojis in
  body text. Code samples use fenced blocks with the appropriate language
  tag.
- **One fact in one place.** If a fact is in the reference section, the
  guides and architecture pages link to it instead of repeating it.
- **Every setting, every command, every endpoint is named once and indexed.**
  Use the reference section as the source of truth for syntax.
- **Relative links.** All links to other documents in this repository use
  repository-relative paths so they work from any checkout layout.
- **Code-first, prose-second.** Examples precede explanations. If a section
  is longer than its example, the example wins.
- **Truthfulness.** Documentation describes the codebase as it exists.
  Speculative features are not described as if they shipped. *(Coming in a
  future phase:* `docs/architecture/*.md` *will deep-dive the real
  subsystems only.)*

---

## Repository documentation map

Documents that already exist in this repository:

| Path                                            | Purpose                                                       |
| ----------------------------------------------- | ------------------------------------------------------------- |
| [`README.md`](../README.md)                     | Project pitch, quick start, env-var cheat sheet, development. |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md)         | Long-form architecture, data flow, extensibility points.       |
| [`DEPLOYMENT.md`](../DEPLOYMENT.md)             | Deployment across local, Docker, Compose, Kubernetes, Helm.   |
| [`RELEASE_PROCESS.md`](../RELEASE_PROCESS.md)   | Operational release procedure: checklists, rollback, sign-off. |
| [`AGENTS.md`](../AGENTS.md)                     | Contribution policy and repository guidelines.                |
| [`CHANGELOG.md`](../CHANGELOG.md)               | Release notes.                                                |
| [`RUN_API.md`](../RUN_API.md)                   | Local HTTP API launch recipe.                                 |
| [`SETUP_POSTGRES.md`](../SETUP_POSTGRES.md)     | Local Postgres + pgvector setup recipe.                       |
| [`tests/load/README.md`](../tests/load/README.md) | How to run the load-test scripts.                            |
| [`tests/perf/README.md`](../tests/perf/README.md) | How to run the performance regression suite.                |
| [`docs/ENGINEERING_PLAYBOOK.md`](./ENGINEERING_PLAYBOOK.md) | Day-to-day engineering manual: how to add parsers, providers, stages, etc. |
| [`docs/operations/README.md`](./operations/README.md) | On-call runbook index.                                  |

Documents planned in later phases of the roadmap — link targets above are
intentionally placeholder. They will land alongside their respective
reference, guide, architecture, contributing, and repository-map phases.

---

## Feedback

Documentation issues and pull requests are welcome. Follow the conventions
in [AGENTS.md](../AGENTS.md) when contributing. *(A dedicated contribution
guide is planned in a future phase.)*