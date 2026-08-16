# Project Root Configuration

SemRAG is portable: no project paths are baked into the backend. Every
project resolves its roots through a single mechanism
(`semrag.core.project_paths.ProjectPaths`), configured either by
environment variables, by the working directory the MCP client runs from,
or by paths registered in the database at index time.

---

## The three area roots

A project may be split across **up to three area roots** that need not
share a parent directory:

| Area | Env var            | Contents                    | Writable? |
|------|--------------------|-----------------------------|-----------|
| rtl  | `SEMRAG_RTL_ROOT`  | RTL / design source         | yes       |
| tb   | `SEMRAG_TB_ROOT`   | UVM / testbench source      | yes       |
| docs | `SEMRAG_DOCS_ROOT` | specifications, docs, logs  | **read-only** |

Every stored file path is **area-prefixed** and portable:
`rtl/i2c_master.sv`, `tb/uvm/i2c_test.sv`, `docs/spec.md`. The prefix tells
the resolver which area root owns the file, so the same relative path is
valid no matter where the roots live on disk.

Optional extras:

- `SEMRAG_PROJECT_ID` — explicit project id for the configured roots
  (otherwise derived from the deepest common ancestor of the roots).
- `root_path` — the legacy single-root display/validation working
  directory. Still supported for backward compatibility; area roots take
  precedence for resolution when present.

All area roots must be absolute (relative values are rejected at settings
load), and are canonicalized (symlinks resolved) before use.

---

## How the active project is resolved

`GET /v1/projects/resolve?cwd=<dir>` (used by the MCP layer) resolves a
working directory to a project in this order:

1. **Explicit configuration** — if any of `SEMRAG_RTL_ROOT` /
   `SEMRAG_TB_ROOT` / `SEMRAG_DOCS_ROOT` are set, the project is the
   configured one; the active *area* is whichever configured root contains
   `cwd` (or `null` when `cwd` is outside all of them).
2. **Safe layout discovery** — with no env roots configured, the nearest
   ancestor of `cwd` containing both `rtl/` and `tb/` subdirectories is
   treated as a legacy single-root project.
3. **Registered projects** — any previously-indexed project whose stored
   roots contain `cwd`.

If nothing matches, the endpoint returns **404** — SemRAG never invents a
project for an unknown directory.

Explicit selection always wins: MCP tools take a `project_id` argument and
use it directly; a blank/whitespace `project_id` means "resolve it for me".

---

## MCP context

The MCP server (`semrag_mcp_server.py`) reads `SEMRAG_CWD` (defaults to
the process cwd) to know which directory to resolve. The helper script
`scripts/opencode_mcp_semrag.sh` exports it from the invoking shell:

```sh
SEMRAG_CWD="${SEMRAG_CWD:-$(pwd)}" exec python semrag_mcp_server.py
```

OpenCode launches the MCP server with this script, so tools operate on
whatever project the current working directory belongs to.

---

## Registration and indexing

`POST /v1/projects/{project_id}/index` accepts:

- `root_path` (legacy single root), **and/or**
- `rtl_root`, `tb_root`, `docs_root`

At least one root is required (400 otherwise). The area roots become
authoritative for path resolution; `root_path`, when present, remains the
display/validation working root. The job's roots are persisted to the
`projects` / `indexing_jobs` rows and read back for subsequent scans.

---

## Validation targets

Validation config entries resolve through the same `ProjectPaths`:

- explicit area prefixes: `rtl:...`, `tb:...`, `docs:...`
- bare first-segment mapping: `rtl/...`, `tb/...`, `docs/...`
- anything else falls back to the legacy `root_path` / working root
  (pre-refactor single-root behaviour, unchanged)

---

## Write safety

The coding agent and the write/patch API endpoints only write under the
writable area roots (`rtl`, `tb`) or the legacy single root. The `docs`
root is read-only. Paths are canonicalized before use and rejected with a
400 when they are absolute, contain `..` traversal, NUL bytes, symlink
escapes, or land outside every configured root.

---

## Database schema

- `projects` and `indexing_jobs` gained nullable `rtl_root` / `tb_root` /
  `docs_root` columns (migration
  `alembic/versions/b7c2d3e4f5a6_add_area_roots_to_projects.py`).
- `chunks` gained `source_type` (rtl/tb/docs) and `source_root` columns,
  stamped at chunking time for provenance.
- A legacy single-root save does **not** clobber an existing project's
  area roots (the upsert updates a column only when a non-`None` value is
  provided).