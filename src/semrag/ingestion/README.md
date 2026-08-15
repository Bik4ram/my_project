# `src/semrag/ingestion` — Indexing Subsystem

> The one-way indexing pipeline. Turns a filesystem into a structured
> graph of files, modules, and packages, then into retrieval-sized
> text chunks with embeddings. Does not know anything about the query
> path, the API surface, or jobs — those consume what this produces.
>
> For the package-level mental model see
> [`src/semrag/README.md`](../README.md). For the end-to-end indexing
> lifecycle and the role of each stage in `services.IngestionService`,
> see [docs/developer_guide.md § Indexing pipeline](../../../docs/developer_guide.md).
> For "where exactly does X live", see
> [docs/CODEBASE_GUIDE.md](../../../docs/CODEBASE_GUIDE.md).

Intended audience: **Developers** and **Maintainers** working on the
indexing path.

---

## 1. Purpose

This package owns everything that converts a project directory on disk
into the three things the rest of the system reads from `storage/`:

1. A **scan manifest** of every file, with content hashes, git
   provenance, and coarse classification.
2. An **IR graph** of `IRNode`s and `IREdge`s — modules, interfaces,
   packages, instantiations, includes, package imports, Python imports,
   TCL sources.
3. A list of **embedded `VectorChunk`s** ready for `PgvectorStore` to
   upsert.

The pipeline is one-way by design: data flows `scanner → parsers →
ir → chunking → embedding → (storage)`. Nothing in this package imports
from `retrieval`, `reasoning`, or `api` — with one deliberate,
documented exception: `chunking/pipeline.py` imports the pure
`classify_document` function from `retrieval/source_authority.py` to
derive the chunk `category` metadata (RI-06). That is a shared
classification *primitive* reused at ingestion time so the stored
metadata matches the query-time classifier exactly — it is not a
dependency on the retrieval subsystem's behavior, and the dataflow
itself stays one-way. Reverse dependency beyond this single pure
function would be a boundary break, not a refactor.

The pipeline is also **per-file-aware but project-bounded** —
incremental re-indexing is real, but every algorithm that needs
project-wide state (e.g. symbol tables) gets it from a `SymbolTables`
instance built in pass 1, not from random file order during pass 2.

---

## 2. Package structure

```
ingestion/
├── __init__.py
├── streaming.py                   Windowed chunk iterator for batched downstream
│                                   stages.
│
├── scanner/                       Filesystem walk, hashing, classification,
│   ├── orchestrator.py            change detection, manifest storage.
│   ├── walker.py
│   ├── classification.py
│   ├── hashing.py
│   ├── git_metadata.py
│   ├── dependency_tracker.py
│   ├── change_detector.py
│   ├── manifest_store.py
│   ├── models.py
│   ├── interfaces.py
│   └── __init__.py
│
├── parsers/                       Language-specific parsers, all returning
│   ├── pipeline.py                the shared ParseResult envelope.
│   ├── models.py
│   ├── interfaces.py
│   ├── systemverilog/             pyslang syntax-tree parser
│   ├── vhdl/                      regex-structural
│   ├── python_tcl/                stdlib ast (Python) + regex-structural
│   │                              (TCL, Shell)
│   ├── documents/                 Markdown, PDF (pdfplumber), Excel
│   │                              (openpyxl)
│   ├── structured_data/           JSON / YAML / CSV / XML shape extraction
│   └── logs/                      Regression log PASS/FAIL/severity extraction
│
├── ir/                            Two-pass IR builder + symbol tables.
│   ├── builder.py
│   ├── models.py
│   ├── symbol_tables.py
│   ├── id_scheme.py
│   ├── pipeline.py                 scan_parse_and_build_ir convenience wiring
│   └── __init__.py
│
├── chunking/                      Per-language chunkers, registry, generic
│   ├── pipeline.py                sliding-window fallback, batched pipeline.
│   ├── interfaces.py
│   ├── models.py
│   ├── id_scheme.py
│   ├── systemverilog_chunker.py   structure-aware
│   ├── python_chunker.py          AST-aware
│   ├── markdown_chunker.py        heading-aware
│   ├── log_chunker.py             event-aware
│   └── generic_window_chunker.py  fallback
│
└── embedding/                     Provider + cache Protocols, three
    ├── factory.py                 provider implementations, three cache
    ├── pipeline.py                backends, batched cache-aware pipeline.
    ├── interfaces.py
    ├── ollama_provider.py
    ├── openai_compatible_provider.py
    ├── tfidf_provider.py
    ├── cache.py                   in-memory + JSON-file
    └── redis_cache.py
```

Five subpackages, one shared `Chunk` model, one streaming helper at the
top level. The shape of each subpackage is fixed: an
`interfaces.py` defining the `Protocol`, a `pipeline.py` that wires
the default registry, and the language/strategy implementations
themselves.

---

## 3. Module responsibilities

### `ingestion/scanner/` — filesystem reality

| Module                  | Responsibility |
| ----------------------- | -------------- |
| `walker.py`             | Lazy `os.walk` with in-place `dirnames` pruning (the `ALWAYS_EXCLUDED_DIRS` set covers `.git`, `Waveforms/`, `.vscode`, `__pycache__`, etc.), `.gitignore` honoring via `pathspec`, extension-based binary check, and an 8 KB content sample for downstream dependency-ref scanning. Reads no more than `MAX_CONTENT_SCAN_BYTES` (5 MB) for that sample; files larger than that are treated as opaque. Returns a generator, not a list, so the orchestrator can stream over million-file repos without materializing them. |
| `classification.py`     | Maps a relative path to a `FileCategory`. Extension lookup table for the common cases plus a small number of path-hint regexes (`/uvm/`, `/tb/`, `/testbench/`, `/assertions?/`, `/coverage/`, `/lint/`) that take priority over extension for ambiguous cases (a `.log` under `Coverage/` is a coverage report, not a plain log). Sniffs the first 8 KB of text files for NUL bytes to upgrade `is_binary`. |
| `hashing.py`            | Streaming content hash, blake3 with sha256 fallback. `hash_file` reads in 4 MB chunks; `hash_bytes` exists for in-memory content (e.g. content pulled from a git blob without writing it to disk). Algorithm name is exposed via `algorithm_name()` and recorded on every `FileRecord` so a manifest imported from a clone with different internals is still verifiable. |
| `git_metadata.py`       | Exactly **two** git invocations per scan, regardless of repo size: `git ls-tree -r -z HEAD` (one bulk call for every tracked file's blob SHA) and `git log --name-status --pretty=format:...` for commit attribution in the `(since_commit, HEAD]` range. `build_blob_info_map` combines both into the `GitBlobInfo` map the orchestrator attaches to each record. The "never shell out per file" rule is the single biggest perf invariant in this subsystem. |
| `dependency_tracker.py`  | Regex-based extraction of *raw* dependency strings from the in-memory 8 KB text sample. Captures `` `include "foo.svh" `` (SV), `import pkg::*;` (SV package), `import foo.bar` / `from foo.bar import` (Python), `source foo.tcl` (TCL), `-f filelist` and bare SV path lines (filelists). Deliberately over- and under-matches; this is a worklist for the IR builder, not a system of record. |
| `change_detector.py`    | Three-way diff of `previous.files` vs `current.files` keyed by `relative_path`. Fast path: when both sides have git blob info and the blob SHA matches, the file is unchanged without re-hashing; falls back to `content_hash` comparison for untracked files or non-git repos. Returns a `ChangeSet` with `added`, `modified`, `deleted_paths`, `unchanged_count`. |
| `manifest_store.py`     | Two `ManifestStore` implementations: `InMemoryManifestStore` (process-lifetime) and `JSONFileManifestStore` (one JSON file per project under `store_dir`). Neither is the production store — `services.IngestionService` injects the Postgres-backed implementation from `storage/postgres`. The in-package implementations exist so the scanner is independently runnable and testable. |
| `models.py`             | Pydantic models: `FileCategory` (the enum the rest of the system dispatches on), `GitBlobInfo`, `FileRecord`, `GitRepositoryMetadata`, `ScanManifest`, `ChangeSet`. All serialized via pydantic so they round-trip through the JSONB manifest column and the Celery task boundary without a DTO layer. |
| `interfaces.py`         | `ManifestStore` and `ScannerInterface` Protocols. Everything outside the scanner depends on these, not on `Scanner` or the model classes. |
| `orchestrator.py`       | The `Scanner` class — the only stateful piece. Loads the previous manifest, calls `git_metadata.get_repository_metadata` and `build_blob_info_map`, walks the repo, partitions files into "reuse previous record" (git blob SHA unchanged) vs "must hash", runs a `ThreadPoolExecutor` for hashing, builds the new `ScanManifest`, computes the `ChangeSet`, saves the manifest. Default 8 hash workers; tunable because NFS-backed home directories saturate at high concurrency. |

### `ingestion/parsers/` — file → structured payload

| Module / subpackage   | Responsibility |
| --------------------- | -------------- |
| `interfaces.py`       | `Parser` Protocol (`name`, `parse(path, FileRecord) -> ParseResult`) and `ParserRegistry`. The registry dispatches by `FileCategory`, not by extension, and supports an **ordered fallback chain** per category — the first parser whose `ParseResult.success` is `True` wins; the last failure is returned (not `None`) so the caller has diagnostics. |
| `models.py`           | `DiagnosticSeverity`, `ParseDiagnostic`, `ParseResult`. The `payload` is `dict[str, Any]` so a new language parser doesn't force a change to this envelope. |
| `pipeline.py`         | `build_default_registry()` — the single place that wires concrete parsers to `FileCategory`. Coverage: SystemVerilog/UVM/assertions (pyslang), VHDL (regex-structural), Python (stdlib `ast`), TCL and Shell (regex-structural), Markdown (regex-structural headings), JSON/YAML/CSV/XML (shape extraction), PDF (`pdfplumber`, text + tables), Excel (`openpyxl`, sheet shape + samples), regression logs (heuristic PASS/FAIL/severity counts). Categories deliberately not wired here (with the same reasoning as the code): `COVERAGE_REPORT`, `TIMING_REPORT`, `LINT_REPORT`, `WAVEFORM`, `BUILD_CONFIG`. |
| `systemverilog/`      | Per-file `SyntaxTree.fromFile` parsing via pyslang. Produces `modules` (with `is_interface`, ports, parameters, instantiations, assertions, procedural blocks, span), `packages`, and per-file diagnostics. The parser docstring explains why syntax-tree parsing, not full `Compilation` elaboration, is the right call for an indexer. |
| `vhdl/`               | Regex-structural — VHDL doesn't have a freely-available grammar parser comparable to pyslang. |
| `python_tcl/`         | `python_parser.py` uses stdlib `ast` to extract modules, classes, functions, and import statements with line numbers. `tcl_parser.py` and `shell_parser.py` are regex-structural. |
| `documents/`          | `markdown_parser.py` extracts heading hierarchy. `pdf_parser.py` uses `pdfplumber` for text + tables. `excel_parser.py` uses `openpyxl` for sheet shape and small samples. |
| `structured_data/`    | JSON / YAML / CSV / XML shape extraction — enough structure for retrieval, not a full DOM walk. |
| `logs/`               | `regression_log_parser.py` extracts PASS/FAIL counts and severity breakdown. |

### `ingestion/ir/` — project graph

| Module            | Responsibility |
| ----------------- | -------------- |
| `models.py`       | `IRNodeKind` (SV_FILE, SV_MODULE, SV_INTERFACE, SV_PACKAGE, PY_MODULE, TCL_SCRIPT), `IREdgeKind` (INSTANTIATES, INCLUDES, IMPORTS_PACKAGE, PY_IMPORTS, TCL_SOURCES), `IRNode`, `IREdge`, `IRGraph` with `nodes_of_kind` / `edges_from` / `unresolved_edges` accessors. This is a **structural linking IR**, not a semantic one — assertion-level "covers register X" is out of scope by design. |
| `id_scheme.py`    | `sv_file_id`, `sv_module_id`, `sv_package_id`, `py_module_id`, `tcl_script_id`, `edge_id`. Every ID is a pure function of `(project_id, relative_path, name/kind)` so re-running on an unchanged file produces byte-identical node IDs. Edge IDs additionally hash in `target_name` so a changing set of refs produces a different edge set, not a collision. |
| `symbol_tables.py`| Project-wide lookup maps keyed by basename (SV files, TCL scripts), module name, package name, and dotted-path suffix (Python). Every index maps to a **list** of candidate node ids, not a single one — name collisions (`fifo` in two subsystems) are surfaced as `ambiguous=True`, not silently resolved. Python `py_module_by_dotted_suffix` strips a single leading `src/` part to match the `src/` layout convention used by this repo. |
| `builder.py`      | `IRBuilder().build(...)` — the two-pass algorithm. **Pass 1** creates every node from the parser payload and populates the symbol tables. **Pass 2** resolves every edge against the now-complete symbol tables. A single-pass approach would make resolution order-dependent on file processing order (a forward-reference bug). The split between SV file-level `\`include` (resolved by basename against `sv_file_by_basename`) and SV package `import` (resolved by bare identifier against `sv_package_by_name`) is possible exactly because the two regexes in `dependency_tracker.py` produce strings with and without file extensions respectively. |
| `pipeline.py`     | `scan_parse_and_build_ir(...)` — the convenience wiring `services.IngestionService` could call instead of staging the three calls itself. Deliberately re-parses every file on every call; incremental *parsing* is a worker-layer concern, not the builder's. |

### `ingestion/chunking/` — retrieval-sized text units

| Module                       | Responsibility |
| ---------------------------- | -------------- |
| `interfaces.py`              | `Chunker` Protocol and `ChunkerRegistry`. The registry runs the primary chunker; if it returns an empty list **and** a fallback is registered, the fallback is run — this is what keeps files like a UVM scoreboard subclass (which parses cleanly but has zero IR module nodes) in retrieval instead of silently vanishing. The registry also stamps each returned chunk's `chunker_name` from the chunker that actually produced it (including the fallback path), so the metadata can't lie about provenance. |
| `models.py`                  | `Chunk` — `chunk_id`, `project_id`, `relative_path`, `chunk_type`, `source_node_id` (the IR node this chunk came from, when one exists), `text`, `line_start`, `line_end`. Plus RI-06 retrieval metadata: `language`, `category`, `parser_name`, `chunker_name`, `symbol_name`, and the derived `is_generated` property. Same content-derived-ID philosophy as every other model in this system. |
| `id_scheme.py`               | `chunk_id(project, path, kind, disambiguator)` and `extract_lines(text, start, end)` — 1-indexed, inclusive, matching the line convention every parser in this package uses. |
| `systemverilog_chunker.py`   | Structure-aware. **Small modules** (≤ 150 lines): one whole-module chunk. **Large modules**: one signature chunk (declaration through the line before the first procedural/assertion), one chunk per assertion with `ASSERTION_CONTEXT_LINES = 2` of surrounding context, and sliding-window body chunks (`BODY_WINDOW_LINES = 60`, overlap `10`). Packages get one whole-package chunk each (SV packages in practice are small). |
| `python_chunker.py`          | AST-aware. One module-overview chunk (everything before the first function/class, non-empty only) plus per-function and per-class chunks. Classes split the same way as SV modules: small (≤ 100 lines) → one chunk; large → signature chunk + one chunk per method. |
| `markdown_chunker.py`        | Heading-aware. Section boundaries are the next heading of *any* level (not just same-or-shallower), which avoids the H1-contains-everything duplication bug that the same-or-shallower rule produced. Hierarchy context is a breadcrumb line prefixed to the chunk, not duplicated parent prose. Any preamble before the first heading becomes its own chunk. |
| `log_chunker.py`             | Event-aware. Splits on UVM severity / generic severity markers, keeps multi-line events (message + packet table + PASS/FAIL verdict, or an `ERROR` line + its continuation) intact in one chunk, groups consecutive single-line events (repetitive `[STATE_COV_ITEM]`-style noise) into bounded windows, and reuses the 60/10 window convention for oversized events and marker-less logs. Emits `chunk_type="log"` with deterministic sequential chunk IDs. |
| `generic_window_chunker.py`  | Fixed-line sliding window (default 60 lines, 10-line overlap). The fallback. Deliberately ignores parsed structure — a generic chunker cannot know what structural boundaries matter for an arbitrary file type. |
| `pipeline.py`                | `build_default_registry()` and `chunk_project(...)`. Skips binary files entirely (their bytes contain NULs that crash psycopg2 binding). `max_workers > 1` runs reads and chunking across a thread pool; default is 1 (sequential) for behavior parity with the original implementation. |

### `ingestion/embedding/` — text → vector

| Module                            | Responsibility |
| --------------------------------- | -------------- |
| `interfaces.py`                   | `EmbeddingProvider` Protocol (`dimension`, `model_name`, `embed_texts`). `EmbeddingCache` Protocol keyed by **text hash**, not chunk_id (two chunks with identical text share one cache entry; a chunk whose text is byte-identical after a re-chunk hits the cache even if its `chunk_id` changed). |
| `ollama_provider.py`              | One request per text against Ollama's `/api/embeddings` (or `/api/embed`), concurrent up to 8 in flight. Exponential backoff with bounded retries. Surfaces dimension mismatches as a hard error so a misconfigured `EMBEDDING_DIMENSION` doesn't silently corrupt the vector column. |
| `openai_compatible_provider.py`   | Standard `POST /v1/embeddings` with `Authorization: Bearer` if an API key is set. Sorts the response by `index` rather than trusting positional order — the API contract only guarantees order via the field. |
| `tfidf_provider.py`               | TF-IDF + truncated SVD (`max_features=50_000`, configurable target dimension, defaults to 768). **Corpus-fit**, not pretrained — `fit()` must be called on a representative batch of chunk text before `embed_texts()` is usable, and the resulting vector space is specific to that fit. A small corpus gets a smaller actual SVD dimension, zero-padded to the target — this is documented via `_actual_dimension`, not hidden. `save` / `load` use `joblib` so the fit survives a process restart. The real limitation is that re-fitting invalidates every existing chunk's vector — see "Add when" for why a pretrained neural model is the production path. |
| `cache.py`                        | `InMemoryEmbeddingCache` (tests) and `JSONFileEmbeddingCache` (one file per `model_name`; the file path is the namespace). `text_hash` reuses `scanner.hashing.hash_bytes` so there's exactly one hashing implementation. |
| `redis_cache.py`                  | `RedisEmbeddingCache`. `MGET` for batch reads, pipelined `SET` for batch writes. Namespaced under `semrag:embedding:{model_name}:{text_hash}` so different models never share keys. JSON-encoded values for portability. |
| `factory.py`                      | `build_embedding_provider(settings)` and `build_embedding_cache(settings)`. Provider selection is by `settings.embedding_provider` (`ollama` / `openai_compatible` / `tfidf`); unknown names raise `EmbeddingConfigurationError`. Cache selection is exact: `redis` requires `SEMRAG_REDIS_URL` to be set and raises rather than silently downgrading. |
| `pipeline.py`                     | `embed_chunks(chunks, provider, cache, batch_size=64)`. Hash every chunk's text, batched `get_many` against the cache, batch-embed only the misses, populate the cache with the new results, and call `cache.flush()` once at the end. Returns `list[VectorChunk]` (from `storage.vector.models`) — the persistence layer's input. |

### `ingestion/streaming.py`

`stream_chunks_in_windows(chunks, window_size)` — a tiny helper that
yields fixed-size windows of chunks for downstream stages that want to
start before the entire chunk list is materialized. The docstring is
explicit: the streaming boundary is `chunk → embed → persist` (the IR
builder must finish before chunking, persistence must finish after
embedding); this helper exists for the chunk-embed-persist span only.

---

## 4. End-to-end ingestion pipeline

The full index path lives in
[`services.IngestionService.index()`](../services/services.py). It is
the only place that calls into every stage in order; this package
deliberately does not provide a one-function "do everything" entry
point outside of `ir.pipeline.scan_parse_and_build_ir` (which exists
for testing the IR builder without Celery). The `services.py` driver
is the real orchestrator.

```
                ┌──────────────────────────────────────────────────────┐
                │                 IngestionService.index()             │
                └─────┬────────┬──────────┬──────────┬──────────┬──────┘
                      ▼        ▼          ▼          ▼          ▼
                  scan       parse     build IR   chunk      embed
                      │        │          │          │          │
                      ▼        ▼          ▼          ▼          ▼
              ┌──────────┐ ┌──────┐ ┌──────────┐ ┌──────┐ ┌────────┐
              │ Scanner  │ │Parser│ │ IRBuilder│ │Chunker│ │Embed-  │
              │          │ │Reg.  │ │          │ │ Reg.  │ │der     │
              └─────┬────┘ └──┬───┘ └─────┬────┘ └──┬───┘ └───┬────┘
                    │         │           │          │         │
                    ▼         ▼           ▼          ▼         ▼
               ScanManifest ParseResults  IRGraph  [Chunk]  [VectorChunk]
                    │         │           │          │         │
                    └────┬────┴─────┬─────┴──────────┴────┬────┘
                         ▼          ▼                     ▼
                  PostgresManifestStore  IRGraphRepository   PgvectorStore
```

Stage contracts, as the data crosses them:

| Stage  | Input                                       | Output                                    |
| ------ | ------------------------------------------- | ----------------------------------------- |
| scan   | `(project_id, root_path)`, `ManifestStore`  | `ScanManifest`, `ChangeSet`               |
| parse  | `ScanManifest.files`, `ParserRegistry`      | `dict[relative_path, ParseResult]`        |
| ir     | `ScanManifest.files`, `dict[…, ParseResult]`| `IRGraph`                                 |
| chunk  | `file_records`, `parse_results`, `ir_graph` | `list[Chunk]`                             |
| embed  | `list[Chunk]`, `EmbeddingProvider`, cache   | `list[VectorChunk]`                       |
| persist| IR graph + `VectorChunk`s                   | rows in Postgres / pgvector tables        |

Every stage is independently importable and unit-testable; the
`IngestionService` driver is the only place that composes them, which
is what makes incremental re-indexing (only re-running the stages that
the `ChangeSet` says are necessary) a configuration concern of the
worker, not a refactor of this package.

---

## 5. Scanner and manifest handling

The scanner is a stateful subsystem: each scan is a function of the
previous scan's manifest. The cycle is:

1. `Scanner.scan(project_id, root_path)` resolves the root, raises
   `NotADirectoryError` if the path doesn't exist.
2. `manifest_store.get_latest_manifest(project_id)` is loaded (or
   `None` for a first-ever scan).
3. `git_metadata.get_repository_metadata(root)` returns the
   repository-level state. For a non-git directory, `is_git_repo=False`
   and everything downstream falls back to blake3-only change detection.
4. `git_metadata.build_blob_info_map(root, previous_head_sha)` runs
   the two bulk git calls and returns `{relative_path: GitBlobInfo}`.
5. `walk_repository(root)` yields `WalkedFile`s lazily. Files inside
   `ALWAYS_EXCLUDED_DIRS` are never descended into; `.gitignore`
   patterns are honored; permissions errors mid-walk are silently
   skipped, not raised.
6. For each file, the orchestrator compares the previous `FileRecord`'s
   `git_info.blob_sha` to the new `git_info.blob_sha`. If both are
   present and equal, the previous record is reused (with
   updated `size_bytes`, `mtime`, and `git_info`). Otherwise, the file
   is queued for hashing.
7. Hashing happens in a `ThreadPoolExecutor` with `max_workers` tunable
   (default 8). Each worker calls `hashing.hash_file(path)` and
   extracts `raw_dependency_refs` from the same 8 KB text sample the
   walker took.
8. The new `ScanManifest` is built, the `ChangeSet` is computed, the
   manifest is persisted via `manifest_store.save_manifest`. Both are
   returned to the caller.

`ChangeSet.has_changes` and `ChangeSet.total_changed` exist as
diagnostics; the worker (in `jobs/`) is the one that decides which
stages to re-run.

---

## 6. Parsing subsystem

Per-file parsing is **independent and per-category**:

- Dispatch goes through `ParserRegistry.parse_file(path, file_record)`.
  If no parser is registered for the file's category, the call returns
  `None` — an unparsed file is an expected case (waveform dumps, git
  internals, anything not yet wired), not an error.
- If a parser is registered, the first one whose `ParseResult.success`
  is `True` wins. On a full-chain failure the last `ParseResult` is
  returned so the caller has diagnostics to log.
- The payload is a `dict[str, Any]`, not a union type, so adding a new
  language parser never requires a change to `ParseResult` itself.

`build_default_registry()` in `parsers/pipeline.py` is the single
place where category-to-parser wiring lives. New parsers register
here, not in the orchestrator, not in the chunker, not in the IR
builder.

---

## 7. IR generation

`IRBuilder.build(project_id, file_records, parse_results)` returns an
`IRGraph`. The algorithm is two passes on purpose:

- **Pass 1** iterates every `(relative_path, FileRecord)` and
  `ParseResult` pair. For the SV categories, the file gets an
  `SV_FILE` node and one `SV_MODULE` / `SV_INTERFACE` / `SV_PACKAGE`
  node per declaration in the parser payload, each with its
  `parent_file_id` set to the file's id. For Python, one `PY_MODULE`
  node per `.py` file. For TCL, one `TCL_SCRIPT` node per `.tcl`
  file. Every node is also indexed in `SymbolTables` for pass 2.
- **Pass 2** iterates the same set. For SV, the file's
  `raw_dependency_refs` is split into `\`include` (has an SV file
  extension, resolved against `sv_file_by_basename`) and `import` (bare
  identifier, resolved against `sv_package_by_name`). Module
  instantiations come from `payload["modules"][…]["instantiations"]`
  and resolve against `sv_module_by_name`. Python imports resolve
  against `py_module_by_dotted_suffix`; TCL `source` references
  resolve against `tcl_by_basename`.

Node and edge IDs are content-derived. The same input always produces
the same IDs, which is what makes `IRGraphRepository.save(ir_graph)`
upsert rather than duplicate-insert on every re-index.

The builder does **not** do semantic resolution — it doesn't know that
`scl` in an assertion expression in module A is the same `scl` declared
as a port in module B. That is a separate, harder analysis pass and is
out of scope for this package.

---

## 8. Chunking subsystem

The chunker selection model is the same as the parser: one Protocol,
one registry, one fallback. Two differences:

- Dispatch is keyed on `FileCategory`, not `IRNodeKind`, because
  chunking's input is broader than IR nodes — Markdown, VHDL, JSON,
  etc. have no IR node at all, but a chunker must still run on them.
- The registry always runs a **fallback** when the primary chunker
  produces an empty list. This is what guarantees that a SV file
  containing only UVM classes (parses cleanly, but has no module-level
  IR nodes, so the SV chunker returns `[]`) is still covered by the
  generic sliding-window chunker rather than vanishing from
  retrieval.

`chunk_project(...)` is the entry point used by the service driver:

- Binary files are skipped before the chunker sees them — PDFs,
  Excels, and waveforms can't be read as UTF-8 (the NULs crash
  psycopg2 binding) and no current chunker reads
  `parse_result.payload` for them.
- Files that have vanished or become unreadable between the scan and
  the chunking pass are silently skipped, not raised.
- `max_workers=1` keeps the original sequential behavior. Higher
  values run reads + chunking across a thread pool, appropriate when
  the cost is dominated by disk I/O (large files, network mounts).
- **RI-06 metadata enrichment** — every chunk is filled in once,
  centrally, by `_enrich_chunk` using canonical single sources:
  `language` from `scanner/classification.language_for_extension`,
  `category` from Source Authority's `classify_document` (the existing
  query-time classifier, reused so stored metadata can't drift from
  it), and `parser_name` from the file's `ParseResult`.
  `chunker_name` is stamped by the registry, `symbol_name` by the
  chunkers. Nothing downstream recalculates these.

The bespoke chunkers' strategies:

- **SystemVerilog** uses `node.attributes["span"]` and
  `node.attributes["assertions"][…]["span"]` to cut at module
  boundaries. The 150-line threshold separates "small enough to be
  one chunk" from "large enough to need signature + assertions +
  body windows" — see the chunker docstring for the rationale.
- **Python** uses the AST-derived `functions` / `classes` lists in
  `node.attributes`. The 100-line class threshold triggers
  signature + per-method chunking.
- **Markdown** uses `parse_result.payload["headings"]` and the
  "next heading of any level" rule. A single file with no headings
  gets one whole-file chunk rather than nothing.

---

## 9. Embedding subsystem

`embed_chunks(chunks, provider, cache, batch_size=64)` is the entry
point. The algorithm:

1. Hash every chunk's text with `text_hash` (which is
   `scanner.hashing.hash_bytes` over UTF-8 bytes — one hashing
   implementation in the codebase).
2. Deduplicate to **unique hashes**, run a single `cache.get_many`,
   and partition into `resolved` and `unresolved_hashes`.
3. For each batch of `unresolved_hashes` (size `batch_size`),
   `provider.embed_texts(batch_texts)` is called, the new
   `text_hash → vector` entries are merged into `resolved`, and
   `cache.set_many` is called.
4. After all batches, `cache.flush()` runs once (this matters for
   `JSONFileEmbeddingCache`, which batches its disk writes).

The cache key is the text hash, not the chunk id. Two different
chunks with identical text (e.g. a license header chunked separately
per file) share one cache entry; a chunk whose text is byte-identical
after a re-chunk hits the cache even if upstream chunking logic
changed its `chunk_id`.

Three cache backends:

- **In-memory** — for tests, scripts, and one-shot processes.
- **JSON file** — one file per `model_name`; safe for a single
  workstation, not for concurrent writers.
- **Redis** — namespaced under `semrag:embedding:{model_name}:{hash}`;
  uses `MGET` for batch reads and pipelined `SET` for batch writes.

Three providers:

- **Ollama** (`/api/embeddings`) — native endpoint, one request per
  text, up to 8 concurrent.
- **OpenAI-compatible** (`/v1/embeddings`) — standard HTTP contract
  that Ollama, vLLM, and llama.cpp's compat servers all implement;
  sorts response by `index` to avoid trusting positional order.
- **TF-IDF + SVD** — corpus-fit, not pretrained. The TF-IDF provider
  is documented as a development / testing baseline that requires
  `fit()` on a representative batch of chunk text before
  `embed_texts()` works. **Re-fitting invalidates the entire vector
  index** (embeddings from two different fits are not in the same
  space), which is the single biggest reason this is not the
  production default.

Provider and cache selection is `Settings`-driven via
`build_embedding_provider()` and `build_embedding_cache()` in
`factory.py`.

---

## 10. Persistence handoff

This package **does not** talk to Postgres directly. The handoff is
to `storage/`:

- `ScanManifest` and `ChangeSet` → `PostgresManifestStore` in
  `storage/postgres/`. Injected as the `ManifestStore` argument to
  `Scanner`.
- `IRGraph` → `IRGraphRepository.save(ir_graph)` in
  `storage/postgres/`. Called by `services.IngestionService`, not by
  this package.
- `list[VectorChunk]` (returned by `embed_chunks`) →
  `PgvectorStore.upsert_chunks(vector_chunks)` in
  `storage/vector/`. Called by `services.IngestionService`, not by
  this package.

The boundary is what lets the JSON-file / in-memory implementations of
`ManifestStore` exist in this package without conflicting with the
production Postgres implementation in `storage/`. The same Protocol,
two implementations, the caller picks.

The end-to-end index run (`scan → parse → IR → chunk → embed →
persist`) is composed by `services.IngestionService.index()` and
labeled with `core.observability.stage()` blocks for structured
logging and metrics. See the developer guide's "Indexing pipeline"
chapter for the canonical walkthrough.

---

## 11. Extension points

| You want to                              | Edit                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Add a parser for a new file type         | New file in `ingestion/parsers/<language>/parser.py`; register in `ingestion/parsers/pipeline.py:build_default_registry`. |
| Change parser fallback chain (e.g. add a tree-sitter SV parser) | `ParserRegistry.register(category, parser)` in the pipeline file — the registry already supports an ordered chain. |
| Add a chunker for a new language          | New file in `ingestion/chunking/<chunker>.py`; register in `ingestion/chunking/pipeline.py:build_default_registry`. |
| Add an embedding provider                | New file in `ingestion/embedding/<provider>.py` implementing the `EmbeddingProvider` Protocol; add a case in `ingestion/embedding/factory.py:build_embedding_provider`; add the new settings to `core/settings.py`. |
| Switch the embedding cache backend       | `SEMRAG_EMBEDDING_CACHE_BACKEND` (already supports `memory` / `jsonfile` / `redis`). Add a new case in `build_embedding_cache` to wire a fourth. |
| Add a new `FileCategory`                 | Extend the enum in `ingestion/scanner/models.py`; add a classification rule in `classification.py`; register parsers / chunkers in their respective `pipeline.py`. |
| Replace the manifest store with a custom backend | Implement the `ManifestStore` Protocol from `ingestion/scanner/interfaces.py`; inject it into the `Scanner` constructor. The Postgres-backed implementation lives in `storage/postgres/`. |
| Replace the change detector              | `Scanner` calls `change_detector.detect_changes(previous, current)` as a free function. Drop in a different implementation with the same signature. |
| Change hashing algorithm                 | `ingestion/scanner/hashing.py` already supports blake3 with sha256 fallback. The algorithm name is recorded on every `FileRecord`, so swapping it is a one-line change but requires a full re-index (manifests from before the change are not directly comparable). |
| Wire a parser for a category that's currently routed nowhere (`COVERAGE_REPORT`, `WAVEFORM`, etc.) | See "Skipped" below — these need either real sample files to validate against or a different scope (metadata only, not full content). |
| Re-fit the TF-IDF provider                | `TfidfEmbeddingProvider.fit(corpus_texts)`; persist with `save(path)`; load with `TfidfEmbeddingProvider.load(path)`. Be aware: re-fitting invalidates every existing chunk's vector. |

---

## 12. Common debugging paths

| Symptom                                       | Look at                                                                       |
| --------------------------------------------- | ----------------------------------------------------------------------------- |
| "Indexing didn't pick up my changes"          | `ingestion/scanner/orchestrator.py` (the reuse-vs-rehash partition) + `change_detector.py` (the three-way diff). Verify with `manifest.files[relpath].git_info.blob_sha` and `content_hash`. |
| A file's not in the manifest at all           | `ingestion/scanner/walker.py` — it's probably in `ALWAYS_EXCLUDED_DIRS` (`.git`, `Waveforms/`, `__pycache__`, `.venv`, etc.) or matched by `.gitignore`. |
| A file shows up but `category` is `UNKNOWN`   | `ingestion/scanner/classification.py` — extension is not in `_EXTENSION_MAP` and no path hint matches. Add a mapping or a hint regex. |
| Parsing fails on a file type                  | `ingestion/parsers/pipeline.py` for the wiring + the specific parser's `parser.py`. A chain failure returns the last `ParseResult` (with `success=False` and diagnostics) — read the `diagnostics` list, not just the exception. |
| Chunks for a Python file look wrong           | `ingestion/chunking/python_chunker.py` — small vs large class threshold is at `SMALL_CLASS_LINE_THRESHOLD = 100`. AST-derived structure comes from the Python parser, not the chunker. |
| Chunks for a SystemVerilog file look wrong    | `ingestion/chunking/systemverilog_chunker.py` — module threshold is `SMALL_MODULE_LINE_THRESHOLD = 150`, body window is `BODY_WINDOW_LINES = 60` with overlap `10`. Structure comes from the SV parser's payload, especially `payload["modules"][…]["span"]`, `payload["modules"][…]["assertions"]`, and `payload["modules"][…]["procedural_blocks"]`. |
| Markdown chunk duplicates the whole document  | This is the historical bug fixed by the "next heading of any level" rule. Verify your version has the fix. |
| A file has zero chunks even though it parsed  | The file's category is one of the IR-less ones (Markdown, structured data, PDF, Excel, log) and the generic fallback didn't run — check that `build_default_registry()` was called (it registers the fallback). For SV files with only UVM classes, this is expected and the fallback handles it. |
| A chunk's `language`/`category`/`symbol_name` looks wrong       | Metadata is enriched once in `chunking/pipeline.py:_enrich_chunk` and stamped by the chunkers/registry. `language` comes from the scanner's extension mapping (`scanner/classification.py:language_for_extension`); `category` from Source Authority's `classify_document` (a path rule beats a chunk-type rule — see `docs/spec.md` vs `RTL/x.sv`). `symbol_name` is only set when the parser knows a module/class/function name; generic/log/markdown chunks are intentionally `None`. |
| "Embedding dimension mismatch" error          | `EMBEDDING_DIMENSION` (in `storage/vector/pgvector_models.py`) must match the provider's actual output. The provider's own `dimension` property is the source of truth for what was configured. |
| Embeddings are slow                           | `ingestion/embedding/pipeline.py:embed_chunks` — increase `batch_size`, or warm the cache by re-running on a corpus the cache already covers. |
| The embedding cache is huge / not evicting    | `InMemoryEmbeddingCache` and `JSONFileEmbeddingCache` have no eviction. `RedisEmbeddingCache` is namespaced by `model_name` so changing the model resets the namespace. |
| First-time scan is very slow on a big repo    | The git calls are `O(repo)` and unavoidable; the file hashing is I/O-bound and parallelized. `Scanner(max_hash_workers=...)` lets you tune the parallelism, but on an NFS-mounted home directory, more workers often means slower scans. |
| A wave file is being hashed as text           | `MAX_CONTENT_SCAN_BYTES` is 5 MB; files larger than that are hashed but not opened for the content sample. If a smaller wave file is sneaking through, check `is_binary_extension` in `classification.py`. |

---

## 13. Common mistakes

- **Shelling out to `git` per file.** Don't. The `git_metadata` module
  is built around the two-bulk-calls invariant; if you need a new git
  fact (e.g. per-file diff against a branch), add it as a bulk call
  there, not as a subprocess in the orchestrator.
- **Reading a file's full content during the walk.** The walker takes
  an 8 KB sample for `is_binary` detection and dependency-ref
  scanning. Reading the full file here makes a 1 GB waveform
  indistinguishable from a 1 KB `.f` filelist for memory and time
  purposes.
- **Trusting parser output without checking `success`.** A parser
  failure is a `ParseResult` with `success=False` and a populated
  `diagnostics` list, not an exception. The IR builder handles this
  correctly (skips a file with no successful result), but new code
  reading from `parse_results` should also check.
- **Trying to do semantic IR resolution in the builder.** The
  structural linking IR is what this package produces. "Does this
  assertion cover that register" needs a different analysis pass.
- **Picking `TfidfEmbeddingProvider` for production without a re-fit
  strategy.** Re-fitting invalidates the entire vector index. If you
  do, document the operational event and budget for re-embedding
  every chunk.
- **Putting files in the source root.** `ingestion/` holds exactly
  the five subpackages and `streaming.py`. Anything else belongs
  inside one of them.
- **Sorting vector responses positionally.** The OpenAI-compat
  contract guarantees order via the `index` field, not via position.
  Sort on the field.
- **Re-hashing on every scan.** The whole point of the
  blob-SHA-vs-content-hash fast path is to avoid re-reading
  unchanged files. If a file is being re-hashed when it shouldn't be,
  `previous_record.git_info.blob_sha` is probably `None` — the
  previous manifest wasn't loaded, or the file wasn't tracked in the
  previous git HEAD.
- **Writing chunk IDs as random UUIDs.** The deterministic-ID
  convention is what makes `PgvectorStore.upsert_chunks` an upsert
  rather than a duplicate insert. UUIDs break this.
- **Reading binary files into the chunker.** `chunk_project` skips
  them explicitly. If you bypass `chunk_project` (e.g. for a custom
  test), check `file_record.is_binary` first.

---

## 14. Architectural boundaries

These rules come from the project-wide
[developer guide](../../../docs/developer_guide.md#common-architectural-rules);
the ones that apply specifically to this package:

- **Ingestion does not import from `retrieval`.** Ingestion is
  upstream; the dataflow is one-way. Reaching back is a boundary
  break, not a refactor. **Sole exception (RI-06):**
  `chunking/pipeline.py` imports `classify_document` from
  `retrieval/source_authority.py` as a shared classification primitive
  for the `category` metadata field — a pure function, no retrieval
  subsystem dependency, no dataflow reversal. Any further reverse
  import would be a boundary break.
- **Ingestion does not import from `api` or `jobs`.** The `services`
  layer and the `jobs` executor are the only consumers of the
  ingestion package, and they go through the public entry points
  (`Scanner.scan`, `ParserRegistry.parse_file`, `IRBuilder.build`,
  `chunk_project`, `embed_chunks`).
- **Ingestion does not talk to the database directly.** The
  `ManifestStore` Protocol is the only persistence interface this
  package uses. The Postgres-backed implementation lives in
  `storage/`.
- **Parsers do not produce IR.** They produce `ParseResult` with a
  payload. The IR builder is the only thing that turns payloads into
  nodes and edges. This is what lets the IR builder be re-run
  incrementally over cached `ParseResult`s.
- **Chunkers do not embed.** `Chunk` is a text object; the
  `EmbeddingProvider` operates on it. Two different
  model-versioning concerns (chunking strategy, embedding model) do
  not get tangled.
- **Embedding providers do not cache.** The cache is layered on top
  of every provider uniformly in `embedding/pipeline.py`; providers
  are pure text-in / vectors-out.
- **One hashing implementation.** `scanner.hashing.hash_bytes` is the
  single primitive; `embedding.cache.text_hash` is a one-line
  wrapper. Adding a second hash function is a coordination problem
  (manifests become non-comparable), not a convenience.
- **The Protocol/registry pattern is the only extension point.**
  Adding a parser, chunker, provider, cache, or manifest store
  always means implementing a Protocol and registering in the
  corresponding `pipeline.py` factory. New code that bypasses the
  Protocol is a refactor of the existing one, not new code.

For the full architectural rules and the *why* behind each, see
[developer_guide.md § Common architectural rules](../../../docs/developer_guide.md).

---

## 15. Related documentation

- [`src/semrag/README.md`](../README.md) — package-level mental model
  and the dependency diagram.
- [`docs/developer_guide.md`](../../../docs/developer_guide.md) — the
  "Indexing pipeline" chapter has the canonical walkthrough of
  `services.IngestionService.index()` stage by stage.
- [`docs/CODEBASE_GUIDE.md`](../../../docs/CODEBASE_GUIDE.md) — file-level
  catalog of every directory in this package.
- [`docs/README.md`](../../../docs/README.md) — documentation index.
- [`src/semrag/ingestion/parsers/systemverilog/README.md`](parsers/systemverilog/README.md) —
  SystemVerilog parser specific notes and tracked gaps.

---

## Skipped (and why)

Documented gaps and intentionally-omitted behaviors, with the
reasoning that already lives in the code:

- **Tree-sitter SV fallback parser.** The
  `ParserRegistry.parse_file` mechanism already supports an ordered
  fallback chain per `FileCategory`, but only the pyslang
  syntax-tree parser is registered for the SV categories today. Files
  pyslang cannot parse at all currently fail with the last
  `ParseResult` rather than being attempted by a more permissive
  parser. See
  [`parsers/systemverilog/README.md`](parsers/systemverilog/README.md).
- **Coverage / lint / timing report parsers.** `COVERAGE_REPORT`,
  `LINT_REPORT`, `TIMING_REPORT` are recognized by the scanner
  (`FileCategory` enum) but not registered in
  `build_default_registry`. Each tool's format is bespoke (URG, UCD,
  Questa, Spyglass, Synopsys) and needs real sample files to
  validate against; this delivery does not have them. A regression
  log parser (`RegressionLogParser`) is wired for the plain-text case
  only.
- **Waveform / build-config parsers.** Waveform dumps (VCD, FSDB, WLF,
  SHM) are binary or near-binary formats whose *content* is not
  retrieval-relevant text. The right scope is metadata-only
  (signal list, time range), which is a future module's concern.
  Filelists (`.f` files) and Makefiles are a strong candidate for a
  later iteration — they directly define compile-order dependency
  edges the IR would want — but are not parsed today.
- **Structural IR for languages other than SV / Python / TCL.** Only
  the IR builder's `IRNodeKind` set is exposed. Markdown, VHDL,
  documents, structured data, and logs parse into
  `ParseResult.payload` but produce no IR nodes; the chunker falls
  through to the generic sliding window for these.
- **Semantic IR resolution.** "This assertion covers register X"
  requires signal-level analysis (is `scl` in this expression the
  same `scl` declared as a port in a different file?). The
  structural linking IR is what this package produces; semantic
  linking is a distinct, harder pass layered on top.
- **Incremental parse cache.** `ir.pipeline.scan_parse_and_build_ir`
  re-parses every file on every call. Incremental parsing
  (re-using cached `ParseResult`s for unchanged files) is a
  worker-layer concern, not a builder-layer concern. The
  `ChangeSet` from the scanner is the input; reusing cached results
  is the worker's responsibility.
- **Postgres-backed `ManifestStore` in this package.** The
  scanner depends on the `ManifestStore` Protocol only;
  `InMemoryManifestStore` and `JSONFileManifestStore` are the two
  in-package implementations. The production Postgres-backed
  implementation lives in `storage/postgres/`. This split keeps the
  scanner independently runnable.
- **`source_node_id` for non-IR-node categories.** `Chunk.source_node_id`
  is `None` for chunks from Markdown, VHDL, JSON/YAML/CSV/XML, PDF,
  Excel, logs — categories the IR builder doesn't create nodes for.
  It is populated only for SV / Python / TCL chunks.

---

## Add when

Improvements that should only be documented as part of the production
implementation if and when they happen:

- **A tree-sitter SV fallback parser** should be added when there are
  enough real-world SV files pyslang cannot parse to justify a second
  parser. Until then the
  [`parsers/systemverilog/README.md`](parsers/systemverilog/README.md)
  gap-tracking note is the right level of visibility.
- **Report parsers (coverage / lint / timing) and waveform metadata
  extraction** should be added when real sample files from a target
  EDA tool are available, and only when retrieval over them is
  actually wanted — the storage cost of per-signal metadata for a
  VCD is real.
- **Filelist / Makefile parsing** should be added when the IR graph's
  value of having compile-order dependency edges exceeds the cost of
  maintaining yet another regex-structural parser.
- **Semantic IR resolution** ("assertion covers register X") should
  be added when retrieval queries actually need that level of
  precision, and only as a separate pass on top of the structural
  IR — not as a refactor of the builder.
- **Switching the embedding default to a pretrained model** should
  happen when a real Ollama / vLLM / llama.cpp endpoint is
  reachable. The TF-IDF provider stays useful for tests, CI
  environments without GPUs, and small-corpus benchmarks, but the
  re-fit-invalidates-everything property makes it the wrong default
  for production. When the switch happens, document it in
  `embedding/factory.py:build_embedding_provider` (and in
  `core/settings.py:Settings.embedding_provider`'s default).
- **An in-package Postgres-backed `ManifestStore`** should be added
  when (a) the scanner is being run by a long-lived process and the
  JSON-file store's lack of locking is a problem, *and* (b) the
  storage team agrees the manifest belongs in this package rather
  than staying in `storage/postgres/`. Today the scanner is run by
  the worker, which already has a Postgres session — no in-package
  implementation needed.
- **A paged git log** for multi-decade histories. `git_metadata`
  today operates on one bounded commit range; very large repos
  should page the `since..HEAD` range. This is a performance
  optimization, not a correctness fix — keep it scoped to the
  `git_metadata` module.
- **A configurable hashing algorithm** beyond the existing
  blake3/sha256 fallback. Swapping requires a full re-index because
  manifests store the algorithm name; only add a third option when a
  real operational need appears.
