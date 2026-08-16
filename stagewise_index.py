"""
Stage-by-stage ingestion runner for SemRAG.

`IngestionService.index()` currently runs scan -> parse -> ir -> chunk ->
embed -> persist as one unbroken call, with no checkpoint between stages.
This script calls the exact same internal building blocks
(`Scanner`, `ParserRegistry`, `IRBuilder`, `chunk_project`, `embed_chunks`,
`IRGraphRepository`, `PgvectorStore`) individually, pickling each stage's
output to `.semrag/stage_cache/<project_id>/` so you can:

  - run one stage at a time, as separate commands / separate shells
  - inspect the output of a stage before deciding to continue
  - re-run a later stage without re-running earlier ones
  - stop after any stage and resume later (state survives on disk)

Usage:
    python stagewise_index.py scan    --project-id myproj --root ~/projects/RTL_UVM
    python stagewise_index.py parse   --project-id myproj --root ~/projects/RTL_UVM
    python stagewise_index.py ir      --project-id myproj
    python stagewise_index.py chunk   --project-id myproj --root ~/projects/RTL_UVM
    python stagewise_index.py embed   --project-id myproj
    python stagewise_index.py persist --project-id myproj

    # or just run everything through in order, still one process per stage:
    python stagewise_index.py all     --project-id myproj --root ~/projects/RTL_UVM

Each command prints a summary so you can eyeball whether that stage did
what you expected before moving to the next one.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from semrag.core.settings import get_settings
from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.factory import build_embedding_cache, build_embedding_provider
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.postgres.manifest_store import PostgresManifestStore
from semrag.storage.postgres.session import build_session_factory
from semrag.storage.vector.pgvector_store import PgvectorStore

STAGE_DIR = Path(".semrag/stage_cache")


def _cache_dir(project_id: str) -> Path:
    d = STAGE_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(project_id: str, name: str, obj) -> None:
    path = _cache_dir(project_id) / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  -> wrote {path}")


def _load(project_id: str, name: str):
    path = _cache_dir(project_id) / f"{name}.pkl"
    if not path.exists():
        raise SystemExit(
            f"Missing checkpoint '{name}' for project {project_id!r} at {path}. "
            f"Run the earlier stage first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Stage 1: scan
# ---------------------------------------------------------------------
def stage_scan(project_id: str, root: str) -> None:
    settings = get_settings()
    settings = get_settings()
    session = build_session_factory(settings=settings)()
    try:
        scanner = Scanner(manifest_store=PostgresManifestStore(session))
        manifest, change_set = scanner.scan(project_id, root)
        session.flush()
        session.commit()
    finally:
        session.close()

    print(f"[scan] {manifest.file_count} files "
          f"(+{len(change_set.added)} added, ~{len(change_set.modified)} modified, "
          f"-{len(change_set.deleted_paths)} deleted)")
    _save(project_id, "manifest", manifest)
    _save(project_id, "change_set", change_set)
    _save(project_id, "root", root)


# ---------------------------------------------------------------------
# Stage 2: parse
# ---------------------------------------------------------------------
def stage_parse(project_id: str) -> None:
    manifest = _load(project_id, "manifest")
    root = Path(_load(project_id, "root")).resolve()

    registry = build_default_registry()
    parse_results = {}
    for rel, rec in manifest.files.items():
        result = registry.parse_file(root / rel, rec)
        if result is not None:
            parse_results[rel] = result

    print(f"[parse] parsed {len(parse_results)} / {manifest.file_count} files")
    _save(project_id, "parse_results", parse_results)


# ---------------------------------------------------------------------
# Stage 3: IR build
# ---------------------------------------------------------------------
def stage_ir(project_id: str) -> None:
    manifest = _load(project_id, "manifest")
    parse_results = _load(project_id, "parse_results")

    ir_graph = IRBuilder().build(project_id, manifest.files, parse_results)

    print(f"[ir] {len(ir_graph.nodes)} nodes, {len(ir_graph.edges)} edges")
    _save(project_id, "ir_graph", ir_graph)


# ---------------------------------------------------------------------
# Stage 4: chunk
# ---------------------------------------------------------------------
def stage_chunk(project_id: str) -> None:
    manifest = _load(project_id, "manifest")
    parse_results = _load(project_id, "parse_results")
    ir_graph = _load(project_id, "ir_graph")
    root = _load(project_id, "root")

    chunks = chunk_project(root, manifest.files, parse_results, ir_graph)

    print(f"[chunk] {len(chunks)} chunks")
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"    {t}: {n}")
    _save(project_id, "chunks", chunks)


# ---------------------------------------------------------------------
# Stage 5: embed
# ---------------------------------------------------------------------
def stage_embed(project_id: str, use_cache: bool = True) -> None:
    settings = get_settings()
    chunks = _load(project_id, "chunks")

    provider = build_embedding_provider(settings)
    cache = build_embedding_cache(settings) if use_cache else InMemoryEmbeddingCache()

    vector_chunks = embed_chunks(chunks, provider, cache)

    print(f"[embed] embedded {len(vector_chunks)} / {len(chunks)} chunks "
          f"using {provider.model_name}")
    _save(project_id, "vector_chunks", vector_chunks)


# ---------------------------------------------------------------------
# Stage 6: persist
# ---------------------------------------------------------------------
def stage_persist(project_id: str) -> None:
    ir_graph = _load(project_id, "ir_graph")
    vector_chunks = _load(project_id, "vector_chunks")

    settings = get_settings()
    session = build_session_factory(settings=settings)()
    try:
        IRGraphRepository(session).save(ir_graph)
        session.flush()
        if vector_chunks:
            PgvectorStore(session).upsert_chunks(vector_chunks)
            session.flush()
        session.commit()
    finally:
        session.close()

    print(f"[persist] wrote {len(ir_graph.nodes)} IR nodes, "
          f"{len(vector_chunks)} chunks to Postgres")


# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run SemRAG ingestion stage by stage")
    sub = parser.add_subparsers(dest="stage", required=True)

    for name in ("scan", "parse", "ir", "chunk", "embed", "persist", "all"):
        p = sub.add_parser(name)
        p.add_argument("--project-id", required=True)
        if name in ("scan", "all"):
            p.add_argument("--root", required=(name == "scan" or name == "all"))
        p.add_argument("--no-cache", action="store_true",
                        help="(embed/all only) don't use the persistent embedding cache")

    args = parser.parse_args()

    if args.stage == "scan":
        stage_scan(args.project_id, args.root)
    elif args.stage == "parse":
        stage_parse(args.project_id)
    elif args.stage == "ir":
        stage_ir(args.project_id)
    elif args.stage == "chunk":
        stage_chunk(args.project_id)
    elif args.stage == "embed":
        stage_embed(args.project_id, use_cache=not args.no_cache)
    elif args.stage == "persist":
        stage_persist(args.project_id)
    elif args.stage == "all":
        stage_scan(args.project_id, args.root)
        stage_parse(args.project_id)
        stage_ir(args.project_id)
        stage_chunk(args.project_id)
        stage_embed(args.project_id, use_cache=not args.no_cache)
        stage_persist(args.project_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
