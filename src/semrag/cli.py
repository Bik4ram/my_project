"""
Console-script entrypoint installed by `pip install -e .` as ``semrag``.

Owns the argparse surface and per-command glue. ``run_local_qa.py``
imports ``main`` from here so the legacy ``python run_local_qa.py``
form keeps working without duplicating command bodies.
"""
from __future__ import annotations

import argparse
import sys

from semrag.core.observability import configure_logging, get_logger
from semrag.core.settings import get_settings
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.factory import build_embedding_cache, build_embedding_provider
from semrag.reasoning.llm.factory import build_llm
from semrag.services import IngestionService, QueryService
from semrag.storage.postgres.session import build_session_factory

logger = get_logger(__name__)


class Console:
    """User-facing output. Stdlib logging handles observability;
    this is purely for the CLI's UX. Errors go to stderr."""

    @staticmethod
    def info(msg: str) -> None:
        print(msg)

    @staticmethod
    def error(msg: str) -> None:
        print(msg, file=sys.stderr)


def _format_debug(answer) -> str:
    """Single-screen debug summary for the `--debug` flag. Renders to
    plain text — no JSON parsing required. Intentionally terse: paths,
    counts, timings, and intent. Full request/response bodies live in
    the per-run debug-artifacts bundle, not in stdout."""
    dbg = answer.debug
    if dbg is None:
        return "(no debug summary attached)"
    lines = [
        "--- DEBUG SUMMARY ---",
        f"intent:               {dbg.intent}",
        f"included_chunks:      {dbg.included_chunk_count}",
        f"excluded_chunks:      {dbg.excluded_chunk_count}",
        f"dedup_dropped:        {dbg.dedup_dropped}",
        f"estimated_tokens:     {dbg.estimated_tokens}",
        f"file_index_included:  {dbg.file_index_included}",
        f"iterations:           {dbg.iterations_used}{' (LIMIT HIT)' if dbg.hit_iteration_limit else ''}",
        "top retrieved paths:",
    ]
    for p in dbg.retrieved_paths:
        lines.append(f"  - {p}")
    return "\n".join(lines)


def _build_ingestion_service(settings, *, ephemeral_cache: bool):
    session_factory = build_session_factory(settings=settings)
    session = session_factory()
    embedding_provider = build_embedding_provider(settings)
    cache = InMemoryEmbeddingCache() if ephemeral_cache else build_embedding_cache(settings)
    service = IngestionService(session, embedding_provider, cache, settings)
    return session, service


def _build_query_service(settings, project_root: str | None):
    session_factory = build_session_factory(settings=settings)
    session = session_factory()
    embedding_provider = build_embedding_provider(settings)
    llm = build_llm(settings)
    return session, QueryService(session, embedding_provider, llm, settings, project_root=project_root)


def cmd_index(project_id: str, root: str) -> int:
    settings = get_settings()
    try:
        session, service = _build_ingestion_service(settings, ephemeral_cache=False)
        try:
            Console.info(f"Indexing {root!r} into project {project_id!r}...")
            result = service.index(project_id, root)
            Console.info(
                f"Indexed {result.chunk_count} chunks across {result.file_count} files "
                f"({result.added} added, {result.modified} modified, {result.deleted} deleted)."
            )
            return 0
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.exception("index failed", extra={"project_id": project_id, "root": root})
        Console.error(f"index failed: {exc}")
        return 1


def cmd_ask(project_id: str, question: str, root: str | None, debug: bool) -> int:
    settings = get_settings()
    try:
        session, service = _build_query_service(settings, project_root=root)
        try:
            Console.info(f"Asking {settings.llm_provider}/{settings.llm_model} about project {project_id!r} ...")
            answer = service.ask(project_id, question)
            if answer.hit_iteration_limit:
                Console.info("(Agent hit its iteration limit without a final answer.)")
            Console.info(answer.final_answer or "(no answer produced)")
            if debug:
                Console.info(_format_debug(answer))
            return 0
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.exception("ask failed", extra={"project_id": project_id})
        Console.error(f"ask failed: {exc}")
        return 1


def main() -> int:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "cli starting",
        extra={
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
        },
    )

    parser = argparse.ArgumentParser(description="SemRAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a project directory into Postgres + pgvector")
    p_index.add_argument("--project-id", required=True)
    p_index.add_argument("--root", required=True)

    p_ask = sub.add_parser("ask", help="Ask a question against an indexed project")
    p_ask.add_argument("--project-id", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--root", required=False, help="Enables the read_file tool if provided.")
    p_ask.add_argument("--debug", action="store_true", help="Print a compact debug summary after the answer.")

    args = parser.parse_args()
    if args.command == "index":
        return cmd_index(args.project_id, args.root)
    if args.command == "ask":
        return cmd_ask(args.project_id, args.question, args.root, args.debug)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
