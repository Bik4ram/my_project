"""
Service layer — the single place where business operations compose
storage, retrieval, embedding, and LLM primitives.

The CLI (`run_local_qa.py`), the HTTP API routers (`api/routers/`), and
tests all depend on services. Services are constructed with explicit
dependencies (an SQLAlchemy session, a settings instance, provider
factories) — no module-level singletons inside services, so a test can
build an `IngestionService` against a tempdir without touching the
module-global rate limiter or query cache.

Two services today:

- `IngestionService.index(project_id, root_path) -> IndexResult`
  Scan → parse → IR → chunk → embed → persist. The CLI's
  `cmd_index` and the API's `index_project` both call this.

- `QueryService.ask(project_id, question, ...) -> QueryAnswer`
  Embed query → retrieve (vector + lexical + RRF + graph + rerank) →
  prompt → agent loop → final answer. The CLI's `cmd_ask` and the API's
  `/query` and `/query/stream` both call this.

Both services also write a debug-artifacts bundle (`debug_artifacts_dir`)
when `Settings.log_level` is `DEBUG` or `TRACE`. See
`core.debug_artifacts`. At INFO+ the writer is a no-op.

Future services (eval, list-files, delete-project) follow the same shape.
"""
from __future__ import annotations

import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semrag.reasoning.coding.tools import CodingToolset

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.core.debug_artifacts import DebugArtifactWriter, new_run_id
from semrag.core.observability import get_logger, stage, timer
from semrag.core.query_trace import QueryTrace, QueryTraceStage
from semrag.core.settings import Settings
from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.interfaces import EmbeddingCache, EmbeddingProvider
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.evaluation.citation_validator import validate_citations
from semrag.reasoning.agents.agent_loop import AgentLoop
from semrag.reasoning.agents.tools import ListFilesTool, ReadFileTool, SearchCodebaseTool, ToolRegistry
from semrag.reasoning.grounding.gate import (
    build_revision_messages,
    check_grounding,
    filter_supported_portion,
)
from semrag.reasoning.llm.interfaces import LLMProvider
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.reasoning.prompting.templates import SYSTEM_PROMPT
from semrag.retrieval.intent import is_structural_query
from semrag.retrieval.pipeline import HybridRetriever
from semrag.retrieval.query_expansion import build_query_expansion_config
from semrag.retrieval.ranking.diversity import DEFAULT_DIVERSITY
from semrag.retrieval.reranking import LexicalOverlapReranker
from semrag.retrieval.source_authority import DEFAULT_SOURCE_AUTHORITY
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.postgres.manifest_store import PostgresManifestStore
from semrag.storage.postgres.models import FileRow
from semrag.storage.vector.models import VectorSearchFilter
from semrag.storage.vector.pgvector_store import PgvectorStore

logger = get_logger(__name__)


@dataclass
class IndexResult:
    project_id: str
    scan_id: str
    file_count: int
    chunk_count: int
    added: int
    modified: int
    deleted: int


@dataclass
class QueryAnswer:
    final_answer: str | None
    iterations_used: int
    hit_iteration_limit: bool
    messages: list[dict]
    debug: "DebugSummary | None" = None


@dataclass
class DetailedQueryAnswer:
    final_answer: str | None
    iterations_used: int
    hit_iteration_limit: bool
    messages: list[dict]
    considered_chunks_count: int = 0
    debug: "DebugSummary | None" = None
    included_chunks: list[Chunk] = None
    agent_messages: list[dict] = None
    citation_analysis: dict | None = None
    grounding: dict | None = None


@dataclass
class DebugSummary:
    """Compact, single-block snapshot of a query run for the CLI's
    `--debug` summary. Designed to be human-readable on a single
    screen — full request/response bodies live in the
    ``DebugArtifactWriter`` bundle, this is just the headline."""
    query: str
    intent: str
    retrieved_paths: list[str] = field(default_factory=list)
    included_chunk_count: int = 0
    excluded_chunk_count: int = 0
    estimated_tokens: int = 0
    iterations_used: int = 0
    hit_iteration_limit: bool = False
    llm_latency_ms: float = 0.0
    file_index_included: bool = False
    dedup_dropped: int = 0


class IngestionService:
    """Composes a full index run: scan → parse → IR → chunk → embed → persist."""

    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        embedding_cache: EmbeddingCache,
        settings: Settings,
        debug_writer: DebugArtifactWriter | None = None,
    ):
        self._session = session
        self._embedding_provider = embedding_provider
        self._embedding_cache = embedding_cache
        self._settings = settings
        self._parser_registry = build_default_registry()
        self._debug = debug_writer or DebugArtifactWriter(
            root_dir=settings.debug_artifacts_dir,
            log_level=settings.log_level,
            retention=settings.debug_artifact_retention,
        )

    def index(
        self,
        project_id: str,
        root_path: str,
        rtl_root: str | None = None,
        tb_root: str | None = None,
        docs_root: str | None = None,
    ) -> IndexResult:
        root = Path(root_path).resolve() if root_path else None
        area_roots = {
            area: raw
            for area, raw in (("rtl", rtl_root), ("tb", tb_root), ("docs", docs_root))
            if raw
        }
        is_multi_root = bool(area_roots)
        if not is_multi_root and root is None:
            raise ValueError("index() requires root_path or at least one area root")

        run_id = new_run_id()
        self._debug.start_run(run_id)

        with stage("ingest.scan", logger=logger, project_id=project_id, run_id=run_id) as stats:
            with timer("ingest.scan.duration_ms", project_id=project_id) as _extras:
                scanner = Scanner(manifest_store=PostgresManifestStore(self._session))
                if is_multi_root:
                    manifest, change_set = scanner.scan_multi_root(
                        project_id, area_roots, display_root=str(root) if root else None
                    )
                else:
                    manifest, change_set = scanner.scan(project_id, str(root))
            self._session.flush()
            stats["file_count"] = manifest.file_count
            stats["added"] = len(change_set.added)
            stats["modified"] = len(change_set.modified)
            stats["deleted"] = len(change_set.deleted_paths)
        self._debug.write_json("scan", "manifest", manifest.model_dump(mode="json"))

        logger.info(
            "scan complete",
            extra={
                "project_id": project_id,
                "file_count": manifest.file_count,
                "added": len(change_set.added),
                "modified": len(change_set.modified),
                "deleted": len(change_set.deleted_paths),
            },
        )

        with stage("ingest.parse", logger=logger, project_id=project_id, files=manifest.file_count) as stats:
            with timer("ingest.parse.duration_ms", project_id=project_id, files=manifest.file_count):
                parse_results: dict[str, Any] = {}
                for rel, rec in manifest.files.items():
                    # absolute_path on the record is authoritative for
                    # every project shape (single-root and multi-root).
                    result = self._parser_registry.parse_file(Path(rec.absolute_path), rec)
                    if result is not None:
                        parse_results[rel] = result
                        if self._debug.is_active:
                            self._debug.write_parse(rel, _summarize_parse(result))
                stats["parsed"] = len(parse_results)

        with stage("ingest.ir", logger=logger, project_id=project_id) as stats:
            with timer("ingest.ir.duration_ms", project_id=project_id):
                ir_graph = IRBuilder().build(project_id, manifest.files, parse_results)
            stats["nodes"] = len(ir_graph.nodes)
            stats["edges"] = len(ir_graph.edges)
        self._debug.write_json("ir", "graph", ir_graph.model_dump(mode="json"))
        self._debug.write_json(
            "knowledge_graph",
            "graph_summary",
            {"node_count": len(ir_graph.nodes), "edge_count": len(ir_graph.edges)},
        )

        with stage("ingest.chunk", logger=logger, project_id=project_id) as stats:
            with timer("ingest.chunk.duration_ms", project_id=project_id):
                project_paths = None
                if is_multi_root:
                    from semrag.core.project_paths import ProjectPaths

                    project_paths = ProjectPaths(
                        project_id=project_id,
                        root_path=root,
                        rtl_root=Path(rtl_root).resolve() if rtl_root else None,
                        tb_root=Path(tb_root).resolve() if tb_root else None,
                        docs_root=Path(docs_root).resolve() if docs_root else None,
                    )
                chunks = chunk_project(
                    str(root) if root else "",
                    manifest.files,
                    parse_results,
                    ir_graph,
                    project_paths=project_paths,
                )
            stats["chunk_count"] = len(chunks)
        logger.info("chunking complete", extra={"project_id": project_id, "chunk_count": len(chunks)})
        self._debug.write_json(
            "chunks",
            "chunk_metadata",
            [
                {
                    "chunk_id": c.chunk_id,
                    "relative_path": c.relative_path,
                    "chunk_type": c.chunk_type,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "language": c.language,
                    "category": c.category,
                    "parser_name": c.parser_name,
                    "chunker_name": c.chunker_name,
                    "symbol_name": c.symbol_name,
                }
                for c in chunks
            ],
        )

        with stage("ingest.embed", logger=logger, project_id=project_id, chunks=len(chunks)) as stats:
            with timer("ingest.embed.duration_ms", project_id=project_id, chunks=len(chunks)):
                vector_chunks = embed_chunks(chunks, self._embedding_provider, self._embedding_cache)
            stats["embedded"] = len(vector_chunks)
        self._debug.write_json(
            "embeddings",
            "embedding_stats",
            {
                "model": self._embedding_provider.model_name,
                "total_chunks": len(chunks),
                "embedded_chunks": len(vector_chunks),
            },
        )

        with stage("ingest.persist", logger=logger, project_id=project_id) as stats:
            with timer("ingest.persist.duration_ms", project_id=project_id):
                IRGraphRepository(self._session).save(ir_graph)
                self._session.flush()
                if vector_chunks:
                    PgvectorStore(self._session).upsert_chunks(vector_chunks)
                    self._session.flush()
                self._session.commit()
            stats["nodes_persisted"] = len(ir_graph.nodes)
            stats["chunks_persisted"] = len(vector_chunks)

        self._debug.write_json("timings", "pipeline", {"note": "per-stage durations are emitted as stage=... event=finish elapsed_ms log records"})
        self._debug.run_completed()

        logger.info(
            "index complete",
            extra={"project_id": project_id, "chunk_count": len(vector_chunks)},
        )

        return IndexResult(
            project_id=project_id,
            scan_id=manifest.scan_id,
            file_count=manifest.file_count,
            chunk_count=len(vector_chunks),
            added=len(change_set.added),
            modified=len(change_set.modified),
            deleted=len(change_set.deleted_paths),
        )


def _summarize_parse(result: Any) -> dict[str, Any]:
    """Coarse summary of a parse result for debug artifacts — full IR graph
    is written separately, this is just enough to see "did we parse this
    file, what type, how many nodes"."""
    summary: dict[str, Any] = {"type": type(result).__name__}
    if hasattr(result, "node_count"):
        summary["node_count"] = result.node_count
    if hasattr(result, "tokens"):
        summary["token_count"] = len(result.tokens or [])
    if hasattr(result, "text"):
        summary["text_length"] = len(result.text or "")
    return summary


# ---------------------------------------------------------------------------
# Chunk hygiene helpers
# ---------------------------------------------------------------------------

# Chunks whose entire purpose is runtime output (simulation logs,
# coverage dumps, lint/timing reports) drown the prompt with noise for
# architectural questions. Filter them out when the question is about
# the shape of the repo, not a debug session.
_NOISE_CHUNK_TYPES: frozenset[str] = frozenset(
    {
        "log",
        "coverage_report",
        "lint_report",
        "timing_report",
        "waveform",
    }
)

# File extensions that signal "this is a runtime artifact, not source"
# even when the chunker falls back to ``generic_window`` (which is what
# the log/coverage parsers produce because the bespoke parsers don't
# extract structured chunks from them). ``.txt`` is included because the
# benchmark repo's coverage reports (``sim/coverage/*.txt``) get
# ``generic_window`` chunks but are still runtime artifacts — the
# scanner already classified them as ``coverage_report``; this extension
# list catches anything the scanner didn't tag.
_NOISE_FILE_EXTENSIONS: tuple[str, ...] = (".log", ".vcd", ".fsdb", ".wlf", ".rpt", ".txt")

# Path fragments that always indicate runtime output (regardless of
# extension or chunk_type). The scanner may classify the file correctly
# but the coverage/log files are still chunked as ``generic_window``
# because the parser is non-structural, so a chunk_type filter misses
# them — path matching closes the gap.
_NOISE_PATH_FRAGMENTS: tuple[str, ...] = (
    "/sim/coverage/",
    "/sim/log/",
    "/coverage/",
    "/logs/",
    "/log/",
)

# A chunk is repetitive noise only when a single non-blank line both
# repeats beyond this count AND accounts for at least this fraction of
# the chunk's non-blank lines — the signature of a log/coverage dump
# (the [STATE_COV_ITEM] block repeats the same line 60+ times). Real
# source chunks are never flagged: blank lines are ignored (file-header
# banners and code formatting repeat them), and RTL/SystemVerilog chunks
# legitimately reuse keywords like ``end`` and signal assignments, but
# never rise to a quarter of their non-blank content.
_MAX_REPEATED_LINE_COUNT = 5
_MAX_REPEATED_LINE_FRACTION = 0.25


def _is_repetitive_noise(text: str) -> bool:
    """True when a chunk is mostly the same non-blank line repeated
    (typical of log/coverage files chunked by the generic sliding-window
    chunker). Cheap exact-match check — no fuzzy/semantic dedup needed
    because the problem is byte-identical repetition."""
    if not text:
        return False
    non_blank = [line for line in text.splitlines() if line.strip()]
    if not non_blank:
        return False
    most_common_count = Counter(non_blank).most_common(1)[0][1]
    return (
        most_common_count > _MAX_REPEATED_LINE_COUNT
        and most_common_count > _MAX_REPEATED_LINE_FRACTION * len(non_blank)
    )


def _deduplicate_chunks(results: list) -> tuple[list, int]:
    """Drop chunks that are pure log noise, and within the same source
    file collapse near-identical chunks to the highest-scored one.

    Returns ``(filtered_results, dropped_count)`` — order preserved from
    the input so the retriever's ranking continues to drive the top-K
    selection done by the prompt assembler."""
    kept: list = []
    dropped = 0

    for result in results:
        if _is_repetitive_noise(result.text):
            dropped += 1
            continue
        kept.append(result)

    by_path: dict[str, list] = defaultdict(list)
    for result in kept:
        by_path[result.relative_path].append(result)

    final: list = []
    for path, group in by_path.items():
        if len(group) == 1:
            final.extend(group)
            continue
        # When a single file contributes several chunks of nearly
        # identical text (e.g. multiple `generic_window` slices of the
        # same coverage report), keep only the highest-scored one.
        unique: list = []
        seen_signatures: set[tuple[int, ...]] = set()
        for result in sorted(group, key=lambda r: r.fused_score, reverse=True):
            signature = _line_signature(result.text)
            if signature in seen_signatures:
                dropped += 1
                continue
            seen_signatures.add(signature)
            unique.append(result)
        final.extend(unique)

    return final, dropped


def _line_signature(text: str) -> tuple[int, ...]:
    """Coarse fingerprint: the hash of each non-trivial line, in order.
    Two chunks with the same signature are essentially the same content
    sliced differently. Trivial lines (whitespace-only) are skipped so
    formatting differences don't break the match."""
    sig: list[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        sig.append(hash(stripped))
    return tuple(sig)


def _citation_analysis(answer: str, included_chunks: list) -> dict:
    """Run citation validation on the final answer against the included
    chunks. Returns a summary dict suitable for QueryTrace storage."""
    validation = validate_citations(answer, included_chunks)
    return {
        "cited_paths": list(validation.cited_paths),
        "valid_count": len(validation.valid_citations),
        "fabricated_count": len(validation.fabricated_citations),
        "fabrication_rate": round(validation.fabrication_rate, 4),
    }


def _controlled_response(
    report,
    original_answer: str,
    settings: Settings,
    included_chunks: list | None = None,
    file_index_text: str | None = None,
) -> str:
    """Build the conservative response when an answer cannot be presented
    as fully grounded: keep the supported portion and explicitly flag the
    claims that the retrieved evidence does not support. For a
    source-specific answer that simply cites nothing, return the
    uncertainty message from the product style."""
    if report.unsupported_entities:
        supported = filter_supported_portion(
            original_answer, report, included_chunks, file_index_text
        ).strip()
        note = (
            "The following claims were not supported by the retrieved evidence "
            "and were removed: {}. The remaining text above is grounded in the "
            "indexed source.".format(", ".join(report.unsupported_entities))
        )
        if supported:
            return f"{supported}\n\n---\n\n{note}"
        return (
            "I couldn't verify enough of this explanation against the indexed "
            "source. The retrieved evidence does not support: {}.".format(
                ", ".join(report.unsupported_entities)
            )
        )
    return (
        "I couldn't verify enough of this explanation against the indexed "
        "source. The retrieved evidence does not support the full answer. "
        "Please narrow the question or check the indexed files directly."
    )


def _apply_grounding_policy(
    question: str,
    original_answer: str,
    included_chunks: list,
    file_index_text: str | None,
    llm,
    settings: Settings,
) -> tuple[str, dict]:
    """Run the A1 grounding gate on the generated answer and apply the
    answer policy:

    1. ``grounded`` → return the answer unchanged.
    2. otherwise → one constrained regeneration that removes the
       unsupported claims (if enabled), re-checked by the gate.
    3. still not grounded → supported portion + explicit uncertainty
       (or the uncertainty message alone), never the unsupported answer.

    Returns ``(final_answer, grounding_info)`` where ``grounding_info``
    is a serializable dict describing what the gate did. Never invents a
    citation, path, or symbol.
    """
    report = check_grounding(original_answer, included_chunks, file_index_text, question)
    info: dict = {
        "status": report.status,
        "unsupported_entities": report.unsupported_entities,
        "unsupported_claims": report.unsupported_claims,
        "valid_citations": report.valid_citations,
        "fabricated_citations": report.fabricated_citations,
        "revision_attempted": False,
        "revised": False,
    }

    if report.status == "grounded":
        return original_answer, info

    if settings.grounding_gate_regenerate:
        attempts = max(1, settings.grounding_gate_max_revisions)
        for _ in range(attempts):
            info["revision_attempted"] = True
            messages = build_revision_messages(
                question,
                original_answer,
                report,
                included_chunks,
                SYSTEM_PROMPT,
                file_index_text,
            )
            try:
                revised = llm.complete(messages)
            except Exception as exc:  # pragma: no cover - network path
                logger.warning(
                    "grounding revision failed",
                    extra={"error": str(exc), "status": report.status},
                )
                break
            revised_text = (revised.content or "").strip()
            if len(revised_text) < settings.grounding_gate_min_length:
                continue
            revised_report = check_grounding(revised_text, included_chunks, file_index_text, question)
            if revised_report.status == "grounded":
                info["revised"] = True
                info["status"] = "revised"
                info["unsupported_entities"] = revised_report.unsupported_entities
                info["unsupported_claims"] = revised_report.unsupported_claims
                return revised_text, info
            report = revised_report
            info["unsupported_entities"] = revised_report.unsupported_entities
            info["unsupported_claims"] = revised_report.unsupported_claims

    info["status"] = "unsupported" if report.unsupported_entities else "insufficient_evidence"
    return (
        _controlled_response(report, original_answer, settings, included_chunks, file_index_text),
        info,
    )


def _filter_for_structural(results: list) -> list:
    """For architecture / overview questions: strip chunks that come
    from log, coverage, lint, timing, or waveform files. The model
    needs real design/testbench source for these answers, not runtime
    output that happens to share vocabulary with the question.

    Two filter passes:

    1. By ``chunk_type`` — explicit labels from the parser (log,
       coverage_report, lint_report, timing_report, waveform).
    2. By ``relative_path`` extension — catches files that fell
       through to the ``generic_window`` chunker because no bespoke
       parser claimed them (.log, .vcd, .fsdb, .wlf, .rpt). Without
       this, structural questions still pull in noise from the
       generic-window chunks of runtime artifacts.

    The filter is a *preference*, not a hard cut: if stripping noise
    would leave the LLM with no evidence at all, the unfiltered
    results are returned. Better to give the model a noisy log dump
    than an empty prompt — the file_index tells the model what the
    repo actually contains, and the dedup step has already stripped
    the worst repetition.
    """
    _MIN_KEEP = 4

    def is_noise(result) -> bool:
        if result.chunk_type in _NOISE_CHUNK_TYPES:
            return True
        path = result.relative_path.lower()
        if any(path.endswith(ext) for ext in _NOISE_FILE_EXTENSIONS):
            return True
        # Path-fragment match: catches coverage/log directories even when
        # the chunker produced ``generic_window`` (which is what
        # non-structural parsers emit). The benchmark case is
        # ``sim/coverage/code_coverage.txt`` — the chunk type is
        # ``generic_window`` so chunk_type filtering misses it.
        if any(fragment in path for fragment in _NOISE_PATH_FRAGMENTS):
            return True
        return False

    kept = [r for r in results if not is_noise(r)]
    if len(kept) >= _MIN_KEEP:
        return kept
    return list(results)


def _project_file_index(session: Session, project_id: str) -> str:
    """Render the project's files as a category-grouped tree, used as
    ``FILE INDEX`` evidence for structural queries.

    Cheap Postgres round-trip — one SELECT on `files`. The output is
    intended for the LLM prompt, not for human consumption, but it is
    human-readable so debug output is also useful.

    The format separates *category labels* from *paths* explicitly
    (``category: rtl_verilog`` rather than ``rtl_verilog/``) so the
    model doesn't mistake an indexing label for an actual filesystem
    directory — a small model will treat ``coverage_report/`` as if
    that were a real top-level directory even when the file lives at
    ``sim/coverage/code_coverage.txt``."""
    stmt = (
        select(FileRow.category, FileRow.relative_path)
        .where(FileRow.project_id == project_id)
        .order_by(FileRow.category, FileRow.relative_path)
    )
    rows = session.execute(stmt).all()

    by_category: dict[str, list[str]] = defaultdict(list)
    for category, path in rows:
        by_category[category].append(path)

    if not by_category:
        return f"(no indexed files in project {project_id})"

    lines: list[str] = [
        "Categories are indexing labels (not filesystem directories).",
        "Each path below is the actual relative path of a file in the project.",
    ]
    for category in sorted(by_category):
        lines.append(f"category: {category}")
        for path in by_category[category]:
            lines.append(f"  {path}")
    return "\n".join(lines)


class QueryService:
    """Composes a query: embed → retrieve → prompt → agent → answer."""

    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        llm: LLMProvider,
        settings: Settings,
        project_root: str | None = None,
        debug_writer: DebugArtifactWriter | None = None,
    ):
        self._session = session
        self._embedding_provider = embedding_provider
        self._llm = llm
        self._settings = settings
        self._project_root = project_root
        self._debug = debug_writer or DebugArtifactWriter(
            root_dir=settings.debug_artifacts_dir,
            log_level=settings.log_level,
            retention=settings.debug_artifact_retention,
        )

    def _build_retriever(self) -> HybridRetriever:
        reranker = (
            LexicalOverlapReranker(
                enable_metadata_tiebreak=self._settings.retrieval_rerank_metadata_tiebreak,
                near_tie_epsilon=self._settings.retrieval_rerank_near_tie_epsilon,
            )
            if self._settings.reranker == "lexical_overlap"
            else None
        )
        source_authority_config = None
        overrides = self._settings.source_authority_overrides or {}
        max_boost = self._settings.source_authority_max_boost
        if overrides or max_boost is not None:
            source_authority_config = DEFAULT_SOURCE_AUTHORITY.with_overrides(
                overrides, max_boost=max_boost
            )
        diversity_config = DEFAULT_DIVERSITY.model_copy(update={
            "enabled": self._settings.retrieval_diversity_enabled,
            "max_per_file": self._settings.retrieval_diversity_max_per_file,
            "max_per_category": self._settings.retrieval_diversity_max_per_category,
            "graph_expansion_quota": self._settings.retrieval_diversity_graph_expansion_quota,
            "neighbour_window": self._settings.retrieval_diversity_neighbour_window,
            "pool_multiplier": self._settings.retrieval_diversity_pool_multiplier,
        })
        return HybridRetriever(
            self._session,
            reranker=reranker,
            enable_graph_expansion=self._settings.enable_graph_expansion,
            enable_intent_boost=self._settings.enable_intent_boost,
            enable_doc_type_prior=self._settings.retrieval_doc_type_prior_enabled,
            noisy_source_path_terms=self._settings.noisy_source_path_terms,
            noisy_source_penalty=self._settings.noisy_source_penalty,
            source_authority_config=source_authority_config,
            diversity_config=diversity_config,
            query_expansion_config=build_query_expansion_config(self._settings),
            graph_expansion_max_depth=self._settings.graph_expansion_max_depth,
            graph_expansion_dependencies_max_depth=self._settings.graph_expansion_dependencies_max_depth,
            graph_expansion_max_per_result=self._settings.graph_expansion_max_per_result,
            graph_expansion_dependencies_max_per_result=self._settings.graph_expansion_dependencies_max_per_result,
            graph_expansion_dependencies_per_depth_limit=self._settings.graph_expansion_dependencies_per_depth_limit,
        )

    def _build_tools(self, project_id: str) -> ToolRegistry:
        retriever = self._build_retriever()
        embed_query_fn = lambda q: self._embedding_provider.embed_texts([q])[0]
        tools = [
            SearchCodebaseTool(retriever, embed_query_fn=embed_query_fn, project_id=project_id),
            ListFilesTool(self._session, project_id),
        ]
        if self._project_root:
            tools.append(ReadFileTool(self._project_root))
        return ToolRegistry(tools)

    def build_coding_toolset(
        self, project_id: str, project_root: str, job_manager, project_paths=None
    ) -> CodingToolset:
        """Build the A2 in-process coding toolset over the frozen core.

        Reuses the same retriever construction as ``_build_tools`` so the
        coding agent searches exactly what /query and /answer search.
        ``CodingToolset`` is a Protocol; concrete imports stay out of
        business code (see the repository guideline). ``project_paths`` (a
        ``core.project_paths.ProjectPaths``) is optional; when provided,
        file operations resolve area-prefixed paths and the docs root is
        read-only."""
        from semrag.reasoning.coding.tools import InProcessCodingToolset

        retriever = self._build_retriever()
        embed_query_fn = lambda q: self._embedding_provider.embed_texts([q])[0]
        return InProcessCodingToolset(
            project_id=project_id,
            session=self._session,
            project_root=Path(project_root).resolve(),
            retriever=retriever,
            embed_query_fn=embed_query_fn,
            settings=self._settings,
            job_manager=job_manager,
            project_paths=project_paths,
        )

    def _project_file_index(self, project_id: str) -> str:
        return _project_file_index(self._session, project_id)

    def ask(self, project_id: str, question: str) -> QueryAnswer:
        """Run retrieval → prompt assembly → agent loop → final answer.

        Why seed the conversation with retrieved context (not just send
        `[SYSTEM_PROMPT, question]` and rely on the agent's
        ``search_codebase`` tool): without it, every question costs at least
        one LLM call to discover the same context the retriever already
        knows about, and small models often pick the wrong query against
        the tool. Pre-fetching via the retriever is the documented RAG
        pattern; the agent's tools remain useful for follow-up refinement
        ("show me the full function" via ``read_file``, etc.).

        Structural queries ("summarize this repository", "what's in this
        repo", etc.) additionally receive a ``FILE INDEX`` block before
        the retrieved context, so the LLM can ground structural claims
        in real paths instead of inferring them from noise (the benchmark
        case where "summarize" pulls in 60-line simulation-log dumps and
        the model hallucinates a Windows path).
        """
        from semrag.core.observability import timer as _timer

        _start = time.perf_counter()
        run_id = new_run_id()
        self._debug.start_run(run_id)
        structural = is_structural_query(question)
        debug_summary = DebugSummary(query=question, intent="structural" if structural else "general")

        # Build the full QueryTrace (populated only when debug is active)
        trace = QueryTrace(
            run_id=run_id,
            project_id=project_id,
            query_text=question,
            timestamp=time.time(),
        ) if self._debug.is_active else None

        try:
            with stage("query.ask", logger=logger, project_id=project_id, run_id=run_id) as stats:
                # Stages 1-2: Preprocessing
                if trace is not None:
                    trace.preprocessing.started_at = time.time()
                    trace.preprocessing.data = {
                        "is_structural": structural,
                    }

                with stage("query.retrieve", logger=logger, project_id=project_id) as retrieve_stats:
                    ranking_trace = None
                    trace_sink = (
                        (lambda t: self._debug.write_json(
                            "retrieval", "ranking_trace", {"query": t.query, "chunks": [c.to_dict() for c in t.chunks()]}
                        ))
                        if self._debug.is_active
                        else None
                    )

                    # Stage 5: Embedding
                    if trace is not None:
                        trace.embedding.started_at = time.time()
                    with timer("query.retrieve.duration_ms", project_id=project_id):
                        retriever = self._build_retriever()
                        _embed_start = time.perf_counter()
                        query_embedding = self._embedding_provider.embed_texts([question])[0]
                        _embed_latency = (time.perf_counter() - _embed_start) * 1000
                        if trace is not None:
                            trace.embedding.completed_at = time.time()
                            trace.embedding.duration_ms = _embed_latency
                            trace.embedding.data = {
                                "model": self._embedding_provider.model_name,
                            }

                        import inspect
                        try:
                            sig = inspect.signature(retriever.retrieve)
                        except (TypeError, ValueError):
                            sig = None
                        if sig is not None and "return_trace" in sig.parameters:
                            results = retriever.retrieve(
                                question,
                                query_embedding,
                                project_id=project_id,
                                top_k=self._settings.final_top_k,
                                filters=VectorSearchFilter(project_id=project_id),
                                return_trace=self._debug.is_active,
                            )
                            if self._debug.is_active and isinstance(results, tuple):
                                results, ranking_trace = results
                        else:
                            results = retriever.retrieve(
                                question,
                                query_embedding,
                                project_id=project_id,
                                top_k=self._settings.final_top_k,
                                filters=VectorSearchFilter(project_id=project_id),
                            )
                        if trace_sink is not None and hasattr(retriever, "set_trace_sink"):
                            retriever.set_trace_sink(trace_sink)

                    # Stages 12-13: Dedup + structural filter (post-retrieval)
                    results, dedup_dropped = _deduplicate_chunks(results)
                    debug_summary.dedup_dropped = dedup_dropped
                    if structural:
                        results = _filter_for_structural(results)
                    retrieve_stats["retrieved"] = len(results)
                    debug_summary.retrieved_paths = [
                        f"{r.relative_path}:{r.line_start}-{r.line_end}" for r in results[:8]
                    ]
                    considered_chunks_count = len(results)

                # Stage 3: Intent — already computed via is_structural_query
                if trace is not None:
                    trace.intent_classification.data = {
                        "intent": "structural" if structural else "general",
                    }

                file_index_text: str | None = None
                if structural:
                    with timer("query.file_index.duration_ms", project_id=project_id):
                        file_index_text = self._project_file_index(project_id)
                    debug_summary.file_index_included = bool(file_index_text)
                if trace is not None:
                    trace.preprocessing.completed_at = time.time()
                    trace.preprocessing.data["file_index_included"] = bool(file_index_text)
                    trace.preprocessing.data["file_index_length"] = len(file_index_text or "")
                    trace.preprocessing.duration_ms = (time.time() - trace.preprocessing.started_at) * 1000

                # Stages 6-15: Retrieval trace
                if trace is not None and ranking_trace is not None:
                    trace.retrieval.data = {
                        "query": ranking_trace.query,
                        "chunks": [c.to_dict() for c in ranking_trace.chunks()],
                        "total_candidates": len(ranking_trace.chunks()),
                        "final_count": len(results),
                        "dedup_dropped": dedup_dropped,
                    }

                # Stage 4: Query expansion — data recorded by the
                # retriever's RankingTrace (RI-05), copied onto the
                # QueryTrace stage for the debug bundle.
                if trace is not None and ranking_trace is not None:
                    trace.query_expansion.data = {
                        "enabled": ranking_trace.expansion_enabled,
                        "expansions": ranking_trace.expansions,
                        "latency_ms": ranking_trace.expansion_latency_ms,
                        "failed": ranking_trace.expansion_failed,
                    }

                with stage("query.prompt", logger=logger, project_id=project_id) as prompt_stats:
                    # Stages 16-18: Context assembly + prompt construction + token estimation
                    if trace is not None:
                        trace.context_assembly.started_at = time.time()
                    with timer("query.prompt.duration_ms", project_id=project_id):
                        prompt = PromptBuilder(
                            model_context_window=self._settings.llm_context_window,
                            reserved_for_response=self._settings.llm_reserved_for_response,
                        ).build(
                            question,
                            results,
                            file_index=file_index_text,
                            max_chunks=6 if structural else None,
                        )
                    prompt_stats["included_chunks"] = len(prompt.included_chunks)
                    prompt_stats["excluded_chunks"] = prompt.excluded_chunk_count
                    prompt_stats["estimated_tokens"] = prompt.estimated_total_tokens
                    debug_summary.included_chunk_count = len(prompt.included_chunks)
                    debug_summary.excluded_chunk_count = prompt.excluded_chunk_count
                    debug_summary.estimated_tokens = prompt.estimated_total_tokens

                    if trace is not None:
                        now = time.time()
                        trace.context_assembly.completed_at = now
                        trace.context_assembly.duration_ms = (now - trace.context_assembly.started_at) * 1000
                        trace.context_assembly.data = {
                            "included_count": len(prompt.included_chunks),
                            "excluded_count": prompt.excluded_chunk_count,
                            "budget_tokens": self._settings.llm_context_window - self._settings.llm_reserved_for_response,
                            "max_chunks": 6 if structural else None,
                        }
                        trace.prompt_construction.data = {
                            "message_count": len(prompt.messages),
                            "system_prompt": SYSTEM_PROMPT[:200],
                        }
                        trace.token_estimation.data = {
                            "estimated_total_tokens": prompt.estimated_total_tokens,
                            "context_window": self._settings.llm_context_window,
                            "reserved_for_response": self._settings.llm_reserved_for_response,
                        }

                if self._debug.is_active:
                    self._debug.write_json(
                        "retrieval",
                        "top_results",
                        [
                            {
                                "chunk_id": r.chunk_id,
                                "relative_path": r.relative_path,
                                "line_start": r.line_start,
                                "line_end": r.line_end,
                                "fused_score": r.fused_score,
                            }
                            for r in results
                        ],
                    )
                    self._debug.write_text(
                        "prompt",
                        "final_prompt.txt",
                        "\n\n".join(m.content for m in prompt.messages),
                    )

                with stage("query.agent", logger=logger, project_id=project_id) as agent_stats:
                    # Stages 19-20: LLM call
                    if trace is not None:
                        trace.llm_stage.started_at = time.time()
                    with _timer("query.agent.duration_ms", project_id=project_id):
                        tools = self._build_tools(project_id)
                        loop = AgentLoop(
                            self._llm, tools, max_iterations=self._settings.llm_max_iterations
                        )
                        messages = [m.model_dump() for m in prompt.messages]
                        result = loop.run(messages)
                    agent_stats["iterations"] = result.iterations_used
                    agent_stats["hit_iteration_limit"] = bool(result.hit_iteration_limit)
                    debug_summary.iterations_used = result.iterations_used
                    debug_summary.hit_iteration_limit = bool(result.hit_iteration_limit)
                    # Extract agent messages and citation analysis for the response
                    agent_messages = result.messages
                    citation_analysis = None
                    grounding = None
                    final_answer = result.final_answer
                    if result.final_answer:
                        if self._settings.grounding_gate_enabled:
                            final_answer, grounding = _apply_grounding_policy(
                                question,
                                result.final_answer,
                                prompt.included_chunks,
                                file_index_text,
                                self._llm,
                                self._settings,
                            )
                        citation_analysis = _citation_analysis(final_answer, prompt.included_chunks)
                    if trace is not None:
                        now = time.time()
                        trace.llm_stage.completed_at = now
                        trace.llm_stage.duration_ms = (now - trace.llm_stage.started_at) * 1000
                        trace.llm_stage.data = {
                            "provider": self._llm.provider_name,
                            "model": self._llm.model_name,
                            "iterations_used": result.iterations_used,
                            "hit_iteration_limit": bool(result.hit_iteration_limit),
                            "grounding_status": (grounding or {}).get("status"),
                        }

                stats["iterations"] = result.iterations_used
                stats["hit_iteration_limit"] = bool(result.hit_iteration_limit)

                if trace is not None:
                    # Stage 21: Citation extraction
                    if final_answer:
                        trace.citation_extraction.data = _citation_analysis(final_answer, prompt.included_chunks)
                    # Stage 22: Final answer
                    trace.final_answer = final_answer
                    trace.iterations_used = result.iterations_used
                    trace.hit_iteration_limit = bool(result.hit_iteration_limit)
                    trace.total_duration_ms = (time.perf_counter() - _start) * 1000

            if self._debug.is_active:
                self._debug.write_json(
                    "llm",
                    "request",
                    {
                        "system": SYSTEM_PROMPT,
                        "user": question,
                        "provider": self._llm.provider_name,
                        "model": self._llm.model_name,
                        "included_chunks": [r.chunk_id for r in prompt.included_chunks],
                        "file_index_included": debug_summary.file_index_included,
                    },
                )
                self._debug.write_json(
                    "llm",
                    "response",
                    {
                        "iterations_used": result.iterations_used,
                        "hit_iteration_limit": result.hit_iteration_limit,
                        "final_answer": result.final_answer,
                        "grounding": grounding,
                    },
                )
                # Persist the full QueryTrace
                if trace is not None:
                    self._debug.write_query_trace(trace.model_dump(mode="json"), compress=False)
            return DetailedQueryAnswer(
                final_answer=final_answer,
                iterations_used=result.iterations_used,
                hit_iteration_limit=result.hit_iteration_limit,
                considered_chunks_count=considered_chunks_count,
                messages=result.messages,
                debug=debug_summary,
                included_chunks=prompt.included_chunks,
                agent_messages=agent_messages,
                citation_analysis=citation_analysis,
                grounding=grounding,
            )
        finally:
            self._debug.run_completed()

    def retrieve(self, project_id: str, question: str, top_k: int | None = None) -> list:
        """Non-agent retrieval path — used by API clients that only want
        chunks, not an answer."""
        retriever = self._build_retriever()
        embedding = self._embedding_provider.embed_texts([question])[0]
        from semrag.storage.vector.models import VectorSearchFilter
        return retriever.retrieve(
            question, embedding, project_id=project_id,
            top_k=top_k or self._settings.final_top_k,
            filters=VectorSearchFilter(project_id=project_id),
        )