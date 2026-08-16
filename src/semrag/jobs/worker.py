"""
Worker function for executing indexing jobs in the background.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from semrag.core import metrics as metrics_module
from semrag.core.observability import get_logger
from semrag.core.settings import Settings
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.interfaces import EmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.jobs.manager import JobManager
from semrag.jobs.models import JobState
from semrag.jobs.pipeline import PipelineStage, ProgressCalculator
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.postgres.manifest_store import PostgresManifestStore
from semrag.storage.vector.pgvector_store import PgvectorStore

logger = get_logger(__name__)


def _update_job_progress(
    job_manager: JobManager,
    job_id: str,
    progress: int,
    stage: str,
) -> None:
    """Update job progress and current stage."""
    job_manager.update_job(
        job_id,
        progress=progress,
        current_stage=stage,
    )


def _update_job_statistics(
    job_manager: JobManager,
    job_id: str,
    statistics: dict,
) -> None:
    """Update job statistics by merging with existing ones."""
    job = job_manager.get_job(job_id)
    if job is None:
        return
    # Merge the new statistics with the existing ones
    updated_stats = {**job.statistics, **statistics}
    job_manager.update_job(job_id, statistics=updated_stats)


def _report_failure(
    job_manager: JobManager,
    job_id: str,
    error: str,
    start_time: float,
) -> None:
    """Mark job as failed with error and timing."""
    completed_at = time.time()
    job_manager.update_job(
        job_id,
        state=JobState.FAILED,
        error=str(error),
        completed_at=completed_at,
        duration=completed_at - start_time,
    )
    job = job_manager.get_job(job_id)
    # Wave 3: surface failed jobs in Prometheus so an alert can fire.
    if job is not None:
        metrics_module.INDEXING_JOBS_FAILED.labels(project_id=job.project_id).inc()
        metrics_module.INDEXING_DURATION.labels(project_id=job.project_id).observe(completed_at - start_time)
    logger.error(
        "Indexing job failed",
        extra={
            "job_id": job_id,
            "project_id": job.project_id if job else None,
            "error": str(error),
            "elapsed_time": completed_at - start_time,
        },
    )


def indexing_worker(
    job_id: str,
    project_id: str,
    root_path: str,
    job_manager: JobManager,
    settings: Settings,
    session_factory: Callable[[], Session],
    embedding_provider_factory: Callable[[Settings], object],  # returns EmbeddingProvider
    rtl_root: str | None = None,
    tb_root: str | None = None,
    docs_root: str | None = None,
) -> None:
    """
    Execute an indexing job in the background.

    ``root_path`` is the legacy single project root; ``rtl_root`` /
    ``tb_root`` / ``docs_root`` are the optional authoritative area roots.
    When any area root is set the project is scanned multi-root (every
    file's ``relative_path`` is area-prefixed); otherwise the legacy
    single-root path runs unchanged. This worker updates the job's
    progress through the stages of the ingestion pipeline.
    """
    start_time = time.time()
    root = Path(root_path).resolve() if root_path else None
    area_roots = {
        area: raw
        for area, raw in (
            ("rtl", rtl_root),
            ("tb", tb_root),
            ("docs", docs_root),
        )
        if raw
    }
    is_multi_root = bool(area_roots)
    logger.info(
        "Indexing job started",
        extra={
            "job_id": job_id,
            "project_id": project_id,
            "root_path": root_path,
            "multi_root": is_multi_root,
            "area_roots": area_roots or None,
        },
    )
    try:
        # Mark job as running
        job_manager.update_job(
            job_id,
            state=JobState.RUNNING,
            started_at=start_time,
        )

        # Initialize progress calculator
        progress_calc = ProgressCalculator()

        # We'll track completed stages and current stage
        completed_stages = []
        current_stage = PipelineStage.QUEUED
        current_stage_progress = 0.0

        # Helper to update progress and state
        def _update_progress(stage: PipelineStage, stage_progress: float = 0.0):
            nonlocal current_stage, current_stage_progress
            current_stage = stage
            current_stage_progress = stage_progress
            # Compute overall progress
            progress = progress_calc.calculate_progress(
                completed_stages, current_stage, current_stage_progress
            )
            _update_job_progress(job_manager, job_id, progress, stage.name.lower())

        # Create a new session for this job. The whole persist stage
        # runs inside one transaction; any failure inside the try-block
        # explicitly rolls back so partial state (manifest row + half a
        # chunk upsert) cannot survive. Stages before PERSISTING run
        # with ``flush()`` only — they accumulate dirty rows in the same
        # transaction but don't commit, so a failure at any point still
        # leaves the database consistent.
        session = session_factory()
        try:
            # Build embedding provider and cache
            embedding_provider = embedding_provider_factory(settings)
            embedding_cache: EmbeddingCache = InMemoryEmbeddingCache()

            # Stage: Scanning
            _update_progress(PipelineStage.SCANNING, 0.0)
            # Check for cancellation
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            scanner = Scanner(manifest_store=PostgresManifestStore(session))
            if is_multi_root:
                manifest, change_set = scanner.scan_multi_root(
                    project_id, area_roots, display_root=str(root) if root else None
                )
            else:
                manifest, change_set = scanner.scan(project_id, str(root))
            session.flush()
            completed_stages.append(PipelineStage.SCANNING)
            _update_progress(PipelineStage.SCANNING, 1.0)

            # Update statistics after scanning
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "files_scanned": manifest.file_count,
                },
            )
            logger.info(
                "Scanning completed",
                extra={"job_id": job_id, "files_scanned": manifest.file_count},
            )

            # Stage: Parsing
            _update_progress(PipelineStage.PARSING, 0.0)
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            from semrag.ingestion.parsers.pipeline import build_default_registry
            parser_registry = build_default_registry()
            parse_results: dict = {}
            parse_items = list(manifest.files.items())
            parse_workers = max(1, getattr(settings, "max_concurrent_parse_workers", 1))

            def _parse_one(item):
                rel, rec = item
                # absolute_path on the record is authoritative for every
                # project shape (single-root and multi-root alike).
                return rel, parser_registry.parse_file(Path(rec.absolute_path), rec)

            if parse_workers <= 1 or len(parse_items) <= 1:
                for rel, rec in parse_items:
                    result = parser_registry.parse_file(Path(rec.absolute_path), rec)
                    if result is not None:
                        parse_results[rel] = result
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=parse_workers) as pool:
                    futures = [pool.submit(_parse_one, item) for item in parse_items]
                    for fut in as_completed(futures):
                        rel, result = fut.result()
                        if result is not None:
                            parse_results[rel] = result
            completed_stages.append(PipelineStage.PARSING)
            _update_progress(PipelineStage.PARSING, 1.0)

            # Update statistics after parsing
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "files_parsed": len(parse_results),
                },
            )
            logger.info(
                "Parsing completed",
                extra={"job_id": job_id, "files_parsed": len(parse_results)},
            )

            # Stage: IR
            _update_progress(PipelineStage.IR, 0.0)
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            from semrag.ingestion.ir.builder import IRBuilder
            ir_graph = IRBuilder().build(project_id, manifest.files, parse_results)
            completed_stages.append(PipelineStage.IR)
            _update_progress(PipelineStage.IR, 1.0)
            logger.info(
                "IR building completed",
                extra={
                    "job_id": job_id,
                    "nodes": len(ir_graph.nodes),
                    "edges": len(ir_graph.edges),
                },
            )

            # Stage: Chunking
            _update_progress(PipelineStage.CHUNKING, 0.0)
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            from semrag.ingestion.chunking.pipeline import chunk_project
            from semrag.core.project_paths import ProjectPaths

            project_paths = None
            if is_multi_root:
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
                max_workers=max(1, getattr(settings, "max_concurrent_parse_workers", 1)),
                project_paths=project_paths,
            )
            completed_stages.append(PipelineStage.CHUNKING)
            _update_progress(PipelineStage.CHUNKING, 1.0)

            # Update statistics after chunking
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "chunks_generated": len(chunks),
                },
            )
            logger.info(
                "Chunking completed",
                extra={"job_id": job_id, "chunks_generated": len(chunks)},
            )

            # Stage: Embedding
            _update_progress(PipelineStage.EMBEDDING, 0.0)
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            # For TFIDF provider, we need to fit and save
            corpus_texts = [c.text for c in chunks]
            if isinstance(embedding_provider, TfidfEmbeddingProvider):
                embedding_provider.fit(corpus_texts)
                # Save the fitted provider for later use by the query endpoint
                from semrag.api.config import get_embedding_provider_dir
                embedding_provider.save(f"{get_embedding_provider_dir()}/{project_id}.joblib")

            vector_chunks = embed_chunks(chunks, embedding_provider, embedding_cache)
            completed_stages.append(PipelineStage.EMBEDDING)
            _update_progress(PipelineStage.EMBEDDING, 1.0)

            # Update statistics after embedding
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "chunks_embedded": len(vector_chunks),
                },
            )
            logger.info(
                "Embedding completed",
                extra={"job_id": job_id, "chunks_embedded": len(vector_chunks)},
            )

            # Stage: Persisting
            _update_progress(PipelineStage.PERSISTING, 0.0)
            job = job_manager.get_job(job_id)
            if job and job.state == JobState.CANCELLED:
                job_manager.update_job(job_id, state=JobState.CANCELLED, completed_at=time.time(), duration=time.time() - start_time)
                return

            # CRITICAL ORDERING: Delete chunks for deleted paths BEFORE
            # saving the IR graph (which deletes stale IR nodes). Chunks
            # reference IR nodes via FK source_node_id, so IR node deletion
            # would fail if chunks still exist.
            if change_set.deleted_paths:
                PgvectorStore(session).delete_by_paths(project_id, list(change_set.deleted_paths))
                session.flush()

            IRGraphRepository(session).save(ir_graph)
            session.flush()

            if vector_chunks:
                # Pages the upsert to keep individual INSERT statement size
                # bounded for very large projects.
                PgvectorStore(session).upsert_chunks_bulk(vector_chunks)
                session.flush()
            # Single transaction boundary for the whole persist stage:
            # manifest + IR + chunk upserts commit atomically. A failure
            # before this point triggers an explicit rollback in the
            # ``except`` branch below so partial writes never persist.
            session.commit()
            completed_stages.append(PipelineStage.PERSISTING)
            _update_progress(PipelineStage.PERSISTING, 1.0)

            # Update statistics after persisting
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "chunks_persisted": len(vector_chunks),
                    "added": len(change_set.added),
                    "modified": len(change_set.modified),
                    "deleted": len(change_set.deleted_paths),
                },
            )
            logger.info(
                "Persisting completed",
                extra={
                    "job_id": job_id,
                    "chunks_persisted": len(vector_chunks),
                    "added": len(change_set.added),
                    "modified": len(change_set.modified),
                    "deleted": len(change_set.deleted_paths),
                },
            )

            # Write debug artifacts and complete
            from semrag.core.debug_artifacts import DebugArtifactWriter, new_run_id
            run_id = new_run_id()
            debug_writer = DebugArtifactWriter(
                root_dir=settings.debug_artifacts_dir,
                log_level=settings.log_level,
                retention=settings.debug_artifact_retention,
            )
            debug_writer.start_run(run_id)
            debug_writer.run_completed()

            # Success
            completed_at = time.time()
            elapsed_time = completed_at - start_time
            # Calculate estimated remaining time (0 since we are done)
            # Calculate throughput (chunks per second)
            throughput = len(vector_chunks) / elapsed_time if elapsed_time > 0 else 0.0
            _update_job_statistics(
                job_manager,
                job_id,
                {
                    "elapsed_time": elapsed_time,
                    "estimated_remaining_time": 0.0,
                    "throughput": throughput,
                },
            )
            job_manager.update_job(
                job_id,
                state=JobState.COMPLETED,
                completed_at=completed_at,
                duration=elapsed_time,
            )
            # Wave 3: surface indexing outcomes in Prometheus. Counters
            # for files / chunks are incremented as they're produced
            # (see the calls scattered through this function); the
            # completion counter + duration histogram land here where
            # the job's terminal state is known.
            metrics_module.INDEXING_JOBS_COMPLETED.labels(project_id=project_id).inc()
            metrics_module.INDEXING_DURATION.labels(project_id=project_id).observe(elapsed_time)
            metrics_module.INDEXING_FILES_PROCESSED.labels(project_id=project_id).inc(
                manifest.file_count
            )
            metrics_module.INDEXING_CHUNKS_GENERATED.labels(project_id=project_id).inc(
                len(chunks)
            )
            metrics_module.INDEXING_CHUNKS_EMBEDDED.labels(project_id=project_id).inc(
                len(vector_chunks)
            )
            logger.info(
                "Indexing job completed successfully",
                extra={
                    "job_id": job_id,
                    "project_id": project_id,
                    "elapsed_time": elapsed_time,
                    "chunks": len(vector_chunks),
                },
            )
        except Exception as session_exc:
            # Anything inside the indexing pipeline that raised will land
            # here. Roll back the worker session so the database doesn't
            # keep half-written manifest rows / chunk upserts; the
            # ``finally`` below still closes the session. The outer
            # ``except Exception`` catches this with its own handler, but
            # we rollback first so the failure path is explicit rather
            # than relying on session.close()'s implicit rollback.
            try:
                session.rollback()
            except Exception:  # noqa: BLE001 — never let rollback mask the cause
                pass
            raise session_exc
        finally:
            session.close()

    except Exception as e:
        _report_failure(job_manager, job_id, str(e), start_time)