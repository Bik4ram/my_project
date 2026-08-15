"""Toolset abstraction for the A2 coding agent.

``CodingToolset`` is the Protocol the orchestrator depends on; the
production implementation (``InProcessCodingToolset``) reuses the frozen
SemiRAG core (retriever, IR symbol index, validation runner, job manager)
rather than shipping a second retrieval implementation. Tests substitute
a fake toolset, so the workflow logic is exercised without PostgreSQL or
Ollama.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.core.settings import Settings
from semrag.jobs.manager import JobManager
from semrag.jobs.models import JobState
from semrag.reasoning.coding.files import (
    file_exists,
    patch_file,
    read_file,
    safe_resolve_target,
    snapshot_paths,
    write_file,
)
from semrag.reasoning.prompting.citation import format_chunk_with_citation
from semrag.retrieval.models import RetrievalResult
from semrag.storage.postgres.models import FileRow, IRNodeRow
from semrag.validation.config import load_validation_config
from semrag.validation.runner import ValidationResult, run_validation


@runtime_checkable
class CodingToolset(Protocol):
    project_id: str

    def project_info(self) -> str: ...
    def list_files(self) -> str: ...
    def search(self, query: str, top_k: int = 10) -> str: ...
    def search_chunks(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...
    def find_symbol(self, symbol: str, limit: int = 20) -> str: ...
    def read_file(self, file_path: str, start: int | None = None, end: int | None = None) -> str: ...
    def write_file(self, file_path: str, content: str) -> dict: ...
    def apply_patch(self, file_path: str, old_text: str, new_text: str) -> dict: ...
    def run_validation(self, target: str) -> ValidationResult: ...
    def validation_targets(self) -> list[str]: ...
    def index(self) -> dict: ...
    def is_retrievable(self, file_path: str) -> bool: ...
    def file_exists(self, file_path: str) -> bool: ...
    def snapshot_files(self) -> set[str]: ...


_SOURCE_EXTENSIONS = (".sv", ".svh", ".v", ".vh", ".py", ".c", ".h", ".cpp", ".cc", ".hpp")
_NOISE_PATH_FRAGMENTS = (
    "/log/", "/logs/", "/coverage/", "/lint/", "/timing/", "/waveform/",
    "/report", "/reports/", "_log.",
)


def _is_source_chunk(result: RetrievalResult) -> bool:
    """Strict source-only preference for planning evidence. Runtime
    artifacts (sim logs, coverage/lint/timing reports) share vocabulary
    with a code-change request but are NOT planable source, so a coding
    agent must not read them as 'the pattern to follow'."""
    path = (result.relative_path or "").lower()
    if any(path.endswith(ext) for ext in _SOURCE_EXTENSIONS):
        return True
    lang = (result.language or "").lower()
    if lang in ("systemverilog", "verilog", "vhdl", "python", "c", "cpp"):
        return True
    return not any(fragment in path for fragment in _NOISE_PATH_FRAGMENTS)


class InProcessCodingToolset:
    """Production toolset: in-process access to the SemiRAG core."""

    def __init__(
        self,
        project_id: str,
        session: Session,
        project_root: Path,
        retriever,
        embed_query_fn,
        settings: Settings,
        job_manager: JobManager,
        index_poll_timeout: float = 600.0,
    ):
        self.project_id = project_id
        self._session = session
        self._root = project_root
        self._retriever = retriever
        self._embed_query_fn = embed_query_fn
        self._settings = settings
        self._job_manager = job_manager
        self._index_poll_timeout = index_poll_timeout

    def project_info(self) -> str:
        return (
            f"project_id={self.project_id} root={self._root}\n"
            f"files={len(self.list_files().splitlines())}"
        )

    def list_files(self) -> str:
        rows = self._session.execute(
            select(FileRow.relative_path)
            .where(FileRow.project_id == self.project_id)
            .order_by(FileRow.relative_path)
        ).scalars().all()
        return "\n".join(rows)

    def _retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        embedding = self._embed_query_fn(query)
        results = self._retriever.retrieve(
            query,
            embedding,
            project_id=self.project_id,
            top_k=top_k,
        )
        kept = [r for r in results if _is_source_chunk(r)]
        if len(kept) >= 4:
            return kept
        return list(results)

    def search(self, query: str, top_k: int = 10) -> str:
        results = self._retrieve(query, top_k)
        return "\n\n".join(format_chunk_with_citation(r) for r in results)

    def search_chunks(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self._retrieve(query, top_k)

    def find_symbol(self, symbol: str, limit: int = 20) -> str:
        rows = (
            self._session.execute(
                select(IRNodeRow.id, IRNodeRow.kind, IRNodeRow.name, IRNodeRow.relative_path)
                .where(IRNodeRow.project_id == self.project_id)
                .where(IRNodeRow.name == symbol)
                .order_by(IRNodeRow.relative_path)
                .limit(limit)
            ).all()
        )
        if not rows:
            return f"No indexed symbol named '{symbol}'."
        return "\n".join(
            f"- {r.name} ({r.kind}) at {r.relative_path}" for r in rows
        )

    def read_file(self, file_path: str, start: int | None = None, end: int | None = None) -> str:
        return read_file(self._root, file_path, start, end)

    def write_file(self, file_path: str, content: str) -> dict:
        return write_file(self._root, file_path, content)

    def apply_patch(self, file_path: str, old_text: str, new_text: str) -> dict:
        return patch_file(self._root, file_path, old_text, new_text)

    def run_validation(self, target: str) -> ValidationResult:
        config = load_validation_config(self.project_id)
        if config is None:
            return ValidationResult(
                status="blocked",
                exit_code=None,
                stdout="",
                stderr="",
                command="",
                duration_seconds=0.0,
                blocked_reason=f"no validation config registered for project '{self.project_id}'",
                target=target,
            )
        return run_validation(self._root, config, target)

    def validation_targets(self) -> list[str]:
        config = load_validation_config(self.project_id)
        if config is None or not getattr(config, "targets", None):
            return ["rtl_lint"]
        return sorted(config.targets.keys())

    def index(self) -> dict:
        job, _created = self._job_manager.create_job(self.project_id, str(self._root))
        job_id = job.job_id
        deadline = time.monotonic() + self._index_poll_timeout
        while time.monotonic() < deadline:
            job = self._job_manager.get_job(job_id)
            if job is None:
                return {"job_id": job_id, "status": "unknown", "error": "job not found"}
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return {
                    "job_id": job_id,
                    "status": job.state.name.lower(),
                    "progress": job.progress,
                    "stage": job.current_stage,
                }
            time.sleep(0.25)
        return {"job_id": job_id, "status": "timed_out"}

    def is_retrievable(self, file_path: str) -> bool:
        try:
            safe_resolve_target(self._root, file_path)
        except Exception:
            return False
        indexed = {
            line.strip()
            for line in self.list_files().splitlines()
            if line.strip()
        }
        return file_path in indexed

    def file_exists(self, file_path: str) -> bool:
        return file_exists(self._root, file_path)

    def snapshot_files(self) -> set[str]:
        return snapshot_paths(self._root)