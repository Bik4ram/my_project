"""
Debug artifact writer.

Layout::

    <debug_artifacts_dir>/
        <run-id>/
            scan/manifest.json
            parse/<relative-path>.parse.json
            ir/graph.json
            chunks/chunk_metadata.json
            embeddings/embedding_stats.json
            knowledge_graph/graph_summary.json
            retrieval/retrieval_trace.json
            prompt/final_prompt.txt
            llm/request.json
            llm/response.json
            timings/pipeline.json

The writer is **only active when `Settings.log_level` is DEBUG or TRACE**.
At INFO+ (the default for production) every method becomes a no-op so the
hot path costs nothing.

``write_query_trace()`` supports optional gzip compression via the
``compress`` parameter. When enabled the file is written as
``query_trace.json.gz`` — useful for full traces that include embedding
vectors that can be 50+ KB each.

Retention: after each successful ``run_completed()`` the writer prunes older
runs in the same parent directory down to ``debug_artifact_retention``.
Retention count ``0`` disables pruning (operator-managed).

A run id is supplied by the caller (``IngestionService.index()`` /
``QueryService.ask()``); the writer does not invent one. The id should be a
short stable string (timestamp+random or just timestamp) because it lives
on disk across process invocations.
"""
from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

from semrag.core.observability import is_debug_enabled


def _is_active(level: str) -> bool:
    return is_debug_enabled(level)


def new_run_id() -> str:
    """Run id is a millisecond timestamp + short random suffix, e.g.
    `1700000000000-a3b7`. Stable across log records in the same run,
    sortable for retention."""
    import secrets

    return f"{int(time.time() * 1000)}-{secrets.token_hex(3)}"


class DebugArtifactWriter:
    def __init__(self, root_dir: str | Path, log_level: str, retention: int):
        self._root = Path(root_dir)
        self._log_level = log_level
        self._retention = retention
        self._run_dir: Path | None = None

    @property
    def is_active(self) -> bool:
        return _is_active(self._log_level)

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    def start_run(self, run_id: str) -> None:
        """Allocate the per-run directory tree. Idempotent: re-calling
        with the same id is a no-op."""
        if not self.is_active:
            return
        self._run_dir = self._root / run_id
        for sub in (
            "scan", "parse", "ir", "chunks", "embeddings",
            "knowledge_graph", "retrieval", "prompt", "llm", "timings",
            "query_traces",
        ):
            (self._run_dir / sub).mkdir(parents=True, exist_ok=True)

    def write_json(self, category: str, name: str, payload: Any) -> None:
        """Generic JSON writer. `category` is one of the subdirs above;
        `name` is the file basename (extension added)."""
        if not self.is_active or self._run_dir is None:
            return
        path = self._run_dir / category / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_safe_json(payload), encoding="utf-8")

    def write_text(self, category: str, name: str, content: str) -> None:
        if not self.is_active or self._run_dir is None:
            return
        path = self._run_dir / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_parse(self, rel_path: str, payload: Any) -> None:
        # Flatten relative path separators so `src/foo.py` becomes
        # `src_foo.py.parse.json` — one file per source file. We pass
        # `safe + ".parse"` so `write_json`'s `.json` suffix produces the
        # correct compound extension.
        safe = rel_path.replace("/", "_").replace("\\", "_")
        self.write_json("parse", f"{safe}.parse", payload)

    def write_query_trace(self, payload: Any, compress: bool = False) -> None:
        """Write a full query trace (``QueryTrace``) under
        ``query_traces/``. When ``compress=True`` the file is written as
        ``query_trace.json.gz`` to reduce disk usage on full traces that
        include embedding vectors."""
        if not self.is_active or self._run_dir is None:
            return
        path = self._run_dir / "query_traces" / ("query_trace.json.gz" if compress else "query_trace.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = _safe_json(payload)
        if compress:
            path.write_bytes(gzip.compress(serialized.encode("utf-8")))
        else:
            path.write_text(serialized, encoding="utf-8")

    def run_completed(self) -> None:
        """Prune older runs after a successful run. `retention=0` skips
        pruning entirely."""
        if not self.is_active or self._retention <= 0:
            return
        if not self._root.exists():
            return
        runs = sorted(
            (p for p in self._root.iterdir() if p.is_dir()),
            key=lambda p: p.name,  # run ids are sortable timestamps
        )
        for stale in runs[: -self._retention]:
            shutil.rmtree(stale, ignore_errors=True)


def _safe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        # Fall back to repr for non-serializable payloads (e.g. dataclass
        # instances we forgot to convert). Better than crashing the run.
        return json.dumps({"repr": repr(payload)}, ensure_ascii=False, indent=2)
