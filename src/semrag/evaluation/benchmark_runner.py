"""
Retrieval benchmark runner.

Runs a set of ``GoldenQuery`` items through a ``HybridRetriever`` and
produces a reproducible retrieval benchmark: a machine-readable JSON
report (``retrieval_benchmark.json``) plus a human-readable summary.

Relationship to ``EvaluationHarness``: the harness measures retrieval
quality against chunk/file ground truth at a single k (precision@k,
recall@k, NDCG@k, hit rate, file recall). This runner produces the
profile a retrieval *improvement* is judged against — top-1/3/5 recall,
MRR, average rank, and the chunk-category composition of the top-k
(logs / RTL / spec docs) — and persists it so any future change to
ranking, source scoring, intent weighting, or diversity can be measured
against a recorded baseline.

Evaluation-only: read-only against the index, does not call the LLM,
does not touch retrieval, ranking, embedding, or chunking code.

Usage::

    runner = RetrievalBenchmark(retriever, embed_query_fn=lambda q: provider.embed_texts([q])[0])
    report = runner.run(get_benchmarks())
    write_benchmark_reports(report, "retrieval_benchmark.json", "retrieval_benchmark.txt")
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from semrag.evaluation.metrics import average_rank, recall_at_k, reciprocal_rank
from semrag.evaluation.models import (
    ChunkCategory,
    GoldenQuery,
    RetrievalBenchmarkQueryResult,
    RetrievalBenchmarkReport,
)
from semrag.retrieval.models import RetrievalResult
from semrag.storage.vector.models import VectorSearchFilter

DEFAULT_BENCHMARK_TOP_K = 10

# Fixed recall cutoffs the benchmark reports regardless of retrieval top-k.
RECALL_CUTOFFS = (1, 3, 5)

# Chunk-category classification hints (deterministic heuristics; see
# ``categorize_chunk``). Chunk types come from ``ingestion/chunking``:
# RTL material is chunked as ``sv_*`` / ``v_*``, markdown docs as
# ``markdown_section`` / ``markdown_preamble``, and everything else
# (logs included) falls back to ``generic_window`` — which is why the
# path hints matter: a ``generic_window`` chunk from ``sim/log/run.log``
# is a log, even though its chunk_type says nothing about it.
_RTL_CHUNK_TYPE_PREFIXES = ("sv_", "v_")
_SPEC_CHUNK_TYPES = frozenset({"markdown_section", "markdown_preamble"})
_LOG_PATH_SEGMENTS = frozenset({"log", "logs"})
_RTL_PATH_SEGMENTS = frozenset({"rtl", "design"})
_SPEC_PATH_SEGMENTS = frozenset({"docs", "doc", "spec", "specs"})
_LOG_SUFFIXES = (".log",)
_RTL_SUFFIXES = (".sv", ".v")


def _path_segments(path: str) -> set[str]:
    return {seg.lower() for seg in path.replace("\\", "/").split("/") if seg}


def categorize_chunk(result: RetrievalResult) -> ChunkCategory | None:
    """Classify a chunk's source category for benchmark reporting.

    Deterministic heuristic, chunk_type first then path hints. Precedence
    is LOG > RTL > SPEC so a chunk that trips several hints counts once.
    Returns ``None`` for chunks outside these categories (Python code,
    configs, scripts, ...).
    """
    path = result.relative_path or ""
    ct = result.chunk_type or ""

    if ct == "log" or _path_segments(path) & _LOG_PATH_SEGMENTS or path.endswith(_LOG_SUFFIXES):
        return ChunkCategory.LOG
    if ct.startswith(_RTL_CHUNK_TYPE_PREFIXES) or _path_segments(path) & _RTL_PATH_SEGMENTS or path.endswith(_RTL_SUFFIXES):
        return ChunkCategory.RTL
    if ct in _SPEC_CHUNK_TYPES or _path_segments(path) & _SPEC_PATH_SEGMENTS:
        return ChunkCategory.SPEC
    return None


def count_categories(results: list[RetrievalResult]) -> dict[ChunkCategory, int]:
    """Count how many chunks of each category appear in ``results``."""
    counts: dict[ChunkCategory, int] = {}
    for result in results:
        category = categorize_chunk(result)
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    return counts


def to_json_dict(report: RetrievalBenchmarkReport) -> dict:
    """JSON-safe dict of the report. Non-finite floats (e.g. the
    ``average_rank = inf`` sentinel for a query with no relevant hit)
    become ``null`` so the machine-readable report stays strict JSON."""
    return _json_safe(report.model_dump())


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def format_human_summary(report: RetrievalBenchmarkReport) -> str:
    """Human-readable one-block summary of the benchmark run."""

    def _pct(x: float) -> str:
        return f"{round(x * 100)}%"

    rows = [
        ("Retrieval Benchmark", None),
        ("Queries", str(report.query_count)),
        ("Top-K", str(report.top_k)),
        ("Top1 Recall", _pct(report.top1_recall)),
        ("Top3 Recall", _pct(report.top3_recall)),
        ("Top5 Recall", _pct(report.top5_recall)),
        ("MRR", f"{report.mean_mrr:.2f}"),
        ("Average Rank", f"{report.mean_average_rank:.1f}"),
        ("Average Logs", f"{report.mean_log_count:.1f}"),
        ("Average RTL", f"{report.mean_rtl_count:.1f}"),
        ("Average Specs", f"{report.mean_spec_count:.1f}"),
    ]
    lines: list[str] = []
    for label, value in rows:
        if value is None:
            lines.append(label)
            lines.append("-" * len(label))
        else:
            dots = max(0, 19 - len(label))
            lines.append(f"{label} {'.' * dots} {value}")
    return "\n".join(lines)


def write_benchmark_reports(
    report: RetrievalBenchmarkReport,
    json_path: Path | str,
    summary_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Persist the machine-readable JSON report and (optionally) the
    human-readable summary. Returns the paths written."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_json_dict(report), indent=2) + "\n")

    written_summary: Path | None = None
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(format_human_summary(report) + "\n")
        written_summary = summary_path
    return json_path, written_summary


class RetrievalBenchmark:
    """Runs golden queries through a retriever and aggregates the
    retrieval benchmark metrics. Mirrors the ``EvaluationHarness``
    wiring (retriever + ``embed_query_fn`` callable) but is retrieval-
    only and reports the multi-k recall profile + top-k composition."""

    def __init__(
        self,
        retriever,
        embed_query_fn,
        top_k: int = DEFAULT_BENCHMARK_TOP_K,
    ):
        self._retriever = retriever
        self._embed_query_fn = embed_query_fn
        self._top_k = top_k

    def run(self, golden_queries: list[GoldenQuery]) -> RetrievalBenchmarkReport:
        per_query = [self._evaluate_query(q) for q in golden_queries]
        n = len(per_query)
        if n == 0:
            return RetrievalBenchmarkReport(query_count=0, top_k=self._top_k)

        finite_ranks = [r.average_rank for r in per_query if r.average_rank != float("inf")]
        mean_avg_rank = sum(finite_ranks) / len(finite_ranks) if finite_ranks else 0.0

        return RetrievalBenchmarkReport(
            query_count=n,
            top_k=self._top_k,
            top1_recall=_mean(r.top1_recall for r in per_query),
            top3_recall=_mean(r.top3_recall for r in per_query),
            top5_recall=_mean(r.top5_recall for r in per_query),
            mean_mrr=_mean(r.mrr for r in per_query),
            mean_average_rank=mean_avg_rank,
            mean_log_count=_mean(r.log_count for r in per_query),
            mean_rtl_count=_mean(r.rtl_count for r in per_query),
            mean_spec_count=_mean(r.spec_count for r in per_query),
            per_query_results=per_query,
        )

    # ------------------------------------------------------------------

    def _evaluate_query(self, golden_query: GoldenQuery) -> RetrievalBenchmarkQueryResult:
        query_embedding = self._embed_query_fn(golden_query.query_text)
        results = self._retriever.retrieve(
            golden_query.query_text,
            query_embedding,
            project_id=golden_query.project_id,
            top_k=self._top_k,
            filters=VectorSearchFilter(project_id=golden_query.project_id),
        )

        retrieved_ids = [r.chunk_id for r in results]
        relevant = golden_query.relevant_chunk_ids
        counts = count_categories(results)

        return RetrievalBenchmarkQueryResult(
            query_id=golden_query.query_id,
            top_k=self._top_k,
            top1_recall=recall_at_k(retrieved_ids, relevant, 1),
            top3_recall=recall_at_k(retrieved_ids, relevant, 3),
            top5_recall=recall_at_k(retrieved_ids, relevant, 5),
            mrr=reciprocal_rank(retrieved_ids, relevant),
            average_rank=average_rank(retrieved_ids, relevant),
            log_count=counts.get(ChunkCategory.LOG, 0),
            rtl_count=counts.get(ChunkCategory.RTL, 0),
            spec_count=counts.get(ChunkCategory.SPEC, 0),
            retrieved_paths=[r.relative_path for r in results],
        )


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
