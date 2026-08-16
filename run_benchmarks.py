#!/usr/bin/env python3
"""
Retrieval benchmark runner.

Usage:
    python run_benchmarks.py                          # all benchmarks
    python run_benchmarks.py --area retrieval         # by repo area
    python run_benchmarks.py --intent verification    # by intent type
    python run_benchmarks.py --verbose                 # per-query detail
    python run_benchmarks.py --json                    # JSON output

Requires a running Postgres with the project ingested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from semrag.core.settings import get_settings
from semrag.evaluation.benchmarks import get_benchmarks, get_benchmarks_by_area, get_benchmarks_by_intent
from semrag.evaluation.harness import EvaluationHarness
from semrag.evaluation.models import AggregateReport, EvalQueryResult, ExpectedIntent, RepoArea
from semrag.retrieval.pipeline import HybridRetriever
from semrag.retrieval.reranking import LexicalOverlapReranker
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.ingestion.embedding.factory import embedding_provider_from_settings


def _format_markdown(report: AggregateReport, verbose: bool) -> str:
    lines = [
        "# Retrieval Benchmark Report",
        "",
        f"**Queries:** {report.query_count}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean Precision@{report.per_query_results[0].metrics.k if report.per_query_results else 10} | {report.mean_precision_at_k:.4f} |",
        f"| Mean Recall@{report.per_query_results[0].metrics.k if report.per_query_results else 10} | {report.mean_recall_at_k:.4f} |",
        f"| Mean Reciprocal Rank (MRR) | {report.mean_mrr:.4f} |",
        f"| Mean NDCG | {report.mean_ndcg_at_k:.4f} |",
        f"| Mean Hit Rate | {report.mean_hit_rate:.4f} |",
        f"| Mean Average Rank | {report.mean_average_rank:.1f} |",
        f"| Mean Primary File Recall | {report.mean_primary_recall:.4f} |",
        f"| Mean Secondary File Recall | {report.mean_secondary_recall:.4f} |",
        f"| Latency P50 | {report.latency_p50_ms:.1f} ms |",
        f"| Latency P95 | {report.latency_p95_ms:.1f} ms |",
        "",
        "## Coverage by Repo Area",
        "",
        "| Area | Primary Recall | Secondary Recall | Files |",
        "|------|---------------|-----------------|-------|",
    ]
    for area, cov in sorted(report.coverage.by_area.items()):
        lines.append(
            f"| {area} | {cov['primary_recall']:.2%} | {cov['secondary_recall']:.2%} | "
            f"{cov['primary_found']}/{cov['primary_files']} primary, {cov['secondary_found']}/{cov['secondary_files']} secondary |"
        )

    lines.extend([
        "",
        "## Coverage by Intent",
        "",
        "| Intent | Primary Recall | Secondary Recall | Files |",
        "|--------|---------------|-----------------|-------|",
    ])
    for intent, cov in sorted(report.coverage.by_intent.items()):
        lines.append(
            f"| {intent} | {cov['primary_recall']:.2%} | {cov['secondary_recall']:.2%} | "
            f"{cov['primary_found']}/{cov['primary_files']} primary, {cov['secondary_found']}/{cov['secondary_files']} secondary |"
        )

    if verbose and report.per_query_results:
        lines.extend(["", "## Per-Query Results", ""])
        for r in report.per_query_results:
            m = r.metrics
            lines.extend([
                f"### {r.query_id}: {r.query_text}",
                "",
                f"- Precision@{m.k}: {m.precision_at_k:.4f}",
                f"- Recall@{m.k}: {m.recall_at_k:.4f}",
                f"- MRR: {m.reciprocal_rank:.4f}",
                f"- NDCG@{m.k}: {m.ndcg_at_k:.4f}",
                f"- Primary recall: {m.primary_file_recall:.2%}",
                f"- Secondary recall: {m.secondary_file_recall:.2%}",
                f"- Latency: {r.latency_ms:.1f} ms",
                "",
            ])
            if m.primary_files_found:
                lines.append(f"  Primary files found: {', '.join(m.primary_files_found)}")
            if m.primary_files_missed:
                lines.append(f"  Primary files MISSING: {', '.join(m.primary_files_missed)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmarks")
    parser.add_argument("--area", type=str, help="Filter by repo area")
    parser.add_argument("--intent", type=str, help="Filter by intent type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-query detail")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--k", type=int, default=10, help="Top-k for evaluation")
    args = parser.parse_args()

    settings = get_settings()
    if args.area:
        area = RepoArea(args.area)
        benchmarks = get_benchmarks_by_area(area)
    elif args.intent:
        intent = ExpectedIntent(args.intent)
        benchmarks = get_benchmarks_by_intent(intent)
    else:
        benchmarks = get_benchmarks()

    if not benchmarks:
        print("No benchmarks match the filter.")
        sys.exit(1)

    print(f"Running {len(benchmarks)} benchmark queries...")

    engine = create_engine(settings.database_url)
    with Session(engine) as session:
        embed_provider = embedding_provider_from_settings(settings)
        retriever = HybridRetriever(
            session,
            reranker=LexicalOverlapReranker() if settings.reranker == "lexical_overlap" else None,
            enable_graph_expansion=settings.enable_graph_expansion,
            enable_intent_boost=settings.enable_intent_boost,
        )
        harness = EvaluationHarness(
            retriever=retriever,
            embed_query_fn=lambda q: embed_provider.embed_texts([q])[0],
            k=args.k,
            use_prompt_builder=False,
        )
        report = harness.evaluate_all(benchmarks)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(_format_markdown(report, verbose=args.verbose))


if __name__ == "__main__":
    main()
