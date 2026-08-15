from __future__ import annotations

import time
from collections import defaultdict

from semrag.evaluation.metrics import (
    average_rank,
    file_recall,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semrag.evaluation.models import (
    AggregateReport,
    CoverageReport,
    EvalQueryResult,
    GoldenQuery,
    RetrievalMetrics,
)
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.retrieval.pipeline import HybridRetriever
from semrag.storage.vector.models import VectorSearchFilter


class EvaluationHarness:
    def __init__(
        self,
        retriever: HybridRetriever,
        embed_query_fn,
        k: int = 10,
        use_prompt_builder: bool = True,
    ):
        self._retriever = retriever
        self._embed_query_fn = embed_query_fn
        self._k = k
        self._prompt_builder = PromptBuilder() if use_prompt_builder else None

    def evaluate_query(self, golden_query: GoldenQuery) -> EvalQueryResult:
        query_embedding = self._embed_query_fn(golden_query.query_text)

        start = time.perf_counter()
        results = self._retriever.retrieve(
            golden_query.query_text,
            query_embedding,
            project_id=golden_query.project_id,
            top_k=self._k,
            filters=VectorSearchFilter(project_id=golden_query.project_id),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [r.chunk_id for r in results]
        retrieved_paths = [r.relative_path for r in results]
        relevant = golden_query.relevant_chunk_ids
        retrieved_set = set(retrieved_paths)

        primary_found = list(golden_query.primary_files & retrieved_set)
        secondary_found = list(golden_query.secondary_files & retrieved_set)
        primary_missed = list(golden_query.primary_files - retrieved_set)
        secondary_missed = list(golden_query.secondary_files - retrieved_set)

        metrics = RetrievalMetrics(
            precision_at_k=precision_at_k(retrieved_ids, relevant, self._k),
            recall_at_k=recall_at_k(retrieved_ids, relevant, self._k),
            reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
            ndcg_at_k=ndcg_at_k(retrieved_ids, relevant, self._k),
            k=self._k,
            primary_file_recall=file_recall(retrieved_set, golden_query.primary_files),
            secondary_file_recall=file_recall(retrieved_set, golden_query.secondary_files),
            hit_rate=hit_rate(retrieved_ids, relevant),
            average_rank=average_rank(retrieved_ids, relevant),
            primary_files_found=primary_found,
            secondary_files_found=secondary_found,
            primary_files_missed=primary_missed,
            secondary_files_missed=secondary_missed,
        )

        excluded = 0
        total_tokens = 0
        if self._prompt_builder:
            prompt = self._prompt_builder.build(golden_query.query_text, results)
            excluded = prompt.excluded_chunk_count
            total_tokens = prompt.estimated_total_tokens

        return EvalQueryResult(
            query_id=golden_query.query_id,
            query_text=golden_query.query_text,
            metrics=metrics,
            latency_ms=latency_ms,
            retrieved_paths=retrieved_paths,
            estimated_total_tokens=total_tokens,
            excluded_chunk_count=excluded,
        )

    def evaluate_all(self, golden_queries: list[GoldenQuery]) -> AggregateReport:
        per_query = [self.evaluate_query(q) for q in golden_queries]

        if not per_query:
            return AggregateReport(
                query_count=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_mrr=0.0,
                mean_ndcg_at_k=0.0,
                mean_hit_rate=0.0,
                mean_average_rank=0.0,
                mean_primary_recall=0.0,
                mean_secondary_recall=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
            )

        n = len(per_query)
        mean_precision = sum(r.metrics.precision_at_k for r in per_query) / n
        mean_recall = sum(r.metrics.recall_at_k for r in per_query) / n
        mean_mrr = sum(r.metrics.reciprocal_rank for r in per_query) / n
        mean_ndcg = sum(r.metrics.ndcg_at_k for r in per_query) / n
        mean_hit_rate = sum(r.metrics.hit_rate for r in per_query) / n
        mean_avg_rank = sum(
            r.metrics.average_rank for r in per_query if r.metrics.average_rank != float("inf")
        )
        finite_ranks = sum(1 for r in per_query if r.metrics.average_rank != float("inf"))
        mean_avg_rank = mean_avg_rank / finite_ranks if finite_ranks > 0 else 0.0
        mean_primary_recall = sum(r.metrics.primary_file_recall for r in per_query) / n
        mean_secondary_recall = sum(r.metrics.secondary_file_recall for r in per_query) / n

        latencies = sorted(r.latency_ms for r in per_query)
        p50 = latencies[len(latencies) // 2]
        p95_index = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p95 = latencies[p95_index]

        coverage = self._build_coverage_report(per_query, golden_queries)

        return AggregateReport(
            query_count=n,
            mean_precision_at_k=mean_precision,
            mean_recall_at_k=mean_recall,
            mean_mrr=mean_mrr,
            mean_ndcg_at_k=mean_ndcg,
            mean_hit_rate=mean_hit_rate,
            mean_average_rank=mean_avg_rank,
            mean_primary_recall=mean_primary_recall,
            mean_secondary_recall=mean_secondary_recall,
            coverage=coverage,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            per_query_results=per_query,
        )

    def _build_coverage_report(
        self,
        results: list[EvalQueryResult],
        queries: list[GoldenQuery],
    ) -> CoverageReport:
        q_map = {q.query_id: q for q in queries}
        total_primary = 0
        total_secondary = 0
        primary_found = 0
        secondary_found = 0

        area_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"primary": 0, "primary_found": 0, "secondary": 0, "secondary_found": 0}
        )
        intent_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"primary": 0, "primary_found": 0, "secondary": 0, "secondary_found": 0}
        )

        for r in results:
            q = q_map.get(r.query_id)
            if q is None:
                continue
            total_primary += len(q.primary_files)
            total_secondary += len(q.secondary_files)
            primary_found += len(r.metrics.primary_files_found)
            secondary_found += len(r.metrics.secondary_files_found)

            area = q.repo_area.value if q.repo_area else "unknown"
            intent = q.expected_intent.value if q.expected_intent else "unknown"

            area_totals[area]["primary"] += len(q.primary_files)
            area_totals[area]["primary_found"] += len(r.metrics.primary_files_found)
            area_totals[area]["secondary"] += len(q.secondary_files)
            area_totals[area]["secondary_found"] += len(r.metrics.secondary_files_found)

            intent_totals[intent]["primary"] += len(q.primary_files)
            intent_totals[intent]["primary_found"] += len(r.metrics.primary_files_found)
            intent_totals[intent]["secondary"] += len(q.secondary_files)
            intent_totals[intent]["secondary_found"] += len(r.metrics.secondary_files_found)

        def _ratio(found: int, total: int) -> float:
            return found / total if total else 1.0

        by_area = {
            area: {
                "primary_recall": _ratio(v["primary_found"], v["primary"]),
                "secondary_recall": _ratio(v["secondary_found"], v["secondary"]),
                "primary_files": v["primary"],
                "secondary_files": v["secondary"],
                "primary_found": v["primary_found"],
                "secondary_found": v["secondary_found"],
            }
            for area, v in sorted(area_totals.items())
        }
        by_intent = {
            intent: {
                "primary_recall": _ratio(v["primary_found"], v["primary"]),
                "secondary_recall": _ratio(v["secondary_found"], v["secondary"]),
                "primary_files": v["primary"],
                "secondary_files": v["secondary"],
                "primary_found": v["primary_found"],
                "secondary_found": v["secondary_found"],
            }
            for intent, v in sorted(intent_totals.items())
        }

        return CoverageReport(
            total_queries=len(results),
            total_primary=total_primary,
            total_secondary=total_secondary,
            primary_found=primary_found,
            secondary_found=secondary_found,
            primary_recall=_ratio(primary_found, total_primary),
            secondary_recall=_ratio(secondary_found, total_secondary),
            by_area=by_area,
            by_intent=by_intent,
        )
