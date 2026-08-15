"""
Retrieval analytics dashboard.

Transforms the outputs of the RB-01 retrieval benchmark and the RB-02
retrieval failure analyzer into engineering analytics: overall retrieval
health, failure distribution, per-stage responsibility attribution,
document-category and query-intent breakdowns, and a deterministic
recommendation engine.

This module is purely analytical. It does not modify retrieval, ranking,
chunking, embedding, reranking, or prompt-assembly code, and it does not
recompute ranking or trace internals — it consumes the recorded
``RetrievalBenchmarkReport`` and ``RetrievalFailureReport`` structures
verbatim.

The analytics answer:

1. Where are retrieval failures happening?    (stage attribution)
2. Which ranking stage is responsible most?   (stage attribution)
3. Which document categories dominate?        (document analysis)
4. Which query intents perform worst?         (intent analysis)
5. What should engineers improve next?        (recommendation engine)

Outputs: ``write_analytics_reports`` persists ``retrieval_analytics.json``
(machine-readable) plus ``retrieval_analytics.md`` (human).

Deterministic rules
-------------------
- Failure distribution: reuses RB-02's aggregate counts verbatim.
- Stage attribution: maps each RB-02 ``FailureCategory`` to one
  responsibility bucket, splitting the two base-retrieval categories
  using the recorded ``StageContributions`` (which signal surfaced the
  chunk) when the trace allows it.
- Document categories: a heuristic over ``chunk_type`` +
  ``relative_path`` (RB-01's convention, finer taxonomy). Failures with
  no metadata (NOT_RETRIEVED chunks never enter the trace) are counted
  as unclassified.
- Query intents: deterministic keyword rules over the query text.
- Recommendations: fixed rule set with a single dominance threshold;
  all triggered rules fire in a fixed order. Diagnostic-only — the
  recommendation text always says "investigate", never suggests code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from semrag.evaluation.models import (
    Bucket,
    DocumentAnalysis,
    DocumentCategory,
    ExpectedChunkAnalysis,
    FailureCategory,
    GoldenQuery,
    IntentAnalysis,
    IntentCategory,
    Recommendation,
    RetrievalAnalyticsReport,
    RetrievalBenchmarkQueryResult,
    RetrievalBenchmarkReport,
    RetrievalFailureReport,
)

# Default dominance threshold for the recommendation engine: a failure
# type or document category "dominates" when it accounts for more than
# this share of all failures (the task's log rule uses > 40%).
DEFAULT_DOMINANCE_THRESHOLD = 0.40

# Responsibility buckets for stage attribution, in report order.
_STAGE_RESPONSIBILITY_ORDER = (
    "vector",
    "lexical",
    "graph",
    "intent",
    "source",
    "reranker",
    "diversity",
    "prompt_cutoff",
    "base_fusion",
    "base_retrieval",
    "unknown",
)

# Document-category heuristics (RB-01 path/type conventions, finer
# taxonomy). See ``categorize_document`` for precedence.
_RTL_CHUNK_TYPE_PREFIXES = ("sv_", "v_")
_MARKDOWN_CHUNK_TYPES = frozenset({"markdown_section", "markdown_preamble"})
_LOG_PATH_SEGMENTS = frozenset({"log", "logs"})
_RTL_PATH_SEGMENTS = frozenset({"rtl", "design"})
_SPEC_PATH_SEGMENTS = frozenset({"docs", "doc", "spec", "specs"})
_UVM_PATH_SEGMENTS = frozenset({"uvm", "testbench", "testbenches", "verif", "verification", "tb"})
_GENERATED_PATH_SEGMENTS = frozenset({"generated", "gen", "build"})
_LOG_SUFFIXES = (".log",)
_RTL_SUFFIXES = (".sv", ".v")

# Query-intent keyword rules, in priority order (first match wins). The
# taxonomy is a user-question type, distinct from retrieval intent
# boosting. Rules are substring checks against the lowercased query.
_INTENT_RULES: list[tuple[IntentCategory, tuple[str, ...]]] = [
    (IntentCategory.DEBUG, (
        "why", "error", "fail", "failed", "stuck", "timeout", "time out",
        "debug", "bug", "assert", "uvm_error", "exception", "crash",
        "deadlock", "mismatch", "overrun", "stall", "not working", "incorrect",
    )),
    (IntentCategory.SPECIFICATION, (
        "spec", "specification", "requirement", "protocol", "pinout",
        "datasheet", "register map", "standard", "axi", "apb", "ahb", "pcie",
    )),
    (IntentCategory.NAVIGATION, (
        "where is", "where are", "which file", "which module", "which function",
        "which component", "who uses", "who instantiates", "who calls", "find",
        "locate", "list", "navigate", "search for", "what file",
    )),
    (IntentCategory.EXPLAIN, (
        "what is", "what are", "what does", "whats", "how does", "how do",
        "how is", "explain", "describe", "overview", "purpose", "meaning",
        "define", "tell me about",
    )),
    (IntentCategory.GENERAL, (
        "summary", "summarize", "general", "high level", "introduction", "intro",
    )),
]


def classify_query_intent(query_text: str) -> IntentCategory:
    """Classify a query into the analytics intent taxonomy using the
    deterministic keyword rules in ``_INTENT_RULES`` (first match wins,
    priority order). Returns ``UNKNOWN`` when no rule matches."""
    text = query_text.lower()
    for intent, keywords in _INTENT_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return IntentCategory.UNKNOWN


def categorize_document(chunk_type: str, relative_path: str) -> DocumentCategory:
    """Classify a failed chunk's source document category from its
    ``chunk_type`` and ``relative_path``. Deterministic heuristic;
    precedence is Logs > Generated > UVM > RTL > Spec/Markdown > Other,
    so a chunk that trips several hints counts once (RB-01's convention).
    ``OTHER`` is the catch-all (Python, configs, scripts, unknown)."""
    path = relative_path or ""
    ct = chunk_type or ""
    segments = _path_segments(path)

    if ct == "log" or segments & _LOG_PATH_SEGMENTS or path.endswith(_LOG_SUFFIXES):
        return DocumentCategory.LOGS
    if segments & _GENERATED_PATH_SEGMENTS:
        return DocumentCategory.GENERATED
    if segments & _UVM_PATH_SEGMENTS:
        return DocumentCategory.UVM
    if (
        ct.startswith(_RTL_CHUNK_TYPE_PREFIXES)
        or segments & _RTL_PATH_SEGMENTS
        or path.endswith(_RTL_SUFFIXES)
    ):
        return DocumentCategory.RTL
    if ct in _MARKDOWN_CHUNK_TYPES:
        if segments & _SPEC_PATH_SEGMENTS:
            return DocumentCategory.SPECIFICATION
        return DocumentCategory.MARKDOWN
    return DocumentCategory.OTHER


def attribute_responsibility(failure: ExpectedChunkAnalysis) -> str:
    """Attribute a failed chunk to the retrieval signal / ranking stage
    responsible, from RB-02's failure category plus the recorded
    ``StageContributions``. The two base-retrieval categories are split
    using which signal surfaced the chunk: vector-only -> ``vector``,
    lexical-only -> ``lexical``, both -> ``base_fusion``, graph-only ->
    ``graph``, and never-surfaced (NOT_RETRIEVED) -> ``base_retrieval``."""
    category = failure.failure_category
    if category is FailureCategory.PROMPT_CUTOFF:
        return "prompt_cutoff"
    if category is FailureCategory.SOURCE_PRIOR:
        return "source"
    if category is FailureCategory.GRAPH_EXPANSION:
        return "graph"
    if category is FailureCategory.INTENT:
        return "intent"
    if category is FailureCategory.RERANKER:
        return "reranker"
    if category is FailureCategory.DIVERSITY:
        return "diversity"
    if category is FailureCategory.NOT_RETRIEVED:
        return "base_retrieval"
    if category is FailureCategory.RANKED_TOO_LOW:
        contributions = failure.contributions
        vector = contributions.vector_rank is not None
        lexical = contributions.lexical_rank is not None
        if vector and not lexical:
            return "vector"
        if lexical and not vector:
            return "lexical"
        if contributions.graph_expansion_score is not None:
            return "graph"
        return "base_fusion"
    return "unknown"


class RetrievalAnalytics:
    """Combines a ``RetrievalBenchmarkReport`` and a
    ``RetrievalFailureReport`` into a ``RetrievalAnalyticsReport``. Pure
    analytics — no retrieval, ranking, or production code is touched."""

    def __init__(self, dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD):
        self._threshold = dominance_threshold

    def analyze(
        self,
        benchmark_report: RetrievalBenchmarkReport,
        failure_report: RetrievalFailureReport,
        golden_queries: list[GoldenQuery] | None = None,
    ) -> RetrievalAnalyticsReport:
        total_failures = failure_report.aggregate.total_failures
        document_analysis, document_unclassified = self._document_analysis(failure_report)

        return RetrievalAnalyticsReport(
            query_count=benchmark_report.query_count,
            expected_chunk_count=failure_report.expected_chunk_count,
            top1_recall=benchmark_report.top1_recall,
            top3_recall=benchmark_report.top3_recall,
            top5_recall=benchmark_report.top5_recall,
            mrr=benchmark_report.mean_mrr,
            average_rank=benchmark_report.mean_average_rank,
            total_failures=total_failures,
            failure_distribution=self._failure_distribution(failure_report),
            stage_attribution=self._stage_attribution(failure_report),
            document_analysis=document_analysis,
            document_unclassified=document_unclassified,
            intent_analysis=self._intent_analysis(benchmark_report, failure_report, golden_queries),
            recommendations=self._recommendations(
                benchmark_report,
                failure_report,
                document_analysis,
                golden_queries,
            ),
        )

    # ------------------------------------------------------------------

    def _failure_distribution(self, failure_report: RetrievalFailureReport) -> list[Bucket]:
        total = failure_report.aggregate.total_failures
        counts = failure_report.aggregate.counts
        return [
            Bucket(
                label=category.value,
                count=counts.get(category.value, 0),
                percentage=_share(counts.get(category.value, 0), total),
            )
            for category in FailureCategory
        ]

    def _stage_attribution(self, failure_report: RetrievalFailureReport) -> list[Bucket]:
        total = failure_report.aggregate.total_failures
        buckets: dict[str, int] = {}
        for query in failure_report.per_query:
            for failure in query.failures:
                if failure.failure_category is None:
                    continue
                responsibility = attribute_responsibility(failure)
                buckets[responsibility] = buckets.get(responsibility, 0) + 1
        return [
            Bucket(
                label=label,
                count=buckets.get(label, 0),
                percentage=_share(buckets.get(label, 0), total),
            )
            for label in _STAGE_RESPONSIBILITY_ORDER
        ]

    def _document_analysis(
        self, failure_report: RetrievalFailureReport
    ) -> tuple[list[DocumentAnalysis], int]:
        total = failure_report.aggregate.total_failures
        counts = {category: 0 for category in DocumentCategory}
        unclassified = 0
        for query in failure_report.per_query:
            for failure in query.failures:
                if failure.failure_category is None:
                    continue
                if not failure.relative_path and not failure.chunk_type:
                    unclassified += 1
                    continue
                category = categorize_document(failure.chunk_type, failure.relative_path)
                counts[category] += 1
        return [
            DocumentAnalysis(
                category=category,
                count=counts[category],
                percentage=_share(counts[category], total),
            )
            for category in DocumentCategory
        ], unclassified

    def _intent_analysis(
        self,
        benchmark_report: RetrievalBenchmarkReport,
        failure_report: RetrievalFailureReport,
        golden_queries: list[GoldenQuery] | None,
    ) -> list[IntentAnalysis]:
        text_by_id: dict[str, str] = {q.query_id: q.query_text for q in failure_report.per_query}
        if golden_queries:
            text_by_id.update({q.query_id: q.query_text for q in golden_queries})

        benchmark_by_id: dict[str, RetrievalBenchmarkQueryResult] = {
            r.query_id: r for r in benchmark_report.per_query_results
        }

        intent_queries: dict[IntentCategory, set[str]] = {i: set() for i in IntentCategory}
        for query_id in benchmark_by_id:
            intent_queries[classify_query_intent(text_by_id.get(query_id, ""))].add(query_id)
        for query in failure_report.per_query:
            intent_queries[classify_query_intent(query.query_text)].add(query.query_id)

        failure_ranks: dict[IntentCategory, list[int]] = {i: [] for i in IntentCategory}
        failure_types: dict[IntentCategory, dict[FailureCategory, int]] = {
            i: {} for i in IntentCategory
        }
        for query in failure_report.per_query:
            intent = classify_query_intent(query.query_text)
            for failure in query.failures:
                if failure.failure_category is None:
                    continue
                if failure.final_rank is not None:
                    failure_ranks[intent].append(failure.final_rank)
                failure_types[intent][failure.failure_category] = (
                    failure_types[intent].get(failure.failure_category, 0) + 1
                )

        result: list[IntentAnalysis] = []
        for intent in IntentCategory:
            query_ids = sorted(intent_queries[intent])
            if not query_ids:
                result.append(IntentAnalysis(intent=intent))
                continue
            benchmark_ids = [qid for qid in query_ids if qid in benchmark_by_id]
            top1 = (
                sum(benchmark_by_id[qid].top1_recall for qid in benchmark_ids) / len(benchmark_ids)
                if benchmark_ids
                else 0.0
            )
            mrr = (
                sum(benchmark_by_id[qid].mrr for qid in benchmark_ids) / len(benchmark_ids)
                if benchmark_ids
                else 0.0
            )
            ranks = failure_ranks[intent]
            average_failure_rank = sum(ranks) / len(ranks) if ranks else 0.0
            type_counts = failure_types[intent]
            result.append(IntentAnalysis(
                intent=intent,
                query_count=len(query_ids),
                top1_recall=top1,
                mrr=mrr,
                average_failure_rank=average_failure_rank,
                failure_count=sum(type_counts.values()),
                most_common_failure_type=_most_common_category(type_counts),
            ))
        return result

    def _recommendations(
        self,
        benchmark_report: RetrievalBenchmarkReport,
        failure_report: RetrievalFailureReport,
        document_analysis: list[DocumentAnalysis],
        golden_queries: list[GoldenQuery] | None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        total = failure_report.aggregate.total_failures

        counts = failure_report.aggregate.counts

        def share(category: FailureCategory) -> float:
            return _share(counts.get(category.value, 0), total)

        # R1: base recall gap — chunks never surfaced at all.
        not_retrieved = counts.get(FailureCategory.NOT_RETRIEVED.value, 0)
        if total and _share(not_retrieved, total) > self._threshold * 100:
            recommendations.append(Recommendation(
                rule="not_retrieved_dominance",
                condition=(
                    f"NOT_RETRIEVED accounts for {_share(not_retrieved, total):.1f}% of "
                    f"failures ({not_retrieved}/{total})."
                ),
                recommendation=(
                    "Investigate embeddings/chunking — the top failure cause is chunks "
                    "never surfaced by any retrieval signal."
                ),
            ))

        # R2: prompt budget dropping retrieved chunks.
        prompt_cutoff = counts.get(FailureCategory.PROMPT_CUTOFF.value, 0)
        if total and _share(prompt_cutoff, total) > self._threshold * 100:
            recommendations.append(Recommendation(
                rule="prompt_cutoff_dominance",
                condition=(
                    f"PROMPT_CUTOFF accounts for {_share(prompt_cutoff, total):.1f}% of "
                    f"failures ({prompt_cutoff}/{total})."
                ),
                recommendation=(
                    "Investigate prompt assembly — retrieved chunks are being dropped by "
                    "the prompt context budget."
                ),
            ))

        # R3: source adjustments dominating demotions.
        source_prior = counts.get(FailureCategory.SOURCE_PRIOR.value, 0)
        if total and _share(source_prior, total) > self._threshold * 100:
            recommendations.append(Recommendation(
                rule="source_prior_dominance",
                condition=(
                    f"SOURCE_PRIOR accounts for {_share(source_prior, total):.1f}% of "
                    f"failures ({source_prior}/{total})."
                ),
                recommendation="Investigate source ranking — source adjustments are the top demotion cause.",
            ))

        # R4: the other single-stage demotion causes.
        for category, rule, message in (
            (FailureCategory.GRAPH_EXPANSION, "graph_expansion_dominance", "Investigate graph expansion."),
            (FailureCategory.INTENT, "intent_dominance", "Investigate intent ranking."),
            (FailureCategory.RERANKER, "reranker_dominance", "Investigate reranker."),
        ):
            if total and share(category) > self._threshold * 100:
                recommendations.append(Recommendation(
                    rule=rule,
                    condition=f"{category.value} accounts for {share(category):.1f}% of failures ({counts.get(category.value, 0)}/{total}).",
                    recommendation=message,
                ))

        # R5: base retrieval + fusion combined (not surfaced OR fused below cut).
        base = (
            counts.get(FailureCategory.NOT_RETRIEVED.value, 0)
            + counts.get(FailureCategory.RANKED_TOO_LOW.value, 0)
        )
        if total and _share(base, total) > 50.0:
            recommendations.append(Recommendation(
                rule="base_retrieval_dominance",
                condition=(
                    f"Base retrieval + fusion (NOT_RETRIEVED + RANKED_TOO_LOW) accounts for "
                    f"{_share(base, total):.1f}% of failures."
                ),
                recommendation=(
                    "Investigate base retrieval recall and fusion — most failures originate "
                    "before any ranking adjustment."
                ),
            ))

        # R6: log chunks dominating document failures (the task's example).
        log_count = next(
            (d.count for d in document_analysis if d.category is DocumentCategory.LOGS),
            0,
        )
        if total and _share(log_count, total) > self._threshold * 100:
            recommendations.append(Recommendation(
                rule="log_documents_dominance",
                condition=(
                    f"Log chunks account for {_share(log_count, total):.1f}% of failures "
                    f"({log_count}/{total})."
                ),
                recommendation="Investigate source ranking — log chunks dominate failures.",
            ))

        # R7: worst non-log document category.
        category_rules = {
            DocumentCategory.UVM: "Investigate UVM/verification indexing and retrieval.",
            DocumentCategory.RTL: "Investigate RTL indexing and retrieval.",
            DocumentCategory.GENERATED: "Investigate generated-file filtering and retrieval.",
            DocumentCategory.SPECIFICATION: "Investigate documentation indexing and retrieval.",
            DocumentCategory.MARKDOWN: "Investigate documentation indexing and retrieval.",
        }
        worst_doc = max(document_analysis, key=lambda d: d.count) if document_analysis else None
        if (
            worst_doc is not None
            and total
            and worst_doc.count > 0
            and worst_doc.category in category_rules
            and _share(worst_doc.count, total) > self._threshold * 100
        ):
            recommendations.append(Recommendation(
                rule="document_category_dominance",
                condition=(
                    f"{worst_doc.category.value} is the dominant failure category "
                    f"({worst_doc.count}/{total} failures)."
                ),
                recommendation=category_rules[worst_doc.category],
            ))

        # R8: worst-performing intent (enough queries to be meaningful).
        intent_analysis = self._intent_analysis(benchmark_report, failure_report, golden_queries)
        eligible = [i for i in intent_analysis if i.query_count >= 2]
        if eligible:
            worst = min(eligible, key=lambda i: i.top1_recall)
            if worst.top1_recall < 0.5:
                recommendations.append(Recommendation(
                    rule="worst_intent",
                    condition=(
                        f"{worst.intent.value} queries have the lowest top-1 recall "
                        f"({worst.top1_recall:.2f}, {worst.query_count} queries)."
                    ),
                    recommendation=f"Investigate retrieval for {worst.intent.value} queries.",
                ))

        # R9: overall retrieval health below the top-1 recall bar.
        if benchmark_report.top1_recall < 0.5 and benchmark_report.query_count > 0:
            recommendations.append(Recommendation(
                rule="low_overall_health",
                condition=f"Overall top-1 recall is {benchmark_report.top1_recall:.2f}.",
                recommendation="Investigate overall retrieval quality — top-1 recall is below 50%.",
            ))

        if not recommendations:
            recommendations.append(Recommendation(
                rule="no_failures",
                condition=f"{failure_report.query_count} queries, 0 retrieval failures",
                recommendation="No retrieval failures to investigate.",
            ))

        return recommendations


def to_json_dict(report: RetrievalAnalyticsReport) -> dict:
    """JSON-safe dict of the report (enums -> values, None -> null)."""
    return _json_safe(report.model_dump(mode="json"))


def format_markdown(report: RetrievalAnalyticsReport) -> str:
    """Human-readable markdown report with the required sections:
    Executive Summary, Overall Health, Failure Distribution, Stage
    Attribution, Document Analysis, Intent Analysis, and Engineering
    Recommendations."""
    lines: list[str] = ["# Retrieval Analytics", ""]

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Queries: {report.query_count}")
    lines.append(f"- Expected chunks: {report.expected_chunk_count}")
    lines.append(f"- Retrieval failures: {report.total_failures}")
    lines.append(f"- Top-1 recall: {_pct(report.top1_recall)}")
    lines.append(f"- MRR: {report.mrr:.2f}")
    worst = _worst_failure(report.failure_distribution)
    if worst is not None:
        lines.append(f"- Dominant failure cause: {worst.label} ({worst.count})")

    lines.append("")
    lines.append("## Overall Health")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Top-1 Recall | {_pct(report.top1_recall)} |")
    lines.append(f"| Top-3 Recall | {_pct(report.top3_recall)} |")
    lines.append(f"| Top-5 Recall | {_pct(report.top5_recall)} |")
    lines.append(f"| MRR | {report.mrr:.2f} |")
    lines.append(f"| Average Rank | {report.average_rank:.2f} |")

    lines.append("")
    lines.append("## Failure Distribution")
    lines.append("")
    for bucket in report.failure_distribution:
        label = bucket.label
        lines.append(f"{label} {' ' * max(1, 20 - len(label))} {bucket.count} ({bucket.percentage:.1f}%)")

    lines.append("")
    lines.append("## Stage Attribution")
    lines.append("")
    lines.append("| Signal / stage | Count | % |")
    lines.append("|---|---|---|")
    for bucket in report.stage_attribution:
        lines.append(f"| {bucket.label} | {bucket.count} | {bucket.percentage:.1f}% |")

    lines.append("")
    lines.append("## Document Analysis")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|---|---|---|")
    for doc in report.document_analysis:
        lines.append(f"| {doc.category.value} | {doc.count} | {doc.percentage:.1f}% |")
    if report.document_unclassified:
        unclassified_pct = _share(report.document_unclassified, report.total_failures)
        lines.append(f"| unclassified (no metadata) | {report.document_unclassified} | {unclassified_pct:.1f}% |")

    lines.append("")
    lines.append("## Intent Analysis")
    lines.append("")
    lines.append("| Intent | Queries | Top-1 Recall | MRR | Avg Failure Rank | Failures | Most Common Failure |")
    lines.append("|---|---|---|---|---|---|---|")
    for intent in report.intent_analysis:
        lines.append(
            f"| {intent.intent.value} | {intent.query_count} | {intent.top1_recall:.2f} | "
            f"{intent.mrr:.2f} | {intent.average_failure_rank:.1f} | {intent.failure_count} | "
            f"{intent.most_common_failure_type} |"
        )

    lines.append("")
    lines.append("## Engineering Recommendations")
    lines.append("")
    if report.recommendations:
        for index, rec in enumerate(report.recommendations, start=1):
            lines.append(f"{index}. **{rec.rule}** — {rec.condition}")
            lines.append(f"   {rec.recommendation}")
            lines.append("")
    else:
        lines.append("No recommendation rules triggered.")

    return "\n".join(lines)


def write_analytics_reports(
    report: RetrievalAnalyticsReport,
    json_path: Path | str,
    md_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Persist the machine-readable JSON report and (optionally) the
    markdown report. Returns the paths written."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_json_dict(report), indent=2) + "\n")

    written_md: Path | None = None
    if md_path is not None:
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_markdown(report) + "\n")
        written_md = md_path
    return json_path, written_md


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _path_segments(path: str) -> set[str]:
    return {seg.lower() for seg in path.replace("\\", "/").split("/") if seg}


def _most_common_category(counts: dict[FailureCategory, int]) -> str:
    """Highest-count category, ties broken to the earliest ``FailureCategory``
    in enum order (deterministic). ``"none"`` when no failures."""
    if not counts:
        return "none"
    best: FailureCategory | None = None
    best_count = -1
    for category in FailureCategory:
        count = counts.get(category, 0)
        if count > best_count:
            best = category
            best_count = count
    return best.value if best is not None else "none"


def _share(count: int, total: int) -> float:
    return count / total * 100.0 if total else 0.0


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _worst_failure(buckets: list[Bucket]) -> Bucket | None:
    if not buckets:
        return None
    worst = max(buckets, key=lambda b: b.count)
    return worst if worst.count > 0 else None


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value
