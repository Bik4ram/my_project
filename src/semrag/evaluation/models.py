from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RepoArea(str, Enum):
    CORE = "core"
    INGESTION = "ingestion"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    API = "api"
    EVALUATION = "evaluation"
    CONFIG = "config"
    GENERAL = "general"


class ExpectedIntent(str, Enum):
    ARCHITECTURE = "architecture"
    VERIFICATION = "verification"
    RTL = "rtl"
    DEBUG = "debug"
    CONFIG = "config"
    API = "api"
    GENERAL = "general"


class GoldenQuery(BaseModel):
    query_id: str
    project_id: str
    query_text: str
    relevant_chunk_ids: set[str] = Field(
        default_factory=set,
        description="Chunk IDs a human has judged relevant — the ground truth.",
    )
    primary_files: set[str] = Field(
        default_factory=set,
        description="Relative paths that must appear in results (core answers).",
    )
    secondary_files: set[str] = Field(
        default_factory=set,
        description="Relative paths that should appear (supporting context).",
    )
    expected_intent: ExpectedIntent = ExpectedIntent.GENERAL
    repo_area: RepoArea = RepoArea.GENERAL
    description: str = ""


class RetrievalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    k: int
    primary_file_recall: float = 0.0
    secondary_file_recall: float = 0.0
    hit_rate: float = 0.0
    average_rank: float = 0.0
    primary_files_found: list[str] = Field(default_factory=list)
    secondary_files_found: list[str] = Field(default_factory=list)
    primary_files_missed: list[str] = Field(default_factory=list)
    secondary_files_missed: list[str] = Field(default_factory=list)


class EvalQueryResult(BaseModel):
    query_id: str
    query_text: str
    metrics: RetrievalMetrics
    latency_ms: float
    retrieved_paths: list[str] = Field(default_factory=list)
    estimated_total_tokens: int = 0
    excluded_chunk_count: int = 0


class CoverageReport(BaseModel):
    total_queries: int = 0
    total_primary: int = 0
    total_secondary: int = 0
    primary_found: int = 0
    secondary_found: int = 0
    primary_recall: float = 0.0
    secondary_recall: float = 0.0
    by_area: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_intent: dict[str, dict[str, float]] = Field(default_factory=dict)


class AggregateReport(BaseModel):
    query_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_hit_rate: float
    mean_average_rank: float
    mean_primary_recall: float
    mean_secondary_recall: float
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    latency_p50_ms: float
    latency_p95_ms: float
    per_query_results: list[EvalQueryResult] = Field(default_factory=list)


class ChunkCategory(Enum):
    """Source-material category of a retrieved chunk, used by the
    retrieval benchmark to report what the top-k is made of (logs vs
    RTL vs spec docs). A heuristic classification of the chunk's
    ``chunk_type`` and ``relative_path``."""

    LOG = "log"
    RTL = "rtl"
    SPEC = "spec"


class RetrievalBenchmarkQueryResult(BaseModel):
    """Per-query outcome of the retrieval benchmark. ``average_rank``
    follows ``metrics.average_rank`` semantics: ``float("inf")`` when no
    relevant chunk was retrieved (the JSON emitter renders it as null)."""

    query_id: str
    top_k: int
    top1_recall: float = 0.0
    top3_recall: float = 0.0
    top5_recall: float = 0.0
    mrr: float = 0.0
    average_rank: float = 0.0
    log_count: int = 0
    rtl_count: int = 0
    spec_count: int = 0
    retrieved_paths: list[str] = Field(default_factory=list)


class RetrievalBenchmarkReport(BaseModel):
    """Aggregate retrieval benchmark: the multi-k recall profile, MRR,
    average rank, and the mean chunk-category composition of the top-k.
    ``mean_average_rank`` is averaged over queries that retrieved at
    least one relevant chunk (finite ranks only), matching the harness."""

    query_count: int = 0
    top_k: int = 10
    top1_recall: float = 0.0
    top3_recall: float = 0.0
    top5_recall: float = 0.0
    mean_mrr: float = 0.0
    mean_average_rank: float = 0.0
    mean_log_count: float = 0.0
    mean_rtl_count: float = 0.0
    mean_spec_count: float = 0.0
    per_query_results: list[RetrievalBenchmarkQueryResult] = Field(default_factory=list)


class FailureCategory(Enum):
    """Attributed cause of a chunk failing to reach the LLM context.

    Each category except the two sentinels maps to the ranking stage
    whose top-k demotion dropped the chunk: ``SOURCE_PRIOR`` covers both
    the noisy-source penalty and the document-type prior (the source
    adjustments), ``RANKED_TOO_LOW`` means the chunk fused below the
    context cut before any later stage intervened, ``NOT_RETRIEVED``
    means it never surfaced through any retrieval signal,
    ``PROMPT_CUTOFF`` means it was retrieved but the prompt context
    budget excluded it, and ``UNKNOWN`` is the no-confident-explanation
    fallback."""

    NOT_RETRIEVED = "NOT_RETRIEVED"
    RANKED_TOO_LOW = "RANKED_TOO_LOW"
    SOURCE_PRIOR = "SOURCE_PRIOR"
    GRAPH_EXPANSION = "GRAPH_EXPANSION"
    INTENT = "INTENT"
    RERANKER = "RERANKER"
    DIVERSITY = "DIVERSITY"
    PROMPT_CUTOFF = "PROMPT_CUTOFF"
    UNKNOWN = "UNKNOWN"


class StageContributions(BaseModel):
    """Ranking-breakdown columns for one expected chunk: the per-stage
    signals the pipeline already records in ``ChunkTrace``, re-exposed
    verbatim so the report recomputes nothing and duplicates nothing."""

    vector_score: float | None = None
    vector_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    graph_expansion_score: float | None = None
    intent_boost: float | None = None
    doc_type_prior: float | None = None
    reranker_score: float | None = None
    diversity: float | None = None
    diversity_reason: str | None = None


class ExpectedChunkAnalysis(BaseModel):
    """Analysis of ONE expected chunk of a query: retrieval status, the
    per-stage rank history reconstructed from the trace, the attributed
    failure category, and a diagnostic-only primary cause + recommended
    investigation (an "inspect X" pointer, never a code change)."""

    query_id: str
    chunk_id: str
    relative_path: str = ""
    chunk_type: str = ""
    retrieved: bool = False
    retrieved_stage: str = ""
    final_rank: int | None = None
    reached_llm: bool = False
    failure_category: FailureCategory | None = None
    primary_cause: str = ""
    recommendation: str = ""
    stage_ranks: dict[str, int] = Field(default_factory=dict)
    contributions: StageContributions = Field(default_factory=StageContributions)


class QueryFailureAnalysis(BaseModel):
    """Per-query failure analysis: the expected chunks, the actual top-k,
    and the classification for every expected chunk that failed."""

    query_id: str
    query_text: str
    top_k: int
    expected_chunk_count: int = 0
    failure_count: int = 0
    actual_top_k: list[str] = Field(default_factory=list)
    failures: list[ExpectedChunkAnalysis] = Field(default_factory=list)


class FailureAggregate(BaseModel):
    """Aggregate failure summary across all analyzed queries."""

    total_failures: int = 0
    counts: dict[str, int] = Field(default_factory=dict)


class RetrievalFailureReport(BaseModel):
    """Complete failure-analysis report: the per-query analyses plus the
    aggregate failure summary."""

    query_count: int = 0
    expected_chunk_count: int = 0
    failure_count: int = 0
    aggregate: FailureAggregate = Field(default_factory=FailureAggregate)
    per_query: list[QueryFailureAnalysis] = Field(default_factory=list)


class DocumentCategory(Enum):
    """Source-material category of a FAILED chunk, used by the retrieval
    analytics to answer "which document categories dominate failures?".
    Finer than the benchmark's ``ChunkCategory`` (which collapses spec
    docs and drops UVM/generated code); a heuristic over the chunk's
    ``chunk_type`` and ``relative_path``, applied analytics-side only."""

    RTL = "rtl"
    SPECIFICATION = "specification"
    MARKDOWN = "markdown"
    UVM = "uvm"
    LOGS = "logs"
    GENERATED = "generated"
    OTHER = "other"


class IntentCategory(Enum):
    """Analytics-side query intent (a user-question taxonomy). Distinct
    from the retrieval intent used for intent boosting; classified from
    the query text with deterministic keyword rules."""

    EXPLAIN = "explain"
    DEBUG = "debug"
    NAVIGATION = "navigation"
    SPECIFICATION = "specification"
    GENERAL = "general"
    UNKNOWN = "unknown"


class Bucket(BaseModel):
    """A named count + share (percentage) bucket used by the failure
    distribution and stage attribution tables. Percentage is over the
    total failures (0.0 when there are none)."""

    label: str
    count: int = 0
    percentage: float = 0.0


class DocumentAnalysis(BaseModel):
    """Failure count + percentage for one document category."""

    category: DocumentCategory
    count: int = 0
    percentage: float = 0.0


class IntentAnalysis(BaseModel):
    """Per-intent analytics: retrieval health over that intent's queries
    plus its failure profile. ``average_failure_rank`` is the mean final
    rank of failed chunks that had a rank (NOT_RETRIEVED chunks never
    ranked and are excluded); ``most_common_failure_type`` is the top
    ``FailureCategory`` value, or ``"none"`` when the intent had no
    failures. Ties break to the earliest category in enum order."""

    intent: IntentCategory
    query_count: int = 0
    top1_recall: float = 0.0
    mrr: float = 0.0
    average_failure_rank: float = 0.0
    failure_count: int = 0
    most_common_failure_type: str = "none"


class Recommendation(BaseModel):
    """A deterministic engineering recommendation. ``rule`` is a stable
    rule identifier, ``condition`` states the triggered evidence, and
    ``recommendation`` is a diagnostic-only pointer ("investigate X"),
    never a code change."""

    rule: str
    condition: str
    recommendation: str


class RetrievalAnalyticsReport(BaseModel):
    """The complete retrieval analytics report produced from a
    ``RetrievalBenchmarkReport`` + ``RetrievalFailureReport`` pair:
    overall health, failure distribution, stage attribution, document
    analysis, intent analysis, and engineering recommendations."""

    query_count: int = 0
    expected_chunk_count: int = 0
    top1_recall: float = 0.0
    top3_recall: float = 0.0
    top5_recall: float = 0.0
    mrr: float = 0.0
    average_rank: float = 0.0
    total_failures: int = 0
    failure_distribution: list[Bucket] = Field(default_factory=list)
    stage_attribution: list[Bucket] = Field(default_factory=list)
    document_analysis: list[DocumentAnalysis] = Field(default_factory=list)
    document_unclassified: int = 0
    intent_analysis: list[IntentAnalysis] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
