from __future__ import annotations

from semrag.evaluation.models import GoldenQuery, ExpectedIntent, RepoArea

BENCHMARKS: list[GoldenQuery] = [
    # ── Architecture ──────────────────────────────────────────────────────
    GoldenQuery(
        query_id="arch_01",
        project_id="semrag",
        query_text="What is the repository architecture? How does the retrieval pipeline work?",
        primary_files={
            "src/semrag/retrieval/pipeline.py",
            "src/semrag/retrieval/models.py",
        },
        secondary_files={
            "src/semrag/retrieval/fusion.py",
            "src/semrag/retrieval/intent.py",
            "src/semrag/retrieval/graph_expansion.py",
            "src/semrag/retrieval/reranking.py",
            "src/semrag/retrieval/lexical_search.py",
            "src/semrag/retrieval/vector_search_adapter.py",
            "ARCHITECTURE.md",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Core retrieval system architecture — pipeline orchestration, models, search, fusion, reranking.",
    ),
    GoldenQuery(
        query_id="arch_02",
        project_id="semrag",
        query_text="How does graph expansion work? How are structurally related chunks retrieved?",
        primary_files={
            "src/semrag/retrieval/graph_expansion.py",
            "src/semrag/storage/postgres/graph_queries.py",
        },
        secondary_files={
            "src/semrag/retrieval/pipeline.py",
            "src/semrag/storage/graph/interfaces.py",
            "src/semrag/storage/graph/neo4j_store.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Graph expansion mechanism — how the retrieval pipeline pulls in structurally related chunks via IR graph traversal.",
    ),
    GoldenQuery(
        query_id="arch_03",
        project_id="semrag",
        query_text="How is query intent classified and used to boost retrieval results?",
        primary_files={
            "src/semrag/retrieval/intent.py",
        },
        secondary_files={
            "src/semrag/retrieval/pipeline.py",
            "src/semrag/services/services.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Intent classification — how QueryIntent is determined from keywords and used to bias chunk ranking.",
    ),
    # ── Retrieval ─────────────────────────────────────────────────────────
    GoldenQuery(
        query_id="retrieval_01",
        project_id="semrag",
        query_text="How does Reciprocal Rank Fusion work? How are vector and lexical scores combined?",
        primary_files={
            "src/semrag/retrieval/fusion.py",
            "src/semrag/retrieval/pipeline.py",
        },
        secondary_files={
            "src/semrag/retrieval/models.py",
            "src/semrag/retrieval/reranking.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="RRF fusion — combining vector and lexical ranked lists by position.",
    ),
    GoldenQuery(
        query_id="retrieval_02",
        project_id="semrag",
        query_text="How does the reranker work? How are chunks rescored with Jaccard similarity?",
        primary_files={
            "src/semrag/retrieval/reranking.py",
            "src/semrag/retrieval/pipeline.py",
        },
        secondary_files={
            "src/semrag/retrieval/models.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Reranking stage — LexicalOverlapReranker and HTTPRerankerClient.",
    ),
    GoldenQuery(
        query_id="retrieval_03",
        project_id="semrag",
        query_text="How does lexical search work? How is PostgreSQL full-text search used?",
        primary_files={
            "src/semrag/retrieval/lexical_search.py",
        },
        secondary_files={
            "src/semrag/storage/vector/models.py",
            "src/semrag/retrieval/pipeline.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Lexical search via Postgres ts_rank_cd over the chunks text_search column.",
    ),
    GoldenQuery(
        query_id="retrieval_04",
        project_id="semrag",
        query_text="How does vector search work? How are embeddings stored and searched in pgvector?",
        primary_files={
            "src/semrag/storage/vector/pgvector_store.py",
            "src/semrag/storage/vector/interfaces.py",
        },
        secondary_files={
            "src/semrag/storage/vector/models.py",
            "src/semrag/storage/vector/pgvector_models.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.STORAGE,
        description="Vector search via pgvector — storage and cosine-similarity search over embeddings.",
    ),
    # ── RTL / Design ──────────────────────────────────────────────────────
    GoldenQuery(
        query_id="config_01",
        project_id="semrag",
        query_text="How are embeddings generated? How does the embedding pipeline work?",
        primary_files={
            "src/semrag/ingestion/embedding/pipeline.py",
            "src/semrag/ingestion/embedding/interfaces.py",
        },
        secondary_files={
            "src/semrag/ingestion/embedding/factory.py",
            "src/semrag/ingestion/embedding/tfidf_provider.py",
            "src/semrag/ingestion/embedding/ollama_provider.py",
            "src/semrag/ingestion/embedding/openai_compatible_provider.py",
            "src/semrag/ingestion/embedding/cache.py",
        },
        expected_intent=ExpectedIntent.CONFIG,
        repo_area=RepoArea.INGESTION,
        description="Embedding generation — how text gets turned into vectors via provider abstraction.",
    ),
    GoldenQuery(
        query_id="config_02",
        project_id="semrag",
        query_text="Where are chunk metadata stored? How does the chunk storage schema work?",
        primary_files={
            "src/semrag/storage/vector/pgvector_models.py",
            "src/semrag/storage/vector/pgvector_store.py",
        },
        secondary_files={
            "src/semrag/storage/postgres/models.py",
            "src/semrag/ingestion/chunking/pipeline.py",
        },
        expected_intent=ExpectedIntent.CONFIG,
        repo_area=RepoArea.STORAGE,
        description="Chunk metadata storage — ChunkRow model, pgvector schema, chunk persistence.",
    ),
    GoldenQuery(
        query_id="config_03",
        project_id="semrag",
        query_text="How does the service layer work? How are ingestion and query orchestrated?",
        primary_files={
            "src/semrag/services/services.py",
        },
        secondary_files={
            "src/semrag/retrieval/pipeline.py",
            "src/semrag/ingestion/scanner/orchestrator.py",
            "src/semrag/reasoning/agents/tools.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.CORE,
        description="Service layer — IngestionService and QueryService orchestrating the full pipeline.",
    ),
    # ── API ───────────────────────────────────────────────────────────────
    GoldenQuery(
        query_id="api_01",
        project_id="semrag",
        query_text="How is search exposed through FastAPI? What endpoints are available?",
        primary_files={
            "src/semrag/api/app.py",
            "src/semrag/api/routers/query.py",
            "src/semrag/api/schemas.py",
        },
        secondary_files={
            "src/semrag/api/routers/projects.py",
            "src/semrag/api/dependencies.py",
            "src/semrag/api/config.py",
            "src/semrag/api/auth.py",
            "src/semrag/api/rate_limit.py",
            "src/semrag/api/query_cache.py",
        },
        expected_intent=ExpectedIntent.API,
        repo_area=RepoArea.API,
        description="FastAPI application — query and project endpoints, schemas, auth, rate limiting.",
    ),
    # ── Monitoring / Observability ────────────────────────────────────────
    GoldenQuery(
        query_id="observability_01",
        project_id="semrag",
        query_text="How is observability implemented? How are debug artifacts and metrics tracked?",
        primary_files={
            "src/semrag/core/observability.py",
            "src/semrag/core/debug_artifacts.py",
            "src/semrag/core/settings.py",
        },
        secondary_files={
            "src/semrag/retrieval/pipeline.py",
            "src/semrag/services/services.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.CORE,
        description="Observability — metrics, logging, debug artifact writer, stage timing.",
    ),
    GoldenQuery(
        query_id="retrieval_trace_01",
        project_id="semrag",
        query_text="How does the ranking trace work? How is per-stage chunk scoring recorded?",
        primary_files={
            "src/semrag/retrieval/pipeline.py",
        },
        secondary_files={
            "src/semrag/retrieval/models.py",
            "src/semrag/core/debug_artifacts.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.RETRIEVAL,
        description="Ranking trace system — ChunkTrace, RankingTrace, trace sinks, debug JSON export.",
    ),
    # ── Ingestion ─────────────────────────────────────────────────────────
    GoldenQuery(
        query_id="ingestion_01",
        project_id="semrag",
        query_text="How does the ingestion pipeline work? How are files scanned, parsed, chunked, and indexed?",
        primary_files={
            "src/semrag/ingestion/scanner/orchestrator.py",
            "src/semrag/ingestion/ir/builder.py",
            "src/semrag/ingestion/chunking/pipeline.py",
        },
        secondary_files={
            "src/semrag/ingestion/parsers/pipeline.py",
            "src/semrag/ingestion/embedding/pipeline.py",
            "src/semrag/services/services.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.INGESTION,
        description="Full ingestion pipeline — scan, parse, build IR graph, chunk, embed, persist.",
    ),
    # ── Verification ──────────────────────────────────────────────────────
    GoldenQuery(
        query_id="verification_01",
        project_id="semrag",
        query_text="How does the verification/evaluation framework work? How is retrieval quality measured?",
        primary_files={
            "src/semrag/evaluation/harness.py",
            "src/semrag/evaluation/metrics.py",
            "src/semrag/evaluation/models.py",
        },
        secondary_files={
            "src/semrag/evaluation/citation_validator.py",
        },
        expected_intent=ExpectedIntent.VERIFICATION,
        repo_area=RepoArea.EVALUATION,
        description="Evaluation framework — harness, precision/recall/NDCG metrics, citation validation.",
    ),
    GoldenQuery(
        query_id="verification_02",
        project_id="semrag",
        query_text="How does the prompt builder assemble context? How are chunks formatted and budgeted?",
        primary_files={
            "src/semrag/reasoning/prompting/pipeline.py",
            "src/semrag/reasoning/prompting/context_assembler.py",
            "src/semrag/reasoning/prompting/citation.py",
        },
        secondary_files={
            "src/semrag/reasoning/prompting/templates.py",
            "src/semrag/reasoning/prompting/token_estimator.py",
        },
        expected_intent=ExpectedIntent.VERIFICATION,
        repo_area=RepoArea.REASONING,
        description="Prompt assembly — context_assembler budget packing, citation formatting, templates.",
    ),
    # ── Config ────────────────────────────────────────────────────────────
    GoldenQuery(
        query_id="config_04",
        project_id="semrag",
        query_text="How is the application configured? How are settings managed?",
        primary_files={
            "src/semrag/core/settings.py",
        },
        secondary_files={
            "src/semrag/api/config.py",
            "pyproject.toml",
        },
        expected_intent=ExpectedIntent.CONFIG,
        repo_area=RepoArea.CONFIG,
        description="Application configuration — Settings pydantic model, env vars, API config.",
    ),
    # ── Reasoning / LLM ───────────────────────────────────────────────────
    GoldenQuery(
        query_id="reasoning_01",
        project_id="semrag",
        query_text="How does the LLM agent loop work? How are tools used for follow-up questions?",
        primary_files={
            "src/semrag/reasoning/agents/agent_loop.py",
            "src/semrag/reasoning/agents/tools.py",
        },
        secondary_files={
            "src/semrag/services/services.py",
            "src/semrag/reasoning/llm/interfaces.py",
        },
        expected_intent=ExpectedIntent.ARCHITECTURE,
        repo_area=RepoArea.REASONING,
        description="Agent loop — how the multi-turn LLM conversation works with tool access.",
    ),
]


def get_benchmarks() -> list[GoldenQuery]:
    return list(BENCHMARKS)


def get_benchmarks_by_area(area: RepoArea) -> list[GoldenQuery]:
    return [q for q in BENCHMARKS if q.repo_area == area]


def get_benchmarks_by_intent(intent: ExpectedIntent) -> list[GoldenQuery]:
    return [q for q in BENCHMARKS if q.expected_intent == intent]
