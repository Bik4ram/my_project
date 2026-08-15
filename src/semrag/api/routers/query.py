"""
Query endpoints.

Two endpoints answering the same question, at two different levels of
client sophistication:

- POST /v1/projects/{project_id}/query - one request, one JSON response.
  What a simple client (a script, a CI check) wants.
- POST /v1/projects/{project_id}/query/stream - Server-Sent Events,
  emitting a stage-progress event as each retrieval phase completes
  (vector search, lexical search, fusion, graph expansion, reranking,
  prompt assembly) before a final "result" event with the same payload
  the non-streaming endpoint returns. This is real, working SSE
  infrastructure - Part 1's "streaming" requirement - even though there's
  no LLM response to stream yet (that's Part 13's job): streaming the
  retrieval pipeline's own stages is genuinely useful on its own (a
  client can show "searching..." / "expanding context..." progress for a
  large project where retrieval itself takes a moment), and it establishes
  the exact SSE pattern Part 13 will extend to stream LLM tokens, rather
  than that part having to invent a different streaming mechanism.

Neither endpoint calls an LLM - both return an AssembledPrompt-shaped
(messages ready to send to a model), consistent with this series' part
boundaries: retrieval and prompt construction are Parts 10-11's job,
calling a model is Part 13's job, and this API layer exposes what's
actually built rather than a preview of a part that doesn't exist yet.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from semrag.api.auth import require_api_key
from semrag.api.config import get_embedding_provider_dir
from semrag.api.dependencies import get_db_session, get_query_service, get_settings
from semrag.api.project_auth import get_project_authorizer
from semrag.api.query_cache import cache_key, get_shared_query_cache
from semrag.api.rate_limit import enforce_rate_limit
from semrag.api.schemas import QueryRequest, QueryResponse
from semrag.core.settings import Settings
from semrag.ingestion.embedding.factory import build_embedding_provider
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.reasoning.llm.interfaces import LLMProvider
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.retrieval.pipeline import HybridRetriever
from semrag.retrieval.query_expansion import build_query_expansion_config
from semrag.retrieval.ranking.diversity import DEFAULT_DIVERSITY
from semrag.retrieval.ranking.source import apply_source_ranking
from semrag.retrieval.reranking import LexicalOverlapReranker
from semrag.retrieval.source_authority import DEFAULT_SOURCE_AUTHORITY
from semrag.storage.vector.models import VectorSearchFilter

router = APIRouter(prefix="/v1/projects", tags=["query"])

logger = logging.getLogger(__name__)


def _build_reranker(enable: bool, settings: Settings) -> LexicalOverlapReranker | None:
    """The lexical-overlap reranker wired with the RI-07 metadata tie-break
    knobs, or None when reranking is disabled for the request."""
    if not enable:
        return None
    return LexicalOverlapReranker(
        enable_metadata_tiebreak=settings.retrieval_rerank_metadata_tiebreak,
        near_tie_epsilon=settings.retrieval_rerank_near_tie_epsilon,
    )


def _source_authority_config(settings: Settings):
    """Build the SourceAuthorityConfig from settings, or None to fall back
    to the default authority weights (mirrors QueryService wiring)."""
    overrides = settings.source_authority_overrides or {}
    max_boost = settings.source_authority_max_boost
    if not overrides and max_boost is None:
        return None
    return DEFAULT_SOURCE_AUTHORITY.with_overrides(overrides, max_boost=max_boost)


def _load_query_embedding(project_id: str, query: str) -> list[float]:
    settings = get_settings()
    if settings.embedding_provider == "tfidf":
        provider = TfidfEmbeddingProvider.load(f"{get_embedding_provider_dir()}/{project_id}.joblib")
    else:
        provider = build_embedding_provider(settings)
    return provider.embed_texts([query])[0]


def _build_retriever(
    db: Session,
    request: QueryRequest,
    settings: Settings,
    reranker: LexicalOverlapReranker | None,
) -> HybridRetriever:
    """Build a HybridRetriever wired with the full set of settings-driven
    ranking stages — noisy-source penalty, source authority, diversity —
    so the direct /query endpoint ranks the same way the QueryService path
    (services.py) does. Without this wiring the noisy-source penalty
    defaulted to 0.0 and verbose simulation logs crowded real source files
    out of the top-k."""
    source_authority_config = _source_authority_config(settings)
    diversity_config = DEFAULT_DIVERSITY.model_copy(update={
        "enabled": settings.retrieval_diversity_enabled,
        "max_per_file": settings.retrieval_diversity_max_per_file,
        "max_per_category": settings.retrieval_diversity_max_per_category,
        "graph_expansion_quota": settings.retrieval_diversity_graph_expansion_quota,
        "neighbour_window": settings.retrieval_diversity_neighbour_window,
        "pool_multiplier": settings.retrieval_diversity_pool_multiplier,
    })
    return HybridRetriever(
        db,
        reranker=reranker,
        enable_graph_expansion=request.enable_graph_expansion,
        enable_intent_boost=request.enable_intent_boost,
        enable_doc_type_prior=settings.retrieval_doc_type_prior_enabled,
        noisy_source_path_terms=settings.noisy_source_path_terms,
        noisy_source_penalty=settings.noisy_source_penalty,
        source_authority_config=source_authority_config,
        diversity_config=diversity_config,
        query_expansion_config=build_query_expansion_config(settings),
        graph_expansion_max_depth=settings.graph_expansion_max_depth,
        graph_expansion_dependencies_max_depth=settings.graph_expansion_dependencies_max_depth,
        graph_expansion_max_per_result=settings.graph_expansion_max_per_result,
        graph_expansion_dependencies_max_per_result=settings.graph_expansion_dependencies_max_per_result,
        graph_expansion_dependencies_per_depth_limit=settings.graph_expansion_dependencies_per_depth_limit,
    )


def _run_query(db: Session, project_id: str, request: QueryRequest) -> QueryResponse:
    query_embedding = _load_query_embedding(project_id, request.query)
    settings = get_settings()

    reranker = _build_reranker(request.enable_reranking, settings)
    retriever = _build_retriever(db, request, settings, reranker)
    results = retriever.retrieve(
        request.query, query_embedding, project_id=project_id, top_k=request.top_k,
        filters=VectorSearchFilter(project_id=project_id),
    )

    prompt = PromptBuilder().build(request.query, results)

    return QueryResponse(
        query=request.query,
        messages=prompt.messages,
        included_chunks=prompt.included_chunks,
        excluded_chunk_count=prompt.excluded_chunk_count,
        estimated_total_tokens=prompt.estimated_total_tokens,
        from_cache=False,
    )


@router.post("/{project_id}/query", response_model=QueryResponse)
def query_project(
    project_id: str,
    request: QueryRequest,
    db: Session = Depends(get_db_session),
    api_key = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    settings: Settings = Depends(get_settings),
    query_service: "QueryService" = Depends(get_query_service),
) -> QueryResponse:
    """Return a prompt ready for an LLM."""
    cache = get_shared_query_cache()
    key = cache_key(project_id, request.query, request.top_k, request.enable_graph_expansion, request.enable_reranking, request.enable_intent_boost)

    cached = cache.get(key)
    if cached is not None:
        return cached.model_copy(update={"from_cache": True})

    response = _run_query(db, project_id, request)
    cache.set(key, project_id, response)
    return response


@router.post("/{project_id}/query/stream")
async def query_project_stream(
    project_id: str,
    request: QueryRequest,
    db: Session = Depends(get_db_session),
    api_key = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    settings: Settings = Depends(get_settings),
    query_service: "QueryService" = Depends(get_query_service),
):
    """Stream events during the agent loop."""
    async def event_generator():
        yield {"event": "stage", "data": json.dumps({"stage": "embedding_query"})}
        query_embedding = _load_query_embedding(project_id, request.query)

        yield {"event": "stage", "data": json.dumps({"stage": "vector_and_lexical_search"})}
        reranker = _build_reranker(request.enable_reranking, settings)
        retriever = HybridRetriever(
            db,
            reranker=None,
            enable_graph_expansion=False,
            enable_intent_boost=request.enable_intent_boost,
            query_expansion_config=build_query_expansion_config(get_settings()),
        )

        from semrag.retrieval.fusion import reciprocal_rank_fusion
        from semrag.retrieval.intent import classify_intent
        from semrag.retrieval.query_expansion import expand_query
        from semrag.retrieval.vector_search_adapter import to_retrieval_results
        from semrag.storage.vector.pgvector_store import PgvectorStore

        vector_results = to_retrieval_results(
            db, PgvectorStore(db).search(query_embedding, top_k=20, filters=VectorSearchFilter(project_id=project_id)),
            signal_name="vector",
        )
        lexical_results = retriever._lexical_search.search(
            request.query, top_k=20, filters=VectorSearchFilter(project_id=project_id)
        )

        expansion_lists = {}
        expansion_config = build_query_expansion_config(get_settings())
        if expansion_config.enabled:
            expansions = expand_query(
                request.query,
                intent=classify_intent(request.query),
                config=expansion_config,
            )
            for idx, formulation in enumerate(expansions, start=1):
                exp_results = retriever._lexical_search.search(
                    formulation, top_k=20, filters=VectorSearchFilter(project_id=project_id)
                )
                signal_name = f"lexical_exp{idx}"
                for r in exp_results:
                    r.signal_scores = {signal_name: float(r.fused_score)}
                expansion_lists[signal_name] = exp_results

        fused = reciprocal_rank_fusion(
            {"vector": vector_results, "lexical": lexical_results, **expansion_lists}
        )

        if request.enable_graph_expansion:
            yield {"event": "stage", "data": json.dumps({"stage": "graph_expansion"})}
            from semrag.retrieval.graph_expansion import (
                expand_with_graph_context,
            )
            from semrag.retrieval.intent import classify_graph_direction

            expansions = expand_with_graph_context(
                db,
                retriever._graph_queries,
                fused[:5],
                project_id,
                direction=classify_graph_direction(request.query),
                max_depth=settings.graph_expansion_max_depth,
                dependencies_max_depth=settings.graph_expansion_dependencies_max_depth,
                max_expansions_per_result=settings.graph_expansion_max_per_result,
                dependencies_max_expansions_per_result=settings.graph_expansion_dependencies_max_per_result,
                dependencies_per_depth_limit=settings.graph_expansion_dependencies_per_depth_limit,
            )
            if expansions:
                fused = reciprocal_rank_fusion({"base": fused, "graph_expansion": expansions})

        yield {"event": "stage", "data": json.dumps({"stage": "source_ranking"})}
        fused = apply_source_ranking(
            fused,
            noisy_source_path_terms=settings.noisy_source_path_terms,
            noisy_source_penalty=settings.noisy_source_penalty,
            enable_doc_type_prior=settings.retrieval_doc_type_prior_enabled,
            source_authority_config=_source_authority_config(settings),
        )

        if reranker is not None:
            yield {"event": "stage", "data": json.dumps({"stage": "reranking"})}
            try:
                fused = reranker.rerank(request.query, fused, top_k=request.top_k)
            except Exception:  # noqa: BLE001 — fail-open, never abort retrieval
                logger.exception("reranker failed; continuing with pre-reranker ranking")
                fused = fused[: request.top_k]
        else:
            fused = fused[: request.top_k]

        yield {"event": "stage", "data": json.dumps({"stage": "assembling_prompt"})}
        prompt = PromptBuilder().build(request.query, fused)

        response = QueryResponse(
            query=request.query,
            messages=prompt.messages,
            included_chunks=prompt.included_chunks,
            excluded_chunk_count=prompt.excluded_chunk_count,
            estimated_total_tokens=prompt.estimated_total_tokens,
        )
        yield {"event": "result", "data": response.model_dump_json()}

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )