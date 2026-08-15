"""QueryTrace — serializable trace covering all 22 pipeline stages.

Every public method in this module is a no-op when the log level is not
DEBUG or TRACE, so production has zero overhead.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class QueryTraceStage(BaseModel):
    """Timing and data for a single pipeline stage."""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0
    data: dict = Field(default_factory=dict)


class QueryTrace(BaseModel):
    """Complete trace of a single query through all 22 pipeline stages.

    Designed to be:
    - Built incrementally as each stage completes
    - Serialized to JSON via ``model_dump(mode="json")``
    - Stored under ``artifacts/query_traces/`` by ``DebugArtifactWriter``
    """

    model_config = {"extra": "ignore"}
    run_id: str = ""
    project_id: str = ""
    query_text: str = ""
    timestamp: float = 0.0
    total_duration_ms: float = 0.0

    # Stages 1-2: Query received + preprocessing
    preprocessing: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 3: Intent classification
    intent_classification: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 4: Query expansion (future slot)
    query_expansion: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 5: Query embedding
    embedding: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stages 6-15: Full retrieval pipeline
    retrieval: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 16: Context assembly
    context_assembly: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 17: Prompt construction
    prompt_construction: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 18: Token estimation
    token_estimation: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stages 19-20: LLM call + response
    llm_stage: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 21: Citation extraction
    citation_extraction: QueryTraceStage = Field(default_factory=QueryTraceStage)

    # Stage 22: Final answer
    final_answer: str | None = None
    iterations_used: int = 0
    hit_iteration_limit: bool = False
    messages: list[dict] = Field(default_factory=list)
