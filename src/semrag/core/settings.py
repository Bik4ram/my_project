"""
Centralized application settings.

A single pydantic-settings model is the single source of truth for every
runtime tunable in SemRAG. Loaded once at process start (the CLI does it in
`run_local_qa.py`, the API does it in `api/app.py`), and passed to
downstream constructors explicitly rather than via module-level singletons.
The old pattern of `os.environ.get(...)` scattered across modules made
deployment coupling invisible — every env-var reader was its own private
contract, with no schema, no validation, no documentation.

Three reasons this is pydantic-settings rather than a hand-rolled class:

1. Validation. Misconfigured deploys fail at startup with an actionable
   message instead of crashing inside the first request.
2. Type coercion. `int`, `bool`, and comma-separated `list[str]` parse
   cleanly without per-call boilerplate.
3. `.env` support. A standard surface area that an operator can drop a
   `.env` next to, or hand to docker-compose, with no project-specific
   knowledge.

`Settings` is intentionally a `BaseSettings` (mutable singleton) rather
than `@dataclass(frozen=True)` because some fields are derived at
construction (e.g. `embedding_dimension` defaults from the configured
provider) and we want hot-swap during tests via `settings.replace(...)`.
For everything that reads settings, this model is the only allowed source.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All env-var-driven settings, validated at construction time."""

    model_config = SettingsConfigDict(
        env_prefix="SEMRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "postgresql+psycopg2://semrag:semrag@localhost:5432/semrag"
    # Connection pool settings for SQLAlchemy.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_recycle: int = 300  # seconds
    db_pool_timeout: int = 30   # seconds

    # ------------------------------------------------------------------
    # Logging / observability
    # ------------------------------------------------------------------
    # TRACE is project-specific — sits below DEBUG, used for very verbose
    # per-chunk / per-embedding / per-message traces. INFO stays concise
    # for production; DEBUG/TRACE opt into artifact writes and per-event
    # detail. See `core.observability.StageLogger` and `core.debug_artifacts`.
    log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "human"] = "human"
    # Where debug artifacts (manifests, parse traces, chunk metadata, etc.)
    # are written. Must be a writable path; artifact writer creates
    # per-run subdirectories.
    debug_artifacts_dir: str = "./.semrag/debug"
    # How many debug runs to retain. Older runs are pruned after each new
    # run completes to bound disk usage. Set to 0 to disable retention
    # pruning (operator-managed).
    debug_artifact_retention: int = 5

    # ------------------------------------------------------------------
    # Embedding provider
    # ------------------------------------------------------------------
    embedding_provider: Literal["ollama", "openai_compatible", "tfidf"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    embedding_base_url: str = "http://localhost:11434"
    embedding_api_key: str | None = None
    embedding_timeout_seconds: float = 300.0
    # Where to persist the file-based embedding cache (used by
    # JSONFileEmbeddingCache). Embedding cache is independent of provider
    # versioning — a future change to provider requires invalidating it.
    embedding_cache_dir: str = "./.embedding_cache"

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    llm_provider: Literal[
        "ollama",
        "openai",
        "anthropic",
        "openai_compatible",
        "vllm",
        "mock",
    ] = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str | None = None
    # Per-call HTTP timeout. 180 s covers a heavy multi-hop generation
    # from a local 7B-class model (measured ~130 s end-to-end for a
    # 4–6k-token prompt); a 90 s cap made the /answer endpoint fail
    # reproducibly (500, read timeout) on exactly those queries even
    # after the built-in retry. The agent loop is bounded by
    # ``llm_max_iterations``, not a wall-clock budget, so a single slow
    # call no longer blows an overall query budget.
    llm_timeout_seconds: float = 180.0
    # Default 2 (was 3). Three attempts with exponential backoff can
    # burn 7+ seconds waiting on a stuck server before giving up; two
    # attempts is enough to ride out a single transient hiccup.
    llm_max_retries: int = 2
    llm_max_iterations: int = 8
    # Token budget for prompt construction (Part 11's PromptBuilder uses
    # these to decide how much retrieved context fits alongside the
    # system prompt, history, and reserved-for-response headroom).
    llm_context_window: int = 8192
    llm_reserved_for_response: int = 1024
    # Hard cap on tokens the LLM may emit per call. Without this, qwen2.5:7b
    # happily generates 2-3k tokens for an "explain this" prompt, taking
    # 60+ seconds on CPU. 800 is enough for a grounded 4-6 bullet answer
    # with citations — anything more is usually the model looping or
    # expanding the same point. Sent to Ollama as ``options.num_predict``
    # (qwen 7B averages ~30 tok/s, so 800 tokens = ~27s ceiling; the actual
    # answer usually comes in 200-400 tokens).
    llm_num_predict: int = 800
    # Whether to probe the LLM endpoint at startup (fail-fast on bad
    # configuration). Disable in test environments that swap providers
    # via mocks.
    llm_probe_at_startup: bool = True

    # ------------------------------------------------------------------
    # Grounding gate (A1). Post-generation evidence check: the generated
    # answer's source-specific claims (file paths, classes, modules,
    # methods, symbols) are verified against the retrieved evidence that
    # was supplied to generation. Answers that affirmatively reference
    # entities absent from that evidence, or that make source-specific
    # claims with no citation at all, are revised or replaced with a
    # controlled uncertainty response instead of being presented as
    # grounded facts.
    grounding_gate_enabled: bool = True
    # When an answer fails the gate, attempt one constrained
    # regeneration (a second LLM call) that removes unsupported claims.
    # Disable to skip the extra latency and go straight to the
    # controlled response.
    grounding_gate_regenerate: bool = True
    # Hard cap on regeneration attempts per query (each attempt is one
    # extra LLM call).
    grounding_gate_max_revisions: int = 1
    # Shortest acceptable answer text after revision; anything shorter is
    # replaced with the controlled uncertainty response.
    grounding_gate_min_length: int = 40

    # ------------------------------------------------------------------
    # A2 — grounded coding agent
    # ------------------------------------------------------------------
    # Hard cap on diagnose → repair → validate iterations before the
    # coding agent stops safely (spec: default max 3).
    coding_max_repair_attempts: int = 3
    # Master switch for the /v1/projects/{id}/code endpoint + MCP tool.
    coding_enabled: bool = True
    # Max distinct files a single plan may touch (A3 multi-file surface).
    # A2's single-file default was 3; A3 allows coherent multi-file
    # engineering changes up to this bound, still read-before-write and
    # within the declared change surface.
    coding_max_plan_files: int = 6

    # ------------------------------------------------------------------
    # Retrie
    # ------------------------------------------------------------------
    vector_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 10
    reranker: Literal["lexical_overlap", "http", "none"] = "lexical_overlap"
    enable_graph_expansion: bool = True
    graph_expansion_max_depth: int = 1
    # Forward (dependencies) traversal depth. Kept separate from the
    # reverse ``graph_expansion_max_depth`` so transaction-trace queries
    # can span a bounded multi-hop chain without widening the reverse
    # (dependent) walk, which has larger fan-out.
    graph_expansion_dependencies_max_depth: int = 4
    graph_expansion_max_per_result: int = 3
    # Forward (dependencies) expansion bounds, separate from the reverse
    # ``graph_expansion_max_per_result`` so transaction-trace queries can
    # surface deeper chain nodes (sequencer/driver/iface) without widening
    # the larger-fan-out reverse walk. The per-seed cap defaults to the
    # maximum the depth x per-depth bounds permit, so a valid multi-hop
    # trace chain is never silently truncated by the cap.
    graph_expansion_dependencies_max_per_result: int = 16
    graph_expansion_dependencies_per_depth_limit: int = 4
    # RRF constant (Cormack et al., 2009). Tunable on real eval data; not
    # touched unless the eval harness shows a clear win.
    rrf_k: int = 60
    # Apply the query-intent aware rerank boost (see retrieval/intent.py).
    # Disable to compare against the unmodified ranking on real eval queries.
    enable_intent_boost: bool = True
    # Apply the document-type-aware prior (see retrieval/doc_type_prior.py).
    # Pure query-time feature: inspects ``chunk_type`` and ``relative_path``
    # only, no schema change, no re-indexing. Disable to A/B against the
    # unmodified ranking on real eval queries.
    retrieval_doc_type_prior_enabled: bool = True
    # Optional per-category authority overrides for the source-ranking
    # adjustment (RI-02; see retrieval/source_authority.py). Empty means
    # the DEFAULT_SOURCE_AUTHORITY weights apply. Overrides merge over the
    # defaults keyed by category name (e.g. {"uvm": 0.12}); the operator
    # tunes only the categories they care about. Env var is JSON, e.g.
    # SEMRAG_SOURCE_AUTHORITY_OVERRIDES='{"uvm": 0.12}'.
    source_authority_overrides: dict[str, float] = Field(default_factory=dict)
    # Optional override of the authority magnitude cap (default 0.5).
    source_authority_max_boost: float | None = None
    # Multiplicative penalty (0.0 = off, 1.0 = drop entirely) applied to
    # chunks whose relative_path matches any entry in
    # ``noisy_source_path_terms`` (e.g. simulation logs). Verbose transcript
    # files systematically over-match natural-language queries through both
    # the vector and lexical channels, crowding out code chunks in the fused
    # top-k; this downweights them relative to real source files.
    noisy_source_path_terms: tuple[str, ...] = ("/log/",)
    noisy_source_penalty: float = 0.4
    # Diversity-aware selection stage (RI-03; see
    # retrieval/ranking/diversity.py). After every ranking score is final
    # and before the top-k slice, the pipeline caps how many chunks a
    # single file / document category can contribute and suppresses
    # adjacent windows of the same file, so the LLM context holds
    # complementary sources instead of many near-identical chunks. All
    # knobs are config-driven; disable to compare against the
    # pre-diversity ranking. ``pool_multiplier`` widens the reranker's
    # candidate pool (top_k * multiplier) when a reranker is configured.
    # ``graph_expansion_quota`` bounds how many graph-expanded candidates
    # may exceed the per-category cap (0 disables the exemption).
    retrieval_diversity_enabled: bool = True
    retrieval_diversity_max_per_file: int = 3
    retrieval_diversity_max_per_category: int = 6
    retrieval_diversity_graph_expansion_quota: int = 3
    retrieval_diversity_neighbour_window: int = 25
    retrieval_diversity_pool_multiplier: int = 3
    # RI-07 metadata tie-break in the lexical-overlap reranker. When
    # enabled, near-tied candidates (within ``near_tie_epsilon``) are
    # deterministically reordered to prefer a chunk whose ``symbol_name``
    # matches a query term — but ONLY for queries the existing intent /
    # structural-query logic identifies as code-navigation (ARCHITECTURE
    # intent or a structural query). Purely a bounded tie-break: it can
    # never displace a materially better relevance score. Disable to
    # compare against the pre-RI-07 reranking.
    retrieval_rerank_metadata_tiebreak: bool = True
    # Near-tie window for the RI-07 metadata tie-break (see above).
    # Default 1e-4 is smaller than the smallest single-leg RRF adjacent-
    # rank delta (~1.5e-4 at rank 20), so only exact / near-exact score
    # ties are ever affected.
    retrieval_rerank_near_tie_epsilon: float = 1e-4
    # Deterministic query expansion (RI-05; see retrieval/query_expansion.py).
    # When enabled, the lexical leg of retrieval is run once for the
    # original query and again for up to ``max_expansions`` additional
    # formulations (original query + intent-specific repository
    # terminology), and the results are fused through the existing RRF.
    # The vector leg is untouched (no extra embeddings = no added network
    # latency); the original query is always searched. All knobs are
    # config-driven; disable to compare against the unmodified retrieval.
    # ``timeout_ms`` is reserved for a future external/LLM expansion
    # mechanism and is unused by the deterministic implementation.
    query_expansion_enabled: bool = True
    query_expansion_max_expansions: int = 3
    query_expansion_max_expansion_length: int = 80
    query_expansion_timeout_ms: float | None = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_keys: str = "dev-local-key"
    rate_limit_per_minute: int = 60
    query_cache_ttl_seconds: int = 300
    embedding_provider_dir: str = "/tmp/semrag_embedding_providers"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = "development"
    trusted_hosts: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=list)
    max_body_size: int = 5 * 1024 * 1024  # 5 MB
    enable_gzip: bool = False
    # Security headers to be set on all responses.
    # Example: {"X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff"}
    security_headers: dict[str, str] = Field(default_factory=dict)
    # Store type for API keys: memory, redis, database. Defaults to
    # ``database`` so a fresh production deploy never silently boots
    # with a per-replica in-memory store (the in-memory store loses
    # every key on restart and is invisible to peer replicas — the
    # class of bug that motivated the secure API key store project).
    # The in-memory store is still available for tests / dev by
    # setting ``SEMRAG_API_KEY_STORE=memory``.
    api_key_store: Literal["memory", "redis", "database"] = "database"
    # Backend for rate limiting: memory, redis
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    # List of allowed root directories for projects. Empty list means no restriction.
    # Each path must be absolute and will be checked for symlinks and traversal.
    project_roots: list[str] = Field(default_factory=list)
    # Expiration time for API keys in days (if supported by the store)
    api_key_expire_days: int = 30
    # Whether to reject default development keys in production
    reject_dev_keys_in_production: bool = True

    # ------------------------------------------------------------------
    # Query Cache
    # ------------------------------------------------------------------
    # Backend for query caching: memory or redis.
    query_cache_backend: Literal["memory", "redis"] = "memory"

    # ------------------------------------------------------------------
    # Embedding Cache
    # ------------------------------------------------------------------
    # Backend for embedding cache: memory, jsonfile, or redis.
    embedding_cache_backend: Literal["memory", "jsonfile", "redis"] = "jsonfile"

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    max_concurrent_parse_workers: int = 4
    max_concurrent_hash_workers: int = 8

    # ------------------------------------------------------------------
    # Persistent stores (Wave 3 enterprise edition)
    # ------------------------------------------------------------------
    # Backend for the job store. ``memory`` (single-replica, per-process)
    # is the dev/test default; ``database`` persists across restarts and
    # replicas, which is the only acceptable configuration for
    # production. A fresh production deploy that booted into ``memory``
    # would silently lose queued/running jobs on every restart and
    # duplicate them across replicas — the class of bug the persistent
    # job store project shipped to fix.
    job_store_backend: Literal["memory", "database"] = "database"
    # Redis connection URL (used when any cache / rate-limiter / embedding
    # cache backend is set to ``redis``). Format: redis://host:port/db.
    # Optional auth via username:password@. Empty string = no Redis configured.
    redis_url: str = ""
    # How often the worker process polls the persistent job store for new
    # QUEUED jobs. Lower numbers react faster at the cost of more DB load;
    # 1.0s matches typical worker-queue trade-offs.
    worker_poll_interval_seconds: float = 1.0
    # When true, the API process also runs an in-process worker that
    # polls the persistent job store. Useful for single-node dev and
    # small deployments where running two processes is overkill. In a
    # multi-replica production deployment, set this to false and run
    # a dedicated worker (the `semrag-worker` entrypoint).
    api_runs_worker: bool = True

    # ------------------------------------------------------------------
    # Observability — Prometheus / OpenTelemetry (Wave 3)
    # ------------------------------------------------------------------
    # Prometheus /metrics endpoint. Toggle off in tests that don't want
    # the registry populated; on in production so a scrape target sees
    # HTTP/RAG/Indexing/Database/Cache/Rate-limit metrics.
    metrics_enabled: bool = True
    # Path the /metrics endpoint is served at. Default matches the
    # Prometheus convention; Kubernetes manifests + Grafana datasource
    # defaults assume /metrics.
    metrics_path: str = "/metrics"
    # OpenTelemetry tracing toggle. When off, the tracer is a no-op and
    # the OTLP exporter never dials out (useful for tests + offline dev).
    tracing_enabled: bool = False
    # OTLP endpoint. Set to an OTel Collector URL (e.g. otel-collector:4317
    # for gRPC, 4318 for HTTP) when tracing_enabled is true. The scheme is
    # selected from the URL: http+ → http/protobuf, otherwise gRPC.
    otlp_endpoint: str = "http://localhost:4317"
    # Logical service name exported via the OTLP resource attribute.
    otel_service_name: str = "semrag"
    # Sampling ratio. 1.0 = trace every span, 0.1 = 10% (production-
    # realistic), 0 = disabled even if tracing_enabled is true (kept
    # distinct so toggling the feature vs the sampling rate are
    # orthogonal switches).
    trace_sample_ratio: float = 1.0
    # Whether to log full exception tracebacks at ERROR level. Keep
    # enabled in dev/staging; turn off in production if the log
    # pipeline only needs the exception class + message.
    log_exceptions: bool = True
    # How many characters of a logged message / payload to keep after
    # redaction. Long content is truncated with an ellipsis to bound
    # log volume; raise for debugging.
    log_max_payload_chars: int = 4096

    @field_validator("api_keys")
    @classmethod
    def _strip_api_keys(cls, raw: str) -> str:
        # Comma-separated; trim each entry, drop empties, dedupe while
        # preserving order so rate-limit iteration is deterministic.
        seen: set[str] = set()
        out: list[str] = []
        for entry in raw.split(","):
            k = entry.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
        return ",".join(out) or "dev-local-key"

    @property
    def api_keys_set(self) -> set[str]:
        """The configured API keys as a set, for O(1) lookup."""
        return {k for k in self.api_keys.split(",") if k}

    def api_key_list(self) -> list[str]:
        """The configured API keys as an ordered list."""
        return [k for k in self.api_keys.split(",") if k]

    def debug_writer(self) -> "DebugArtifactWriter":
        """Factory for the debug artifact writer. Active only when the
        configured log level is DEBUG or TRACE."""
        from semrag.core.debug_artifacts import DebugArtifactWriter

        return DebugArtifactWriter(
            root_dir=self.debug_artifacts_dir,
            log_level=self.log_level,
            retention=self.debug_artifact_retention,
        )

    @field_validator("trusted_hosts", "cors_origins", "project_roots", mode="before")
    @classmethod
    def _list_from_csv_or_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("project_roots")
    @classmethod
    def _validate_project_roots(cls, v: list[str]) -> list[str]:
        # Ensure each path is absolute and normalized.
        from pathlib import Path
        result = []
        for path in v:
            p = Path(path)
            if not p.is_absolute():
                raise ValueError(f"Project root must be an absolute path: {path}")
            # We don't check existence here because the root might not exist yet.
            # But we will resolve symlinks and check for traversal at use time.
            result.append(str(p))
        return result

    # We'll add a model validator to check the api_keys and environment together
    @model_validator(mode="after")
    def _check_dev_keys_after(self) -> "Settings":
        if self.environment == "production" and self.reject_dev_keys_in_production:
            # The default development key is "dev-local-key"
            # We consider a key to be the default if it is exactly "dev-local-key" or if it is in the list and is the default.
            # We'll split the api_keys and check if any of them is the default.
            keys = [k.strip() for k in self.api_keys.split(",") if k.strip()]
            if "dev-local-key" in keys:
                raise ValueError(
                    "Default development API key 'dev-local-key' is not allowed in production. "
                    "Please set SEMRAG_API_KEYS to a custom key or keys."
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Tests can clear this with
    `get_settings.cache_clear()` after mutating env vars."""
    return Settings()


def reset_settings_cache() -> Settings:
    """Force a fresh load from environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()


def env_flag(name: str, default: bool = False) -> bool:
    """Helper for ad-hoc readers that haven't migrated to Settings yet —
    preferred path is always to add the field to Settings. Used only by
    bootstrap code."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")