# Graph Report - /home/bikram/projects/semrag  (2026-07-29)

## Corpus Check
- 235 files · ~412,948 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1622 nodes · 4150 edges · 113 communities (110 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 175 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Chunking Pipeline
- Prompting & Agent Tools
- Scanner & Storage
- Embedding Pipeline
- Evaluation Harness
- Core Services
- API & Database Setup
- Agent Loop & LLM
- Debug & Services Init
- Retrieval Intent Classification
- Observability & Ollama LLM
- Agent Tools (Files & Search)
- Debug Artifacts & Settings
- Agent Tools Protocol
- Scanner Change Detection
- VHDL Parser
- LLM Factory & Providers
- Query Deduplication
- Python Parser (TCL)
- Graph Expansion & Retrieval
- Markdown & Document Parsers
- PostgreSQL Graph Queries
- Observability Logging
- Scanner Classification
- Vector Store Interfaces
- Neo4j Graph Store
- API Auth & Rate Limit
- TF-IDF Embedding Provider
- Reranking
- OpenAI-Compatible Embeddings
- Core Settings
- IR Builder & ID Scheme
- Scanner Manifest & Orchestrator
- Ollama LLM Client
- API Query Cache
- Parser Models & Structured Data
- Hybrid Retriever Pipeline
- API Config & Projects Router
- LLM Interfaces (OpenAI Compatible)
- IR Symbol Tables
- TCL Parser
- Settings Factory & Services Init
- Citation Validator
- PDF Parser
- Parser Interfaces & Pipeline
- Regression Log Parser
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- IR Pipeline Integration
- Structured Data Parsers (JSON/XML)
- Lexical Search (Postgres)
- Neo4j Graph Store Tests
- FastAPI App & Routers
- Graph Store Interfaces
- Parser Registry
- JSON File Manifest Store
- Mock LLM
- Scanner Hashing Tests
- Full Pipeline Integration Tests
- Core Init
- Package Init

## God Nodes (most connected - your core abstractions)
1. `FileRecord` - 71 edges
2. `ParseResult` - 53 edges
3. `Settings` - 45 edges
4. `RetrievalResult` - 45 edges
5. `PgvectorStore` - 39 edges
6. `IRNode` - 38 edges
7. `QueryService` - 36 edges
8. `IRGraph` - 35 edges
9. `IRGraphRepository` - 35 edges
10. `IRBuilder` - 34 edges

## Surprising Connections (you probably didn't know these)
- `Console` --uses--> `InMemoryEmbeddingCache`  [INFERRED]
  run_local_qa.py → src/semrag/ingestion/embedding/cache.py
- `Console` --uses--> `IngestionService`  [INFERRED]
  run_local_qa.py → src/semrag/services/services.py
- `Console` --uses--> `QueryService`  [INFERRED]
  run_local_qa.py → src/semrag/services/services.py
- `_build_ingestion_service()` --calls--> `IngestionService`  [INFERRED]
  run_local_qa.py → src/semrag/services/services.py
- `_build_query_service()` --calls--> `QueryService`  [INFERRED]
  run_local_qa.py → src/semrag/services/services.py

## Import Cycles
- None detected.

## Communities (113 total, 3 thin omitted)

### Community 0 - "Chunking Pipeline"
Cohesion: 0.07
Nodes (53): GenericWindowChunker, Generic sliding-window chunker. Used wherever no bespoke, structure-aware…, chunk_id(), extract_lines(), Deterministic chunk IDs and the shared line-range extraction helper every…, 1-indexed, inclusive on both ends — matching the line-number convention Part…, Chunker, ChunkerRegistry (+45 more)

### Community 1 - "Prompting & Agent Tools"
Cohesion: 0.06
Nodes (56): format_chunk_with_citation(), format_citation_header(), Citation formatting. Every chunk included in a prompt is prefixed with a…, assemble_context(), Context assembly. Greedy packing by fused-score order: results are already…, Returns (included_results, formatted_context_text, used_tokens). ``max_chunks``…, AssembledPrompt, ChatMessage (+48 more)

### Community 2 - "Scanner & Storage"
Cohesion: 0.06
Nodes (48): DeclarativeBase, build_blob_info_map(), bulk_commit_attribution(), _CommitAttribution, get_repository_metadata(), GitCommandError, is_git_repository(), list_tracked_blobs() (+40 more)

### Community 3 - "Embedding Pipeline"
Cohesion: 0.06
Nodes (38): JSONFileEmbeddingCache, Path, Embedding cache implementations. Same two-tier pattern as Part 2's…, One JSON file per (model_name, dimension) combination — a cache entry from one…, text_hash(), EmbeddingCache, EmbeddingProvider, Protocol (+30 more)

### Community 4 - "Evaluation Harness"
Cohesion: 0.10
Nodes (36): EvaluationHarness, Evaluation harness. Runs each GoldenQuery through the real Part 10…, ndcg_at_k(), precision_at_k(), Retrieval quality metrics. Standard information-retrieval formulas, not…, Binary relevance NDCG: DCG@k = sum(rel_i / log2(i+1)) for the actual ranking,…, recall_at_k(), reciprocal_rank() (+28 more)

### Community 5 - "Core Services"
Cohesion: 0.09
Nodes (28): IngestionService, Composes a full index run: scan → parse → IR → chunk → embed → persist., _CapturingLLM, _NoOpCache, _NoOpEmbeddings, Service layer — top-level error handling and shape contracts. Heavy integration…, Bad root should fail with NotADirectoryError before any I/O. The constructor…, Constructor should be pure — no DB or LLM calls happen until `ask` is invoked. (+20 more)

### Community 6 - "API & Database Setup"
Cohesion: 0.09
Nodes (30): Run migrations in 'offline' mode. This configures the context with just a URL…, Run migrations in 'online' mode. In this scenario we need to create an Engine…, run_migrations_offline(), run_migrations_online(), sessionmaker, Per-request database session dependency. One session per request, closed at the…, build_engine(), build_session_factory() (+22 more)

### Community 7 - "Agent Loop & LLM"
Cohesion: 0.12
Nodes (30): ParsedAssertion, ParsedInstantiation, ParsedModule, ParsedPackage, ParsedParameter, ParsedPort, ParsedProceduralBlock, BaseModel (+22 more)

### Community 8 - "Debug & Services Init"
Cohesion: 0.13
Nodes (20): Convenience wrapper — keeps call sites shorter than `logger.log(TRACE_LEVEL,…, trace(), AgentLoop, AgentResult, Agent loop. Standard tool-calling ReAct-style loop: call the LLM with the…, ToolRegistry, CompletionResult, LLMProvider (+12 more)

### Community 9 - "Retrieval Intent Classification"
Cohesion: 0.11
Nodes (22): new_run_id(), Run id is a millisecond timestamp + short random suffix, e.g.…, InMemoryEmbeddingCache, IRBuilder, Scanner, SearchCodebaseTool, High-level services that compose primitives into business operations., DebugSummary (+14 more)

### Community 10 - "Observability & Ollama LLM"
Cohesion: 0.13
Nodes (20): BaseException, Logger, count(), debug(), _emit_metric(), _format_tag(), _log_stage(), Metrics (+12 more)

### Community 11 - "Agent Tools (Files & Search)"
Cohesion: 0.11
Nodes (28): apply_intent_boost(), boost_for_intent(), classify_intent(), _intent_priority(), str, QueryIntent, Query-intent aware retrieval rerank boost. Why this exists: a pure RRF fusion…, Apply the intent-keyed boost to each result's ``fused_score``. A multiplier in… (+20 more)

### Community 12 - "Debug Artifacts & Settings"
Cohesion: 0.10
Nodes (19): DebugArtifactWriter, _is_active(), Any, Path, Debug artifact writer. Layout:: <debug_artifacts_dir>/ <run-id>/…, Prune older runs after a successful run. `retention=0` skips pruning entirely., Allocate the per-run directory tree. Idempotent: re-calling with the same id is…, Generic JSON writer. `category` is one of the subdirs above; `name` is the file… (+11 more)

### Community 13 - "Agent Tools Protocol"
Cohesion: 0.11
Nodes (16): ListFilesTool, Protocol, Session, Agent tools. Three real tools, each wrapping something already built in this…, ReadFileTool, Tool, Path, test_list_files_tool_against_live_db() (+8 more)

### Community 14 - "Scanner Change Detection"
Cohesion: 0.12
Nodes (15): detect_changes(), _is_unchanged(), Change detection between consecutive scans. The core operation is a three-way…, Project Scanner subsystem (Part 2). See interfaces.py for the public API., ManifestStore, Protocol, Public interface of the scanner subsystem. Every other plane (workers, API,…, Persistence boundary for scan manifests. The scanner subsystem only ever needs… (+7 more)

### Community 15 - "VHDL Parser"
Cohesion: 0.15
Nodes (21): Match, Pattern, ParsedVhdlArchitecture, ParsedVhdlEntity, ParsedVhdlGeneric, ParsedVhdlPort, BaseModel, Structured payload for the VHDL parser. (+13 more)

### Community 16 - "LLM Factory & Providers"
Cohesion: 0.13
Nodes (23): build_llm(), LLMConfigurationError, _probe(), RuntimeError, LLM provider factory. A single `build_llm(settings)` returns an `LLMProvider`…, Build the configured LLM provider., One minimal request to confirm the provider is reachable. Catches URL typos,…, OpenAILLM (+15 more)

### Community 17 - "Query Deduplication"
Cohesion: 0.11
Nodes (27): _deduplicate_chunks(), _is_repetitive_noise(), _line_signature(), True when a chunk is mostly the same line repeated (typical of log/coverage…, Drop chunks that are pure log noise, and within the same source file collapse…, Coarse fingerprint: the hash of each non-trivial line, in order. Two chunks…, Tests for the chunk hygiene helpers and structural file-index path in…, With enough non-noise chunks to keep, the filter strips them. (+19 more)

### Community 18 - "Python Parser (TCL)"
Cohesion: 0.16
Nodes (18): AsyncFunctionDef, ClassDef, expr, FunctionDef, Module, ParsedClass, ParsedFunction, ParsedImport (+10 more)

### Community 19 - "Graph Expansion & Retrieval"
Cohesion: 0.15
Nodes (17): expand_with_graph_context(), Session, Graph-based context expansion. Pulls in chunks structurally related to the top…, BaseModel, RetrievalResult, Hybrid retrieval pipeline. Assembles every piece this part built into the flow…, Session, Adapter from Part 6's `VectorSearchResult` to this part's `RetrievalResult`.… (+9 more)

### Community 20 - "Markdown & Document Parsers"
Cohesion: 0.12
Nodes (17): MarkdownFileStructure, ParsedHeading, BaseModel, Path, Markdown parser. Extracts heading hierarchy (ATX `#`-style headings, the…, DiagnosticSeverity, ParseDiagnostic, BaseModel (+9 more)

### Community 21 - "PostgreSQL Graph Queries"
Cohesion: 0.16
Nodes (19): GraphPathResult, PostgresGraphQueries, Session, Shortest path by edge count (unweighted BFS via recursive CTE — the first time…, Everything `start_node_id` (transitively) depends on, following edges in their…, Everything that (transitively) depends on `start_node_id` — edges followed in…, _edge(), _node() (+11 more)

### Community 22 - "Observability Logging"
Cohesion: 0.09
Nodes (17): LogRecord, configure_logging(), get_logger(), _HumanFormatter, is_trace_enabled(), _JsonFormatter, Install handlers/formatters exactly once. Idempotent — calling twice does not…, Standard project idiom: `logger = get_logger(__name__)` Always returns a usable… (+9 more)

### Community 23 - "Scanner Classification"
Cohesion: 0.15
Nodes (21): PathSpec, classify(), is_binary_extension(), looks_like_binary_content(), Extension/path-based file classification. This is deliberately a fast, cheap…, Cheap binary sniff on the first chunk of a file, used as a second line of…, Classify a file given its repo-relative path (posix-style separators expected —…, _load_gitignore_spec() (+13 more)

### Community 24 - "Vector Store Interfaces"
Cohesion: 0.15
Nodes (13): Protocol, Vector store interface. Every other plane depends on this Protocol, not on…, VectorStore, BaseModel, Vector store data models. `VectorChunk` is deliberately not the same object as…, VectorChunk, VectorSearchFilter, VectorSearchResult (+5 more)

### Community 25 - "Neo4j Graph Store"
Cohesion: 0.13
Nodes (9): GraphPathResult, IRGraph, Neo4jGraphStore, Neo4j-backed GraphStore. **Status: written against the real `neo4j` Python…, Idempotent — safe to call on every startup, matching Alembic's `upgrade head`…, Upsert-and-prune, same philosophy as `IRGraphRepository.save()` (Part 5) and…, neo4j_store(), fixture (+1 more)

### Community 26 - "API Auth & Rate Limit"
Cohesion: 0.15
Nodes (16): API key authentication. Bearer-token style (Authorization: Bearer key), checked…, require_api_key(), get_api_keys(), get_rate_limit_per_minute(), API configuration. Same env-var-driven pattern as Part 5's session.py's…, enforce_rate_limit(), get_rate_limiter(), Rate limiting. In-memory sliding window, keyed by API key - deliberately not… (+8 more)

### Community 27 - "TF-IDF Embedding Provider"
Cohesion: 0.13
Nodes (12): TF-IDF + LSA embedding provider. **Why this exists, stated plainly:** Part 1…, Persists the fitted vectorizer + SVD to disk. Necessary specifically because…, Must be called once on a representative batch of the project's chunk text…, TfidfEmbeddingProvider, test_deterministic_given_same_fit_and_input(), test_embeddings_are_unit_normalized(), test_embeddings_have_configured_dimension(), test_fit_on_empty_corpus_raises() (+4 more)

### Community 28 - "Reranking"
Cohesion: 0.19
Nodes (10): HTTPRerankerClient, LexicalOverlapReranker, RuntimeError, Jaccard similarity between query tokens and each candidate's tokens, used to…, Status: real client code against the Cohere/TEI `/rerank` contract, not run…, RerankerError, _mock_response(), _result() (+2 more)

### Community 29 - "OpenAI-Compatible Embeddings"
Cohesion: 0.15
Nodes (15): skipif, EmbeddingProviderError, OpenAICompatibleEmbeddingProvider, RuntimeError, OpenAI-compatible embedding provider. **Status: real, complete HTTP client code…, _mock_response(), Tests the OpenAICompatibleEmbeddingProvider's own logic (retry/backoff, out-of-…, The one test in this file that actually verifies real-world compatibility, not… (+7 more)

### Community 30 - "Core Settings"
Cohesion: 0.12
Nodes (19): env_flag(), get_settings(), Centralized application settings. A single pydantic-settings model is the…, Cached singleton. Tests can clear this with `get_settings.cache_clear()` after…, Force a fresh load from environment (used by tests)., Helper for ad-hoc readers that haven't migrated to Settings yet — preferred…, reset_settings_cache(), clean_env() (+11 more)

### Community 31 - "IR Builder & ID Scheme"
Cohesion: 0.20
Nodes (9): IR builder. Two-pass algorithm, and it must stay two-pass: Pass 1 creates every…, edge_id(), py_module_id(), Deterministic ID generation. Every ID is a pure function of stable inputs…, _slug(), sv_file_id(), sv_module_id(), sv_package_id() (+1 more)

### Community 32 - "Scanner Manifest & Orchestrator"
Cohesion: 0.18
Nodes (16): InMemoryManifestStore, In-memory / JSON-file ManifestStore implementations. Neither of these is the…, Process-lifetime only. Used in unit tests and short-lived scripts., Path, test_chunking_is_idempotent_across_runs(), test_full_pipeline_produces_chunks_across_languages(), _write(), fake_project() (+8 more)

### Community 33 - "Ollama LLM Client"
Cohesion: 0.16
Nodes (13): OllamaLLM, Ollama accepts the same `messages`/`tools` shape as OpenAI; pass through only…, Flatten Ollama's `message` into the OpenAI-shaped dict the shared parser…, Ollama `/api/chat` client. Reuses the OpenAI-compatible base for retries,…, _mock_response(), OllamaLLM native `/api/chat` client. Covers the differences from…, test_complete_captures_ollama_token_usage(), test_complete_hits_api_chat_endpoint() (+5 more)

### Community 34 - "API Query Cache"
Cohesion: 0.19
Nodes (12): get_query_cache_ttl_seconds(), cache_key(), get_shared_query_cache(), Query result cache. TTL-based, in-memory, keyed by a hash of (project_id, query…, A module-level singleton, same pattern as rate_limit.py's limiter - state…, TTLQueryCache, _response(), test_cache_key_differs_by_query() (+4 more)

### Community 35 - "Parser Models & Structured Data"
Cohesion: 0.23
Nodes (13): ParseResult, Generic envelope every language parser returns. `payload` holds the language-…, CsvParser, CsvStructure, _fail(), JsonYamlStructure, BaseModel, Path (+5 more)

### Community 36 - "Hybrid Retriever Pipeline"
Cohesion: 0.19
Nodes (12): HybridRetriever, Session, Protocol, Reranking. **Same limitation class as Part 9's embedding provider, same two-…, Reranker, _tokenize(), Path, The real proof of Part 10: a multi-file project run through the entire pipeline… (+4 more)

### Community 37 - "API Config & Projects Router"
Cohesion: 0.23
Nodes (14): get, get_embedding_provider_dir(), Where each project's fitted TfidfEmbeddingProvider is persisted (see…, index_project(), list_files(), post, Session, Project indexing and file listing endpoints. POST… (+6 more)

### Community 38 - "LLM Interfaces (OpenAI Compatible)"
Cohesion: 0.19
Nodes (9): Response, StreamChunk, Ollama native chat completion client. Ollama's `/api/chat` endpoint is the only…, LLMProviderError, _post_with_retries_shared(), RuntimeError, OpenAI-compatible chat completion client. Targets the standard `POST…, Centralized retry+timeout loop with structured metric emission. Returns the… (+1 more)

### Community 39 - "IR Symbol Tables"
Cohesion: 0.19
Nodes (11): _dotted_path_suffixes(), Project-wide symbol tables. Built in one pass over every node before any edge…, Returns (resolved_id, ambiguous, candidates). resolved_id is None if there were…, `src/semrag/ingestion/scanner/hashing.py` ->…, SymbolTables, _module_node(), test_dotted_path_suffixes_strip_src_root(), test_dotted_path_suffixes_without_src_root() (+3 more)

### Community 40 - "TCL Parser"
Cohesion: 0.27
Nodes (12): ParsedProc, ParsedSourceRef, BaseModel, Structured payload for the TCL parser., TclFileStructure, Path, TCL parser. Deliberately NOT a full TCL grammar parser. TCL's actual grammar is…, TclParser (+4 more)

### Community 41 - "Settings Factory & Services Init"
Cohesion: 0.14
Nodes (10): BaseSettings, field_validator, _build_ingestion_service(), All env-var-driven settings, validated at construction time., Settings, build_embedding_cache(), Default file-backed cache. Tests can swap in `InMemoryEmbeddingCache`., _project_file_index() (+2 more)

### Community 42 - "Citation Validator"
Cohesion: 0.25
Nodes (11): CitationValidationResult, BaseModel, Citation validator. Real faithfulness/hallucination measurement (Part 1's…, validate_citations(), _chunk(), test_all_citations_valid(), test_citation_without_line_numbers(), test_empty_context_makes_every_citation_fabricated() (+3 more)

### Community 43 - "PDF Parser"
Cohesion: 0.26
Nodes (12): ParsedPdfPage, ParsedPdfTable, PdfFileStructure, PdfParser, BaseModel, Path, PDF parser. Uses pdfplumber for both text and table extraction — the two things…, _build_fixture_pdf() (+4 more)

### Community 44 - "Parser Interfaces & Pipeline"
Cohesion: 0.18
Nodes (8): Parser, ParserRegistry, Path, Protocol, Parser plugin interface and registry. Design decision: parsers are dispatched…, Contract every language-specific parser implements., Tries each registered parser for this file's category in order, returning the…, Default parser pipeline assembly. This is the single place that wires concrete…

### Community 45 - "Regression Log Parser"
Cohesion: 0.24
Nodes (11): BaseModel, Path, Regression/simulation log parser. Deliberately generic, and deliberately scoped…, RegressionLogParser, RegressionLogSummary, Data models for the Project Scanner subsystem. These are the objects the…, Path, _record() (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.26
Nodes (11): OpenAICompatibleLLM, OpenAI-shape chat completions. Subclasses override `_endpoint()` and…, _mock_response(), _sse_lines(), test_completion_with_tool_calls(), test_empty_lines_and_malformed_json_are_skipped_not_fatal(), test_malformed_response_raises_clear_error(), test_retries_then_succeeds() (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.26
Nodes (8): IRGraphRepository, Session, _node(), test_ambiguous_edge_preserves_candidate_ids(), test_resaving_prunes_removed_nodes_and_edges(), test_resaving_updates_existing_node_attributes(), test_save_and_load_roundtrip(), test_two_projects_do_not_interfere()

### Community 48 - "Community 48"
Cohesion: 0.36
Nodes (11): PgvectorStore, Session, _chunk(), _ensure_project(), _random_vector(), test_delete_by_project_removes_only_that_projects_chunks(), test_delete_chunks_removes_them_from_search(), test_search_respects_chunk_type_filter() (+3 more)

### Community 49 - "Community 49"
Cohesion: 0.21
Nodes (8): IREdge, IREdgeKind, IRNodeKind, BaseModel, str, Intermediate Representation (IR) data models. Scope of this IR, stated…, IRGraph persistence. Same upsert-and-prune pattern as `manifest_store.py`'s…, Reconstructs an IRGraph from stored rows — used by tests to verify round-trip…

### Community 50 - "Community 50"
Cohesion: 0.27
Nodes (11): ExcelFileStructure, ExcelParser, ParsedSheet, BaseModel, Path, Excel parser. Uses openpyxl (read-only mode for memory efficiency on large…, Path, _record() (+3 more)

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (11): ParsedShellFunction, ParsedShellSourceRef, BaseModel, Path, Shell script parser. Same regex-structural category as TCL: shell function…, ShellFileStructure, ShellParser, Path (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.22
Nodes (12): extract_raw_dependency_refs(), Lightweight, regex-based dependency reference extraction. Explicitly NOT a…, Returns a de-duplicated, order-preserved list of raw reference strings found in…, FileCategory, str, Coarse classification used for downstream routing (which parser subpackage in…, test_dedup_preserves_order(), test_extract_python_imports() (+4 more)

### Community 53 - "Community 53"
Cohesion: 0.27
Nodes (11): Reciprocal Rank Fusion (RRF). RRF combines ranked lists using each item's *rank…, `result_lists` maps a signal name (e.g. "vector", "lexical") to its ranked…, reciprocal_rank_fusion(), Shared models for the retrieval plane. `RetrievalResult` is deliberately…, _result(), test_empty_lists_produce_empty_result(), test_fused_score_is_deterministic_and_positive(), test_item_appearing_in_both_lists_outranks_single_list_item() (+3 more)

### Community 54 - "Community 54"
Cohesion: 0.28
Nodes (12): EventSourceResponse, get_db_session(), Session, _load_query_embedding(), post, Session, query_project(), query_project_stream() (+4 more)

### Community 55 - "Community 55"
Cohesion: 0.21
Nodes (7): build_embedding_provider(), EmbeddingConfigurationError, RuntimeError, Embedding provider factory — single entry point for CLI and API., EmbeddingProviderError, OllamaEmbeddingProvider, RuntimeError

### Community 56 - "Community 56"
Cohesion: 0.23
Nodes (8): is_structural_query(), True when the user is asking about the *shape* of the project, not a specific…, _filter_for_structural(), QueryService, For architecture / overview questions: strip chunks that come from log,…, Composes a query: embed → retrieve → prompt → agent → answer., Run retrieval → prompt assembly → agent loop → final answer. Why seed the…, Non-agent retrieval path — used by API clients that only want chunks, not an…

### Community 57 - "Community 57"
Cohesion: 0.32
Nodes (9): _build_query_service(), cmd_ask(), cmd_index(), Console, _format_debug(), main(), Command-line entrypoint for the full indexing + query pipeline. Both `index`…, User-facing output. Stdlib logging handles observability; this is purely for… (+1 more)

### Community 58 - "IR Pipeline Integration"
Cohesion: 0.38
Nodes (10): End-to-end convenience wiring: scan -> parse every scannable file -> build IR.…, scan_parse_and_build_ir(), build_fixture_project(), Path, End-to-end test: a real multi-file fixture project on disk, run through the…, test_ambiguous_module_name_across_files_is_flagged(), test_end_to_end_ir_graph(), test_node_ids_are_stable_across_rebuilds() (+2 more)

### Community 59 - "Structured Data Parsers (JSON/XML)"
Cohesion: 0.42
Nodes (11): JsonParser, XmlParser, Path, _record(), test_csv_extracts_columns_and_sample_rows(), test_json_array_shape(), test_json_invalid_fails_gracefully(), test_json_object_shape() (+3 more)

### Community 60 - "Lexical Search (Postgres)"
Cohesion: 0.35
Nodes (9): PostgresLexicalSearch, Session, Lexical search over Part 6's `chunks` table, using PostgreSQL's built-in full-…, _chunk(), _seed(), test_finds_exact_identifier_match(), test_no_match_returns_empty(), test_respects_project_filter() (+1 more)

### Community 61 - "Neo4j Graph Store Tests"
Cohesion: 0.33
Nodes (10): _edge(), _node(), Mirrors `tests/unit/storage/test_graph_queries.py` exactly (same graph shapes,…, _seed(), test_resync_prunes_removed_nodes(), test_shortest_path_found(), test_shortest_path_returns_none_when_unreachable(), test_transitive_dependencies_diamond_shape() (+2 more)

### Community 62 - "FastAPI App & Routers"
Cohesion: 0.27
Nodes (7): FastAPI, create_app(), FastAPI application assembly., api_client(), auth_headers(), fixture, API test fixtures. Overrides get_db_session to yield the same transactional…

### Community 63 - "Graph Store Interfaces"
Cohesion: 0.32
Nodes (4): GraphPathResult, GraphStore, Protocol, Graph store interface. Deliberately the same three operations…

### Community 64 - "Parser Registry"
Cohesion: 0.67
Nodes (6): build_default_registry(), Path, _record(), test_registry_dispatches_by_category(), test_registry_returns_none_for_unregistered_category(), test_uvm_and_assertion_categories_use_same_sv_parser()

### Community 65 - "JSON File Manifest Store"
Cohesion: 0.43
Nodes (3): JSONFileManifestStore, Path, Persists one JSON file per project under `store_dir`. Not safe for concurrent…

### Community 67 - "Scanner Hashing Tests"
Cohesion: 0.53
Nodes (5): Path, test_hash_bytes_matches_hash_file(), test_hash_file_changes_with_content(), test_hash_file_is_deterministic(), test_hash_large_file_streams_correctly()

### Community 68 - "Full Pipeline Integration Tests"
Cohesion: 0.60
Nodes (5): Path, Full pipeline test: real files on disk -> real Scanner -> real parser registry…, test_full_pipeline_persists_to_postgres(), test_rescanning_after_file_deletion_prunes_postgres_state(), _write()

## Knowledge Gaps
- **1 isolated node(s):** `semrag`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FileRecord` connect `Chunking Pipeline` to `Parser Registry`, `Scanner & Storage`, `Parser Models & Structured Data`, `Agent Loop & LLM`, `TCL Parser`, `PDF Parser`, `Parser Interfaces & Pipeline`, `Regression Log Parser`, `Scanner Change Detection`, `VHDL Parser`, `Community 50`, `Python Parser (TCL)`, `Markdown & Document Parsers`, `Community 51`, `Structured Data Parsers (JSON/XML)`, `IR Builder & ID Scheme`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `Settings` connect `Settings Factory & Services Init` to `Core Services`, `Retrieval Intent Classification`, `Observability & Ollama LLM`, `Debug Artifacts & Settings`, `LLM Factory & Providers`, `Observability Logging`, `Community 55`, `Community 56`, `Core Settings`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `PromptBuilder` connect `Prompting & Agent Tools` to `Evaluation Harness`, `Core Services`, `Retrieval Intent Classification`, `Community 54`, `Community 56`, `Vector Store Interfaces`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `Settings` (e.g. with `_HumanFormatter` and `_JsonFormatter`) actually correct?**
  _`Settings` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `PgvectorStore` (e.g. with `HybridRetriever` and `DebugSummary`) actually correct?**
  _`PgvectorStore` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `semrag` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Chunking Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07359781121751026 - nodes in this community are weakly interconnected._