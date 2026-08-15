"""Service layer — top-level error handling and shape contracts.

Heavy integration is covered by the existing `test_full_pipeline_integration`
tests; this file focuses on:
- input validation (NotADirectoryError on bad root)
- dataclass shape of IndexResult / QueryAnswer
- non-existent path rejection before any I/O
- that QueryService.ask() seeds the agent with retrieved context
  (the bug that motivated this whole iteration: previously the agent
  started with `[SYSTEM_PROMPT, question]` only and had to call
  `search_codebase` itself to discover the same context the retriever
  could have surfaced up front).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from semrag.core.settings import Settings
from semrag.reasoning.llm.interfaces import CompletionResult
from semrag.retrieval.models import RetrievalResult
from semrag.services.services import (
    IndexResult,
    IngestionService,
    QueryAnswer,
    QueryService,
)


class _NoOpEmbeddings:
    """Stand-in for an EmbeddingProvider that records calls without
    touching the network. Just enough for the constructor."""
    model_name = "stub"
    provider_name = "stub"

    def embed_texts(self, texts):
        return [[0.0] * 4 for _ in texts]


class _NoOpCache:
    def get(self, key):
        return None

    def set(self, key, value):
        pass


class _StaticEmbeddings:
    """Embedding provider for the seeded-context test — always returns
    the same vector so the retriever's input is deterministic."""

    model_name = "static"
    provider_name = "static"

    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _StubRetriever:
    """Returns a fixed list of RetrievalResult so the test can assert
    on what the agent receives without needing Postgres or a real
    embedding."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results
        self.calls: list[tuple[str, list[float]]] = []

    def retrieve(self, query_text, query_embedding, *, project_id, top_k, filters=None):
        self.calls.append((query_text, query_embedding))
        return list(self._results)[:top_k]


class _CapturingLLM:
    """Records the messages the agent passes to `complete()` so the
    test can assert retrieval-augmented context is present."""

    def __init__(self, response: CompletionResult):
        self._response = response
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        return self._response

    def stream(self, messages, tools=None):
        raise NotImplementedError


def _settings() -> Settings:
    return Settings(
        llm_probe_at_startup=False,
        debug_artifact_retention=0,  # don't write artifacts in unit tests
        debug_artifacts_dir="/tmp/semrag-noop-debug",
    )


def test_index_result_is_dataclass_with_expected_fields():
    result = IndexResult(
        project_id="p", scan_id="s", file_count=0, chunk_count=0,
        added=0, modified=0, deleted=0,
    )
    assert result.project_id == "p"


def test_query_answer_has_expected_fields():
    answer = QueryAnswer(
        final_answer="x", iterations_used=1,
        hit_iteration_limit=False, messages=[],
    )
    assert answer.final_answer == "x"
    assert answer.iterations_used == 1
    assert answer.messages == []


def test_ingestion_rejects_nonexistent_root(tmp_path):
    """Bad root should fail with NotADirectoryError before any I/O. The
    constructor takes a session but we don't reach it for a bad path."""
    service = IngestionService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_NoOpEmbeddings(),
        embedding_cache=_NoOpCache(),
        settings=_settings(),
    )
    bad = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError):
        service.index("p", str(bad))


def test_ingestion_rejects_file_not_directory(tmp_path):
    f = tmp_path / "regular_file.txt"
    f.write_text("x")
    service = IngestionService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_NoOpEmbeddings(),
        embedding_cache=_NoOpCache(),
        settings=_settings(),
    )
    with pytest.raises(NotADirectoryError):
        service.index("p", str(f))


def test_query_service_construction_does_not_touch_session():
    """Constructor should be pure — no DB or LLM calls happen until
    `ask` is invoked."""
    service = QueryService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_NoOpEmbeddings(),
        llm=None,  # type: ignore[arg-type]
        settings=_settings(),
        project_root=None,
    )
    assert service._project_root is None


def test_ask_seeds_agent_with_retrieved_context(monkeypatch):
    """QueryService.ask() must retrieve first and pass the assembled
    context to the agent — not just `[SYSTEM_PROMPT, question]`. The
    agent should see a [Source: ...] citation in its first user
    message, which is what makes the answer grounded."""
    captured_results = [
        RetrievalResult(
            chunk_id="c1",
            project_id="proj1",
            chunk_type="rtl_module",
            relative_path="rtl/i2c_master.sv",
            text="module i2c_master (input logic clk); endmodule",
            line_start=10,
            line_end=14,
            fused_score=0.9,
            signal_scores={"vector": 0.9},
        )
    ]
    retriever = _StubRetriever(captured_results)

    # Patch the retriever factory inside the service to avoid a live
    # Postgres / real HybridRetriever.
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)

    llm = _CapturingLLM(
        CompletionResult(
            content="i2c_master is the RTL module "
            "[Source: rtl/i2c_master.sv, lines 10-14].",
            finish_reason="stop",
        )
    )
    service = QueryService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_StaticEmbeddings(),
        llm=llm,
        settings=_settings(),
    )

    answer = service.ask("proj1", "what is i2c_master?")

    # Retrieval was actually called with the user's query.
    assert retriever.calls, "retriever was not invoked"
    assert retriever.calls[0][0] == "what is i2c_master?"

    # The agent received more than just system + user — the user
    # message must contain the citation the retriever produced.
    assert llm.calls, "agent was never invoked"
    user_message = next(m for m in llm.calls[0] if m["role"] == "user")
    assert "[Source: rtl/i2c_master.sv" in user_message["content"], (
        "the user message handed to the LLM should contain the "
        f"retrieved citation, got: {user_message['content']!r}"
    )
    assert "CONTEXT:" in user_message["content"]

    # And the final answer flows back to the caller.
    assert answer.final_answer == (
        "i2c_master is the RTL module [Source: rtl/i2c_master.sv, lines 10-14]."
    )
    assert answer.iterations_used == 1
    assert not answer.hit_iteration_limit


class _StubFileIndexSession:
    """In-memory stand-in for the SQLAlchemy session that backs
    `_project_file_index`. Yields a fixed (category, path) list as if
    from `SELECT category, relative_path FROM files`."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def all(self):
                return self._rows

        return _Result(self._rows)


def test_ask_injects_file_index_for_structural_query(monkeypatch):
    """For 'summarize the repository' style queries the user message
    must include a FILE INDEX block listing real paths from the DB."""
    real_results = [
        RetrievalResult(
            chunk_id="c1",
            project_id="proj1",
            chunk_type="sv_module",
            relative_path="rtl/i2c_master.sv",
            text="module i2c_master (input logic clk); endmodule",
            line_start=10,
            line_end=14,
            fused_score=0.9,
            signal_scores={"vector": 0.9},
        )
    ]
    retriever = _StubRetriever(real_results)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)

    file_index_session = _StubFileIndexSession([
        ("rtl_verilog", "rtl/i2c_master.sv"),
        ("testbench_uvm", "tb/i2c_master_pkg.sv"),
    ])
    monkeypatch.setattr(QueryService, "_session", file_index_session, raising=False)

    llm = _CapturingLLM(
        CompletionResult(content="Repo has RTL and UVM testbench.", finish_reason="stop")
    )
    service = QueryService(
        session=file_index_session,
        embedding_provider=_StaticEmbeddings(),
        llm=llm,
        settings=_settings(),
    )

    answer = service.ask("proj1", "Summarize the repository structure.")

    user_message = next(m for m in llm.calls[0] if m["role"] == "user")
    assert "FILE INDEX:" in user_message["content"]
    assert "rtl/i2c_master.sv" in user_message["content"]
    assert "tb/i2c_master_pkg.sv" in user_message["content"]
    # The structural prompt also has the FILE INDEX guidance text.
    system_message = next(m for m in llm.calls[0] if m["role"] == "system")
    assert "FILE INDEX" in system_message["content"]

    assert answer.debug is not None
    assert answer.debug.file_index_included is True
    assert answer.debug.intent == "structural"


def test_ask_does_not_inject_file_index_for_non_structural_query(monkeypatch):
    """Specific subsystem questions don't need a FILE INDEX."""
    captured = [
        RetrievalResult(
            chunk_id="c1",
            project_id="proj1",
            chunk_type="sv_module",
            relative_path="rtl/i2c_master.sv",
            text="module i2c_master; endmodule",
            line_start=10,
            line_end=14,
            fused_score=0.9,
            signal_scores={"vector": 0.9},
        )
    ]
    retriever = _StubRetriever(captured)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)

    llm = _CapturingLLM(
        CompletionResult(content="answer", finish_reason="stop")
    )
    service = QueryService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_StaticEmbeddings(),
        llm=llm,
        settings=_settings(),
    )
    answer = service.ask("proj1", "How does the scoreboard verify data?")
    user_message = next(m for m in llm.calls[0] if m["role"] == "user")
    assert "FILE INDEX:" not in user_message["content"]
    assert answer.debug.file_index_included is False
    assert answer.debug.intent == "general"


def test_ask_drops_repetitive_log_chunks(monkeypatch):
    """Repetitive log chunks must not leak into the prompt — the
    benchmark case where 60 [STATE_COV_ITEM] lines dominate the
    context and the model hallucinates a Windows path."""
    repeated = "\n".join(["[STATE_COV_ITEM] state_t IDLE"] * 30)
    real = "module i2c_master (input logic clk); endmodule"
    results = [
        RetrievalResult(
            chunk_id="log1",
            project_id="proj1",
            chunk_type="log",
            relative_path="sim/log/write1.log",
            text=repeated,
            fused_score=0.9,
            signal_scores={"vector": 0.9},
        ),
        RetrievalResult(
            chunk_id="rtl1",
            project_id="proj1",
            chunk_type="sv_module",
            relative_path="rtl/i2c_master.sv",
            text=real,
            fused_score=0.7,
            signal_scores={"vector": 0.7},
        ),
    ]
    retriever = _StubRetriever(results)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _CapturingLLM(
        CompletionResult(content="answer", finish_reason="stop")
    )
    service = QueryService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_StaticEmbeddings(),
        llm=llm,
        settings=_settings(),
    )
    answer = service.ask("proj1", "what is i2c_master?")
    user_message = next(m for m in llm.calls[0] if m["role"] == "user")
    assert "STATE_COV_ITEM" not in user_message["content"]
    assert "module i2c_master" in user_message["content"]
    assert answer.debug.dedup_dropped == 1
