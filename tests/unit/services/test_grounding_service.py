"""A1 grounding policy wired into QueryService.ask().

Verifies the answer policy end-to-end at the service layer:
grounded answers pass through, unsupported answers are revised or
replaced by a controlled response, and negative queries stay clean.
"""
from __future__ import annotations

from semrag.core.settings import Settings
from semrag.reasoning.llm.interfaces import CompletionResult
from semrag.retrieval.models import RetrievalResult
from semrag.services.services import QueryService


class _StaticEmbeddings:
    model_name = "static"
    provider_name = "static"

    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _StubRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def retrieve(self, query_text, query_embedding, *, project_id, top_k, filters=None):
        self.calls.append((query_text, query_embedding))
        return list(self._results)[:top_k]


class _SequencedLLM:
    """Returns a different CompletionResult per call in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        content = self._responses.pop(0)
        return CompletionResult(content=content, finish_reason="stop")

    def stream(self, messages, tools=None):
        raise NotImplementedError


def _chunk(relative_path: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c_{relative_path}",
        project_id="proj1",
        relative_path=relative_path,
        chunk_type="sv_class",
        text=text,
        line_start=1,
        line_end=30,
        fused_score=1.0,
        signal_scores={},
    )


EVIDENCE = [
    _chunk(
        "tb/i2c_master_seq_lib/i2c_master_read_seq.svh",
        "class i2c_master_read_seq extends i2c_master_base_seq;\n"
        "task i2c_master_read_seq::body();\n"
        "    assert(item.randomize() with {rd_wr == 0;});\n"
        "endtask :body\n",
    )
]


def _settings(**overrides) -> Settings:
    base = dict(
        llm_probe_at_startup=False,
        debug_artifact_retention=0,
        debug_artifacts_dir="/tmp/semrag-noop-debug",
    )
    base.update(overrides)
    return Settings(**base)


def _service(retriever, llm, settings) -> QueryService:
    return QueryService(
        session=None,  # type: ignore[arg-type]
        embedding_provider=_StaticEmbeddings(),
        llm=llm,
        settings=settings,
    )


def test_grounded_answer_passes_through_unchanged(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM([
        "i2c_master_read_seq constrains rd_wr == 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 1-30]."
    ])
    service = _service(retriever, llm, _settings())

    answer = service.ask("proj1", "What does i2c_master_read_seq do?")

    assert answer.final_answer == (
        "i2c_master_read_seq constrains rd_wr == 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 1-30]."
    )
    assert answer.grounding["status"] == "grounded"
    assert answer.grounding["revision_attempted"] is False
    assert len(llm.calls) == 1  # no second LLM call


def test_unsupported_answer_is_blocked_when_revision_disabled(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM([
        "The i2c_master_test starts the read sequence."
    ])
    service = _service(
        retriever, llm, _settings(grounding_gate_regenerate=False)
    )

    answer = service.ask("proj1", "What starts the read sequence?")

    assert answer.grounding["status"] == "unsupported"
    assert "i2c_master_test" in answer.grounding["unsupported_entities"]
    assert "i2c_master_test" in answer.final_answer
    assert "does not support" in answer.final_answer
    assert len(llm.calls) == 1  # gate ran, no revision attempted


def test_unsupported_answer_is_revised_when_revision_enabled(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM([
        "The i2c_master_test starts the read sequence.",  # original (bad)
        "i2c_master_read_seq constrains rd_wr == 0 "       # revision (good)
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 1-30].",
    ])
    service = _service(retriever, llm, _settings())

    answer = service.ask("proj1", "What starts the read sequence?")

    assert answer.grounding["status"] == "revised"
    assert answer.grounding["revision_attempted"] is True
    assert answer.grounding["revised"] is True
    assert "i2c_master_test" not in answer.final_answer
    assert "i2c_master_read_seq" in answer.final_answer
    assert len(llm.calls) == 2


def test_revision_failure_falls_back_to_controlled_response(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    # The revision ALSO hallucinates -> never grounded.
    llm = _SequencedLLM([
        "The i2c_master_test starts the read sequence.",
        "The i2c_master_test then drives the transaction.",
    ])
    service = _service(retriever, llm, _settings())

    answer = service.ask("proj1", "What starts the read sequence?")

    assert answer.grounding["status"] == "unsupported"
    assert answer.grounding["revision_attempted"] is True
    assert answer.grounding["revised"] is False
    assert "does not support" in answer.final_answer
    assert len(llm.calls) == 2


def test_negative_query_stays_grounded(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM([
        "There is no class named i2c_master_xyz_sequencer in the retrieved evidence."
    ])
    service = _service(retriever, llm, _settings())

    answer = service.ask("proj1", "Which class extends i2c_master_xyz_sequencer?")

    assert answer.grounding["status"] == "grounded"
    assert answer.final_answer == (
        "There is no class named i2c_master_xyz_sequencer in the retrieved evidence."
    )
    assert len(llm.calls) == 1


def test_source_specific_answer_without_citation_is_insufficient(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM([
        "i2c_master_read_seq constrains rd_wr to 0 in its body."
    ])
    service = _service(
        retriever, llm, _settings(grounding_gate_regenerate=False)
    )

    answer = service.ask("proj1", "What does i2c_master_read_seq do?")

    assert answer.grounding["status"] == "insufficient_evidence"
    assert "couldn't verify" in answer.final_answer
    assert len(llm.calls) == 1


def test_gate_disabled_restores_r4_behavior(monkeypatch):
    retriever = _StubRetriever(EVIDENCE)
    monkeypatch.setattr(QueryService, "_build_retriever", lambda self: retriever)
    llm = _SequencedLLM(["The i2c_master_test starts the read sequence."])
    service = _service(
        retriever, llm, _settings(grounding_gate_enabled=False)
    )

    answer = service.ask("proj1", "What starts the read sequence?")

    # R4 behavior: answer passes through untouched, no grounding field.
    assert answer.final_answer == "The i2c_master_test starts the read sequence."
    assert answer.grounding is None
    assert len(llm.calls) == 1