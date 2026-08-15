import random

from semrag.storage.vector.models import VectorChunk, VectorSearchFilter
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore
from semrag.storage.postgres.models import Project


def _random_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(EMBEDDING_DIMENSION)]


def _chunk(chunk_id: str, project_id: str, chunk_type: str, seed: int, text: str = "sample text") -> VectorChunk:
    return VectorChunk(
        chunk_id=chunk_id,
        project_id=project_id,
        relative_path="RTL/i2c_master.sv",
        chunk_type=chunk_type,
        text=text,
        embedding=_random_vector(seed),
        line_start=1,
        line_end=10,
    )


def _ensure_project(db_session, project_id: str) -> None:
    db_session.merge(Project(project_id=project_id, root_path="/repo"))
    db_session.flush()


def test_upsert_and_search_returns_nearest_neighbor(db_session):
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)

    target = _chunk("chunk-1", "proj1", "sv_module", seed=1, text="i2c master module")
    decoy = _chunk("chunk-2", "proj1", "sv_module", seed=999, text="unrelated chunk")
    store.upsert_chunks([target, decoy])
    db_session.flush()

    # Query with the *exact* embedding of `target` — it must come back
    # first, with the highest score of anything in the collection.
    results = store.search(query_vector=_random_vector(1), top_k=2)
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score > results[1].score


def test_search_respects_project_filter(db_session):
    _ensure_project(db_session, "proj_a")
    _ensure_project(db_session, "proj_b")
    store = PgvectorStore(db_session)

    chunk_a = _chunk("chunk-a", "proj_a", "sv_module", seed=5)
    chunk_b = _chunk("chunk-b", "proj_b", "sv_module", seed=5)  # same embedding, different project
    store.upsert_chunks([chunk_a, chunk_b])
    db_session.flush()

    results = store.search(
        query_vector=_random_vector(5), top_k=10, filters=VectorSearchFilter(project_id="proj_a")
    )
    assert {r.chunk_id for r in results} == {"chunk-a"}


def test_search_respects_chunk_type_filter(db_session):
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)

    module_chunk = _chunk("chunk-mod", "proj1", "sv_module", seed=7)
    assertion_chunk = _chunk("chunk-assert", "proj1", "assertion", seed=7)
    store.upsert_chunks([module_chunk, assertion_chunk])
    db_session.flush()

    results = store.search(
        query_vector=_random_vector(7), top_k=10, filters=VectorSearchFilter(chunk_type="assertion")
    )
    assert {r.chunk_id for r in results} == {"chunk-assert"}


def test_upsert_is_idempotent_and_updates_existing_chunk(db_session):
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)

    chunk_v1 = _chunk("chunk-1", "proj1", "sv_module", seed=1, text="version 1")
    store.upsert_chunks([chunk_v1])
    db_session.flush()

    chunk_v2 = _chunk("chunk-1", "proj1", "sv_module", seed=2, text="version 2")
    store.upsert_chunks([chunk_v2])
    db_session.flush()

    results = store.search(query_vector=_random_vector(2), top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].text == "version 2"


def test_delete_chunks_removes_them_from_search(db_session):
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)

    chunk = _chunk("chunk-1", "proj1", "sv_module", seed=1)
    store.upsert_chunks([chunk])
    db_session.flush()

    store.delete_chunks(["chunk-1"])
    db_session.flush()

    results = store.search(query_vector=_random_vector(1), top_k=10)
    assert results == []


def test_delete_by_project_removes_only_that_projects_chunks(db_session):
    _ensure_project(db_session, "proj_a")
    _ensure_project(db_session, "proj_b")
    store = PgvectorStore(db_session)

    store.upsert_chunks([
        _chunk("chunk-a", "proj_a", "sv_module", seed=1),
        _chunk("chunk-b", "proj_b", "sv_module", seed=2),
    ])
    db_session.flush()

    store.delete_by_project("proj_a")
    db_session.flush()

    results_a = store.search(query_vector=_random_vector(1), top_k=10, filters=VectorSearchFilter(project_id="proj_a"))
    results_b = store.search(query_vector=_random_vector(2), top_k=10, filters=VectorSearchFilter(project_id="proj_b"))
    assert results_a == []
    assert len(results_b) == 1


def _metadata_chunk(chunk_id: str, project_id: str, seed: int) -> VectorChunk:
    return VectorChunk(
        chunk_id=chunk_id,
        project_id=project_id,
        relative_path="RTL/i2c_master.sv",
        chunk_type="sv_module",
        text="module i2c_master; endmodule",
        embedding=_random_vector(seed),
        line_start=1,
        line_end=2,
        language="systemverilog",
        category="rtl",
        parser_name="pyslang_syntax",
        chunker_name="systemverilog",
        symbol_name="i2c_master",
    )


def test_ri06_metadata_persistence_round_trip(db_session):
    """Metadata survives upsert → search intact (the full storage path)."""
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)
    store.upsert_chunks([_metadata_chunk("chunk-meta", "proj1", seed=11)])
    db_session.flush()

    results = store.search(query_vector=_random_vector(11), top_k=5)
    assert len(results) == 1
    result = results[0]
    assert result.language == "systemverilog"
    assert result.category == "rtl"
    assert result.parser_name == "pyslang_syntax"
    assert result.chunker_name == "systemverilog"
    assert result.symbol_name == "i2c_master"
    assert result.line_start == 1
    assert result.line_end == 2


def test_upsert_updates_metadata_on_conflict(db_session):
    """Re-upserting an existing chunk refreshes its metadata columns."""
    _ensure_project(db_session, "proj1")
    store = PgvectorStore(db_session)
    store.upsert_chunks([_metadata_chunk("chunk-meta", "proj1", seed=11)])
    db_session.flush()

    updated = _metadata_chunk("chunk-meta", "proj1", seed=12)
    updated.symbol_name = "renamed"
    updated.text = "module renamed; endmodule"
    store.upsert_chunks([updated])
    db_session.flush()

    results = store.search(query_vector=_random_vector(12), top_k=5)
    assert len(results) == 1
    assert results[0].symbol_name == "renamed"
    assert results[0].text == "module renamed; endmodule"
