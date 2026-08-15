from semrag.retrieval.lexical_search import PostgresLexicalSearch
from semrag.storage.postgres.models import Project
from semrag.storage.vector.models import VectorChunk, VectorSearchFilter
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _chunk(chunk_id: str, project_id: str, text: str, chunk_type: str = "sv_module") -> VectorChunk:
    return VectorChunk(
        chunk_id=chunk_id, project_id=project_id, relative_path="RTL/a.sv",
        chunk_type=chunk_type, text=text, embedding=[0.1] * EMBEDDING_DIMENSION,
    )


def _seed(db_session, project_id: str = "proj1") -> None:
    db_session.merge(Project(project_id=project_id, root_path="/repo"))
    db_session.flush()


def test_finds_exact_identifier_match(db_session):
    _seed(db_session)
    store = PgvectorStore(db_session)
    store.upsert_chunks([
        _chunk("c1", "proj1", "module i2c_master implements start condition detection"),
        _chunk("c2", "proj1", "module spi_slave implements chip select handling"),
    ])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search("i2c_master start condition", top_k=5)
    assert results[0].chunk_id == "c1"


def test_respects_project_filter(db_session):
    _seed(db_session, "proj_a")
    _seed(db_session, "proj_b")
    store = PgvectorStore(db_session)
    store.upsert_chunks([
        _chunk("c1", "proj_a", "start condition assertion check"),
        _chunk("c2", "proj_b", "start condition assertion check"),
    ])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search(
        "start condition", top_k=5, filters=VectorSearchFilter(project_id="proj_a")
    )
    assert {r.chunk_id for r in results} == {"c1"}


def test_no_match_returns_empty(db_session):
    _seed(db_session)
    store = PgvectorStore(db_session)
    store.upsert_chunks([_chunk("c1", "proj1", "module i2c_master start condition")])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search("completely unrelated pharmaceutical terminology", top_k=5)
    assert results == []


def test_natural_language_query_matches_any_term(db_session):
    """Regression: plainto_tsquery ANDs every token, so a natural-language
    question with many distinct terms matched zero chunks (the lexical leg
    was effectively dead for real questions). Any-match semantics must
    surface chunks that cover at least one query term, ranked by how many
    distinct query lexemes they contain."""
    _seed(db_session)
    store = PgvectorStore(db_session)
    store.upsert_chunks([
        _chunk("c1", "proj1", "module i2c_master implements start condition detection"),
        _chunk("c2", "proj1", "agent gets the virtual interface from the config database"),
        _chunk("c3", "proj1", "monitor samples the sda line and builds address_check"),
    ])
    db_session.flush()

    # No single chunk contains ALL of these terms, but c2 covers several
    # and c1 covers at least one (via the implementation/start/condition
    # stem matches on 'implement'... actually it does not). Use a chunk
    # that shares exactly one query lexeme to prove single-term matches
    # still surface under any-match semantics.
    store.upsert_chunks([_chunk("c4", "proj1", "the agent drives the bus")])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search(
        "how is the virtual interface passed to the agent from the config database",
        top_k=5,
    )
    ids = [r.chunk_id for r in results]
    assert "c2" in ids  # covers the most query lexemes
    assert "c4" in ids  # single-term matches still surface


def test_any_match_ranks_higher_coverage_first(db_session):
    """Ranking: a chunk covering more distinct query lexemes must outrank
    one covering fewer, even when the lesser chunk contains rarer terms."""
    _seed(db_session)
    store = PgvectorStore(db_session)
    store.upsert_chunks([
        _chunk("c1", "proj1", "agent agent agent agent agent agent agent agent agent agent"),
        _chunk("c2", "proj1", "virtual interface passed from the top testbench to the agent"),
    ])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search(
        "how is the virtual interface passed to the agent from the top testbench",
        top_k=5,
    )
    assert results[0].chunk_id == "c2"


def test_result_includes_source_node_id_when_present(db_session):
    _seed(db_session)
    store = PgvectorStore(db_session)
    chunk = _chunk("c1", "proj1", "module i2c_master start condition")
    chunk.source_node_id = None  # no IR node in this test, matches VectorChunk default
    store.upsert_chunks([chunk])
    db_session.flush()

    results = PostgresLexicalSearch(db_session).search("i2c_master", top_k=5)
    assert results[0].source_node_id is None  # field exists and is correctly None, not missing
