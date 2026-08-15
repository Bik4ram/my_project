from semrag.evaluation.citation_validator import validate_citations
from semrag.retrieval.models import RetrievalResult


def _chunk(relative_path: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c_{relative_path}", project_id="proj1", relative_path=relative_path,
        chunk_type="sv_module", text="text", fused_score=1.0,
    )


def test_all_citations_valid():
    response = "The assertion fails per [Source: RTL/i2c_master.sv, lines 12-18]."
    result = validate_citations(response, [_chunk("RTL/i2c_master.sv")])
    assert result.fabricated_citations == set()
    assert result.valid_citations == {"RTL/i2c_master.sv"}
    assert result.has_fabricated_citations is False


def test_fabricated_citation_detected():
    response = "See [Source: RTL/nonexistent.sv, lines 1-5] for details."
    result = validate_citations(response, [_chunk("RTL/i2c_master.sv")])
    assert result.has_fabricated_citations is True
    assert "RTL/nonexistent.sv" in result.fabricated_citations


def test_mixed_valid_and_fabricated():
    response = (
        "First point from [Source: RTL/i2c_master.sv, lines 1-3], "
        "second from [Source: RTL/made_up.sv]."
    )
    result = validate_citations(response, [_chunk("RTL/i2c_master.sv")])
    assert result.valid_citations == {"RTL/i2c_master.sv"}
    assert result.fabricated_citations == {"RTL/made_up.sv"}
    assert result.fabrication_rate == 0.5


def test_no_citations_in_response():
    result = validate_citations("No citations here at all.", [_chunk("RTL/i2c_master.sv")])
    assert result.cited_paths == set()
    assert result.fabrication_rate == 0.0
    assert result.has_fabricated_citations is False


def test_citation_without_line_numbers():
    response = "See [Source: docs/spec.md] for the register map."
    result = validate_citations(response, [_chunk("docs/spec.md")])
    assert result.valid_citations == {"docs/spec.md"}


def test_empty_context_makes_every_citation_fabricated():
    response = "See [Source: RTL/i2c_master.sv, lines 1-2]."
    result = validate_citations(response, [])
    assert result.has_fabricated_citations is True
    assert result.fabrication_rate == 1.0
