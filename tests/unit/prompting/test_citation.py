from semrag.reasoning.prompting.citation import format_chunk_with_citation, format_citation_header
from semrag.retrieval.models import RetrievalResult


def _result(line_start=None, line_end=None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c1", project_id="proj1", relative_path="RTL/i2c_master.sv",
        chunk_type="sv_module", text="module i2c_master; endmodule",
        line_start=line_start, line_end=line_end, fused_score=1.0,
    )


def test_citation_includes_line_range_when_known():
    header = format_citation_header(_result(line_start=12, line_end=18))
    assert header == "[Source: RTL/i2c_master.sv, lines 12-18]"


def test_citation_omits_line_range_when_unknown():
    header = format_citation_header(_result())
    assert header == "[Source: RTL/i2c_master.sv]"


def test_formatted_chunk_includes_citation_and_text():
    formatted = format_chunk_with_citation(_result(line_start=1, line_end=3))
    assert formatted.startswith("[Source: RTL/i2c_master.sv, lines 1-3]")
    assert "module i2c_master" in formatted
