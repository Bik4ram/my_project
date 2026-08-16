from datetime import UTC, datetime
from pathlib import Path

from semrag.ingestion.chunking.generic_window_chunker import GenericWindowChunker
from semrag.ingestion.chunking.log_chunker import LogChunker
from semrag.ingestion.scanner.models import FileCategory, FileRecord

REL_PATH = Path("fake.log")


def _record() -> FileRecord:
    return FileRecord(
        project_id="proj1", absolute_path="/tmp/fake.log", relative_path="fake.log",
        size_bytes=1, mtime=datetime.now(UTC), content_hash="x",
        category=FileCategory.LOG, extension=".log", is_binary=False,
    )


def _chunks(text: str) -> list:
    chunker = LogChunker()
    return chunker.chunk(_record(), None, text, [])


def _text_of(chunks: list) -> list[str]:
    return [c.text for c in chunks]


def test_error_with_multiline_continuation_stays_together():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_ERROR @ 500: tb.uut [CHK] data mismatch",
        "#   property expected: 0x3f",
        "#   actual:            0x2a",
        "# UVM_INFO @ 600: reporter [DONE] Test finished",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 3
    assert "data mismatch" in chunks[1].text
    assert "property expected" in chunks[1].text
    assert "actual:" in chunks[1].text
    assert chunks[1].line_start == 2
    assert chunks[1].line_end == 4


def test_fatal_with_stack_trace_stays_together():
    log = "\n".join([
        "# UVM_FATAL @ 100: tb [FATAL] assertion failed in checker",
        "#   at check_data (checker.sv:42)",
        "#   at run_phase (tb.sv:120)",
        "#   UVM_FATAL at time 100 ns: 1 occurrences",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 1
    assert "checker.sv:42" in chunks[0].text
    assert chunks[0].text.count("UVM_FATAL") >= 1


def test_assertion_failure_is_an_event_boundary():
    log = "\n".join([
        "# [STATE_COV_ITEM] writing state_cov[3]",
        "# ASSERTION FAILURE: tb.uut.a1 expected 1 got 0",
        "#   severity: error",
    ])
    chunks = _chunks(log)
    # The STATE_COV line is one group; the assertion failure plus its
    # continuation is a separate chunk.
    assert len(chunks) == 2
    assert "ASSERTION FAILURE" in chunks[1].text
    assert "severity: error" in chunks[1].text


def test_warning_is_an_event_boundary():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_WARNING @ 100: tb [WARN] timing not met",
        "#   slack: -0.1ns",
        "# UVM_INFO @ 200: reporter [DONE] done",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 3
    assert chunks[1].text.startswith("# UVM_WARNING")
    assert "slack: -0.1ns" in chunks[1].text


def test_plain_pass_fail_info_are_not_boundaries():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# PASS: tb.uut check_1",
        "# FAIL: tb.uut check_2",
        "# UVM_INFO @ 100: reporter [DONE] done",
    ])
    chunks = _chunks(log)
    # PASS/FAIL alone must not start a new event; the RNTST line and the two
    # single-line result lines group into one window.
    assert len(chunks) == 2
    assert "PASS: tb.uut check_1" in chunks[0].text
    assert "FAIL: tb.uut check_2" in chunks[0].text


def test_repetitive_single_line_events_grouped_into_bounded_windows():
    lines = [
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
    ]
    lines += [
        f"# UVM_INFO @ {i}: cov [STATE_COV_ITEM] writing state_cov[{i}]"
        for i in range(1, 130)
    ]
    log = "\n".join(lines)

    chunks = _chunks(log)
    assert len(chunks) >= 3
    # Every repetitive line survives in exactly one chunk (nothing deleted).
    all_text = "\n".join(c.text for c in chunks)
    for i in range(1, 130):
        assert f"state_cov[{i}]" in all_text
    # Bounded chunk sizes.
    for c in chunks:
        assert c.line_end - c.line_start + 1 <= 60


def test_very_long_event_split_with_overlap():
    log = "\n".join(
        ["# UVM_ERROR @ 0: tb [BIG] long event start"]
        + [f"#   continuation {i}" for i in range(200)]
    )
    chunker = LogChunker(window_lines=60, overlap_lines=10)
    chunks = chunker.chunk(_record(), None, log, [])

    assert len(chunks) > 1
    assert chunks[0].line_start == 1
    assert chunks[-1].line_end == 201
    covered = set()
    for c in chunks:
        covered.update(range(c.line_start, c.line_end + 1))
    assert covered == set(range(1, 202))
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.line_start <= a.line_end


def test_multiple_events_split_correctly():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_INFO @ 10: tb [X_MATCH_ITEM] address match",
        "#   ADDR: 0xdeadbeef",
        "#   DATA: 0x1",
        "# [X_PASS] match check passed",
        "# UVM_ERROR @ 20: tb [X_MATCH_ITEM] data mismatch",
        "#   ADDR: 0x1234",
        "#   DATA: 0x2",
        "# [X_FAIL] match check failed",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 3
    # First event absorbs its trailing [X_PASS] verdict.
    assert "ADDR: 0xdeadbeef" in chunks[1].text
    assert "match check passed" in chunks[1].text
    # Second event absorbs its trailing [X_FAIL] verdict.
    assert "ADDR: 0x1234" in chunks[2].text
    assert "match check failed" in chunks[2].text


def test_event_at_file_start():
    log = "\n".join([
        "# UVM_FATAL @ 0: tb [FATAL] first line is an event",
        "#   detail line",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 1
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 2


def test_event_at_file_end():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_ERROR @ 500: tb [CHK] last line is an event",
        "#   trailing continuation",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 2
    assert chunks[-1].line_end == 3
    assert "last line is an event" in chunks[-1].text


def test_deterministic_chunk_ids():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_ERROR @ 500: tb [CHK] boom",
        "#   continuation",
        "# UVM_INFO @ 600: reporter [DONE] done",
    ])
    a = _chunks(log)
    b = _chunks(log)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.line_start for c in a] == [c.line_start for c in b]
    for c in a:
        assert c.chunk_type == "log"
        assert c.source_node_id is None
        assert c.relative_path == "fake.log"
        assert c.project_id == "proj1"


def test_chunks_are_chronologically_ordered():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# UVM_ERROR @ 500: tb [CHK] boom one",
        "# UVM_INFO @ 100: reporter [RNTST] second run",
        "# UVM_ERROR @ 900: tb [CHK] boom two",
    ])
    chunks = _chunks(log)
    starts = [c.line_start for c in chunks]
    assert starts == sorted(starts)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.line_start > a.line_end  # strictly increasing, no overlap


def test_summary_marker_swallows_to_eof():
    log = "\n".join([
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test",
        "# --- UVM Report Summary ---",
        "# ** Report counts by severity",
        "# UVM_INFO : 1885",
        "# UVM_ERROR : 0",
        "# UVM_INFO : 1",
        "# UVM_FATAL : 0",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 2
    assert "Report counts by severity" in chunks[1].text
    assert "UVM_INFO : 1885" in chunks[1].text
    assert chunks[1].text.count("UVM_") >= 3  # count lines didn't re-split


def test_no_recognizable_markers_degrades_to_generic_windowing():
    lines = [f"line {i}" for i in range(200)]
    log = "\n".join(lines)
    chunks = _chunks(log)

    generic = GenericWindowChunker(window_lines=60, overlap_lines=10)
    generic_chunks = generic.chunk(_record(), None, log, [])

    assert [c.line_start for c in chunks] == [c.line_start for c in generic_chunks]
    assert [c.line_end for c in chunks] == [c.line_end for c in generic_chunks]
    assert chunks[0].line_start == 1
    assert chunks[-1].line_end == 200


def test_empty_file_produces_no_chunks():
    assert _chunks("") == []


def test_small_markerless_file_produces_one_chunk():
    log = "\n".join([f"line {i}" for i in range(10)])
    chunks = _chunks(log)
    assert len(chunks) == 1
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 10


def test_whitespace_only_file_produces_no_chunks():
    assert _chunks(" \n \n") == []


def test_bracketed_verdict_tags_absorbed_like_real_corpus():
    log = "\n".join([
        "# UVM_INFO E:/.../stat_monitor.svh(50) @ 155ns: ...mon2 [STATE_COV_ITEM] the packet",
        "# UVM_INFO E:/.../monitor.svh(69) @ 165ns: ...mon [ADDR_MATCH_ITEM] checking address",
        "# ---------------------------------------------------------------------",
        "# item_2  i2c_master_sequence_item  -  @701",
        "#   address  integral  7  'h13",
        "# ---------------------------------------------------------------------",
        "# UVM_INFO E:/.../scoreboard.svh(42) @ 165ns: ...scb [ADDR_PASS] Address = 0010011",
        "# UVM_INFO E:/.../stat_monitor.svh(50) @ 175ns: ...mon2 [STATE_COV_ITEM] the packet",
    ])
    chunks = _chunks(log)
    assert len(chunks) == 3
    # The ADDR_MATCH_ITEM event absorbs its trailing [ADDR_PASS] verdict,
    # exactly like the real corpus's write1_test_log.log structure.
    assert "ADDR_MATCH_ITEM" in chunks[1].text
    assert "[ADDR_PASS]" in chunks[1].text
    assert "address" in chunks[1].text


def test_invalid_overlap_configuration_rejected():
    import pytest
    with pytest.raises(ValueError):
        LogChunker(window_lines=50, overlap_lines=50)