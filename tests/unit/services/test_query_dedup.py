"""Tests for the chunk hygiene helpers and structural file-index path
in QueryService — these run without Postgres/LLM, so they're cheap
unit tests rather than integration tests."""
from __future__ import annotations

from semrag.retrieval.models import RetrievalResult
from semrag.services.services import (
    _deduplicate_chunks,
    _filter_for_structural,
    _is_repetitive_noise,
    _line_signature,
)


def _result(
    chunk_id: str,
    relative_path: str,
    text: str,
    chunk_type: str = "sv_module",
    fused_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="p",
        chunk_type=chunk_type,
        relative_path=relative_path,
        text=text,
        fused_score=fused_score,
        signal_scores={"vector": fused_score},
    )


# ---------------------------------------------------------------------------
# Repetitive-noise detection
# ---------------------------------------------------------------------------


def test_is_repetitive_noise_flags_log_dump():
    """60 identical lines = log dump, must be flagged."""
    repeated = "\n".join(["[STATE_COV_ITEM] state_t IDLE"] * 60)
    assert _is_repetitive_noise(repeated)


def test_is_repetitive_noise_passes_normal_code():
    """Diverse lines from an SV module — not repetitive."""
    code = "\n".join(
        [
            "module i2c_master(",
            "    input clk, rst,",
            "    output SDA, SCL",
            ");",
            "    always @(posedge clk) begin",
            "        case (state)",
            "            IDLE: state = START;",
            "            START: state = ADDR;",
            "        endcase",
            "    end",
            "endmodule",
        ]
    )
    assert not _is_repetitive_noise(code)


def test_is_repetitive_noise_passes_rtl_fsm_body_with_blank_lines():
    """Regression: real RTL FSM bodies reuse `end` and signal assignments
    and contain many blank lines. The old filter counted blank lines and
    flagged these as log dumps, silently dropping the exact chunks that
    answer RTL questions. Real rtl/i2c_master.sv:108-164 repeats `end`
    8x across 49 non-blank lines (16%) — under the 25% fraction bound."""
    body = [
        "                    scl = 1; // posedge scl",
        "                    if(wr_en == 1'b1) begin",
        "                        state = WSTOP_1;",
        "                    end",
        "                    else begin",
        "                        state = READ_DATA;",
        "                        sda = 1;",
        "                    end",
        "                WSTOP_1 : begin",
        "                    scl = 0; // negedge scl",
        "                    sda = 1;",
        "                    if(stop == 1) begin",
        "                        state = WSTOP_2;",
        "                        sda = 0;",
        "                    end",
        "                    else begin",
        "                        state = WRITE_DATA;",
        "                        sda = 1;",
        "                    end",
        "                WSTOP_2 : begin",
        "                    scl = 1; // posedge scl",
        "                    sda = 1;",
        "                    if(stop == 1) begin",
        "                        state = WSTOP_3;",
        "                        sda = 0;",
        "                    end",
        "                    else begin",
        "                        state = READ_ACK;",
        "                        sda = 1;",
        "                    end",
        "                WSTOP_3 : begin",
        "                    scl = 0; // negedge scl",
        "                    sda = 1;",
        "                    if(stop == 1) begin",
        "                        state = IDLE;",
        "                        sda = 0;",
        "                    end",
        "                    else begin",
        "                        state = READ_DATA;",
        "                        sda = 1;",
        "                    end",
    ]
    # 8 closing `end` keywords scattered through the FSM body, as in the
    # real i2c_master.sv — not clustered, not the majority of lines.
    code = "\n".join(
        body + ["                end"] * 8 + ["", "    endcase"]
    )
    assert not _is_repetitive_noise(code)


def test_is_repetitive_noise_passes_file_header_banner():
    """Regression: UVM file headers use a box-drawing banner. Each `##`
    line carries unique content (class name, project, date, author), so
    no single non-blank line dominates. Real write1_seq.svh:1-30 has 24
    non-blank lines with zero exact repeats — never noise."""
    banner = "\n".join(
        [
            "/*###################################################################*\\",
            "##              Class Name: i2c_master_write1_seq                    ##",
            "##              Project Name: i2c_master                             ##",
            "##              Date:   7/12/2023                                    ##",
            "##              Author: Kholoud Ebrahim Darwseh                      ##",
            "##",
            "##              This sequence drives a write transaction                 ##",
            "##              on the I2C master interface.                           ##",
            "##",
            r"\*###################################################################*/",
        ]
    )
    assert not _is_repetitive_noise(banner)


def test_is_repetitive_noise_flags_partial_log_dump():
    """A real coverage dump repeats a long path line far more than any
    source chunk (assertions.txt repeats a D:/questasim path 19x in 53
    lines ≈ 36%). That must still be flagged."""
    dump = "\n".join(
        [
            "ASSERTION RESULTS:",
            "-------------------------------",
            "Name                  File(Line)        Failure Pass",
            "/uvm_pkg/uvm_callbacks/uvm_callbacks__1/initialize/immed__503",
            "                       D:/questasim_10.0b/win32/../uvm_callback.svh(503)  0   2",
        ]
        + [
            "/uvm_pkg/uvm_callbacks/uvm_callbacks__N/initialize/immed__503",
            "                       D:/questasim_10.0b/win32/../uvm_callback.svh(503)  0   2",
        ]
        * 9
    )
    assert _is_repetitive_noise(dump)


def test_is_repetitive_noise_handles_empty_input():
    assert not _is_repetitive_noise("")


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_deduplicate_chunks_drops_repetitive_logs():
    noise = "\n".join(["[STATE_COV_ITEM] x"] * 30)
    real = "module a; endmodule"
    results = [
        _result("c1", "sim/log/x.log", noise, chunk_type="log", fused_score=0.9),
        _result("c2", "rtl/a.sv", real, chunk_type="sv_module", fused_score=0.5),
    ]
    kept, dropped = _deduplicate_chunks(results)
    assert dropped == 1
    assert [r.chunk_id for r in kept] == ["c2"]


def test_deduplicate_chunks_collapses_duplicate_file_slices():
    """Two generic_window chunks of the same coverage file with the
    same non-empty lines must collapse to one."""
    same = "TOTAL COVERGROUP: 100%\nbin data 80%\nbin addr 90%"
    results = [
        _result("c1", "sim/cov/coverage.txt", same, chunk_type="generic_window", fused_score=0.7),
        _result("c2", "sim/cov/coverage.txt", same, chunk_type="generic_window", fused_score=0.4),
    ]
    kept, dropped = _deduplicate_chunks(results)
    assert dropped == 1
    assert len(kept) == 1
    # Highest-scored wins.
    assert kept[0].chunk_id == "c1"


def test_deduplicate_chunks_keeps_different_files():
    a = "module a; endmodule"
    b = "module b; endmodule"
    results = [
        _result("c1", "rtl/a.sv", a),
        _result("c2", "rtl/b.sv", b),
    ]
    kept, dropped = _deduplicate_chunks(results)
    assert dropped == 0
    assert {r.chunk_id for r in kept} == {"c1", "c2"}


def test_line_signature_strips_whitespace_only_lines():
    sig_a = _line_signature("foo\n\nbar")
    sig_b = _line_signature("foo\n   \nbar")
    assert sig_a == sig_b


def test_line_signature_distinguishes_different_content():
    assert _line_signature("foo") != _line_signature("bar")


# ---------------------------------------------------------------------------
# Structural filter
# ---------------------------------------------------------------------------


def test_filter_for_structural_drops_log_coverage_lint_timing_waveform():
    """With enough non-noise chunks to keep, the filter strips them."""
    results = [
        _result("a", "rtl/i2c_master.sv", "module x;", chunk_type="sv_module"),
        _result("b", "sim/log/x.log", "log", chunk_type="log"),
        _result("c", "sim/cov/x.txt", "cov", chunk_type="coverage_report"),
        _result("d", "tb/i2c_scoreboard.svh", "scb", chunk_type="sv_module"),
        _result("e", "tb/i2c_driver.svh", "drv", chunk_type="sv_module"),
        _result("f", "tb/i2c_monitor.svh", "mon", chunk_type="sv_module"),
    ]
    kept = _filter_for_structural(results)
    assert {r.chunk_id for r in kept} == {"a", "d", "e", "f"}


def test_filter_for_structural_falls_back_when_filtering_leaves_too_few():
    """If the filter would leave fewer than 4 chunks, return the
    unfiltered list — better to give the model some noisy log content
    than a prompt with zero evidence blocks."""
    results = [
        _result("a", "sim/log/x.log", "log", chunk_type="log"),
        _result("b", "sim/log/y.log", "log", chunk_type="log"),
    ]
    kept = _filter_for_structural(results)
    assert {r.chunk_id for r in kept} == {"a", "b"}


def test_filter_for_structural_keeps_generic_window_chunks():
    """generic_window chunks can be legitimate (the scoreboard file,
    which has no SV module/package IR nodes, only chunks via the
    fallback chunker). The structural filter keeps them; the dedup
    step is what strips the repetitive noise."""
    results = [
        _result("a", "rtl/i2c_master.sv", "module x;", chunk_type="sv_module"),
        _result("b", "tb/i2c_master_scoreboard.svh", "class x; endclass", chunk_type="generic_window"),
    ]
    kept = _filter_for_structural(results)
    assert {r.chunk_id for r in kept} == {"a", "b"}


def test_filter_for_structural_drops_logs_by_extension():
    """Files that fell through to the generic_window chunker (.log,
    .vcd, .rpt) still get filtered — without the extension check,
    runtime artifacts slip past the chunk_type filter and pollute
    structural answers with the wrong evidence. Plain .txt coverage
    and .do scripts are kept (the dedup step handles repetitive noise)."""
    results = [
        _result("a", "rtl/i2c_master.sv", "module x;", chunk_type="sv_module"),
        _result("b", "tb/i2c_driver.svh", "drv", chunk_type="sv_module"),
        _result("c", "tb/i2c_monitor.svh", "mon", chunk_type="sv_module"),
        _result("d", "tb/i2c_seq_lib.svh", "seq", chunk_type="sv_module"),
        _result("e", "sim/log/write1_test_log.log", "[STATE_COV_ITEM] x", chunk_type="generic_window"),
        _result("f", "sim/wave.vcd", "$var", chunk_type="generic_window"),
        _result("g", "sim/coverage.rpt", "Total:", chunk_type="generic_window"),
        _result("h", "sim/coverage/code_coverage.txt", "totals", chunk_type="generic_window"),
        _result("i", "sim/i2c_master_run.do", "run -all", chunk_type="generic_window"),
    ]
    kept = _filter_for_structural(results)
    # 5 design chunks survive; 4 noise chunks dropped.
    assert {r.chunk_id for r in kept} == {"a", "b", "c", "d", "i"}