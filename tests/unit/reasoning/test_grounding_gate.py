"""A1 grounding gate — deterministic, evidence-local tests.

Covers the A1 test plan: grounded answers pass, unsupported
class/method/relationship/file-path are blocked, missing citations on
source-specific answers are flagged, negative queries do not fabricate,
mixed answers keep the supported portion, and generic explanatory
language is not blocked.
"""
from __future__ import annotations

from semrag.reasoning.grounding.gate import (
    check_grounding,
    filter_supported_portion,
)
from semrag.retrieval.models import RetrievalResult

READ_SEQ = """
class i2c_master_read_seq extends i2c_master_base_seq;
    `uvm_object_utils(i2c_master_read_seq)
    i2c_master_sequence_item item;
    extern function new(string name = "i2c_master_read_seq");
    extern task body();
endclass :i2c_master_read_seq

function i2c_master_read_seq::new(string name = "i2c_master_read_seq");
    super.new(name);
endfunction :new

task i2c_master_read_seq::body();
    item = i2c_master_sequence_item::type_id::create("item");
    start_item(item);
    base_rand_ass:assert(item.randomize() with {rd_wr == 0; address inside {[0:'h3E]};});
    finish_item(item);
endtask :body
"""

READ_TEST = """
class i2c_master_read_test extends i2c_master_base_test;
    `uvm_component_utils(i2c_master_read_test)
    i2c_master_read_seq rd_seq;
    extern function new(string name = "i2c_master_read_test", uvm_component parent);
    extern function void build_phase(uvm_phase phase);
    extern task run_phase(uvm_phase phase);
endclass :i2c_master_read_test

function void i2c_master_read_test::build_phase(uvm_phase phase);
    super.build_phase(phase);
    rd_seq = i2c_master_read_seq::type_id::create("rd_seq");
endfunction :build_phase

task i2c_master_read_test::run_phase(uvm_phase phase);
    super.run_phase(phase);
    phase.raise_objection(this);
    repeat(40) begin
        rd_seq.start(mstr_env.mstr_agnt.sqr);
    end
    phase.drop_objection(this);
endtask :run_phase
"""

SEQUENCER = """
class i2c_master_sequencer extends uvm_sequencer #(i2c_master_sequence_item);
    `uvm_component_utils(i2c_master_sequencer)
    extern function new(string name = "i2c_master_sequencer", uvm_component parent);
endclass :i2c_master_sequencer

function i2c_master_sequencer::new(string name = "i2c_master_sequencer", uvm_component parent);
    super.new(name, parent);
endfunction :new
"""

DRIVER = """
class i2c_master_driver extends uvm_driver #(i2c_master_sequence_item);
    `uvm_component_utils(i2c_master_driver)
    i2c_master_agent_config drv_confg;
    virtual i2c_master_if vif;
    extern task run_phase(uvm_phase phase);
    extern task drive(input i2c_master_sequence_item item);
endclass :i2c_master_driver

task i2c_master_driver::run_phase(uvm_phase phase);
    super.run_phase(phase);
    initialize_dut();
    forever begin
        seq_item_port.get_next_item(drv_item);
        drive(drv_item);
        seq_item_port.item_done();
    end
endtask :run_phase

task i2c_master_driver::drive(input i2c_master_sequence_item item);
    wait(!(vif.rst));
    vif.start <= 1'b1;
    vif.address <= item.address;
    vif.rd_wr <= item.rd_wr;
endtask :drive
"""


def _chunk(relative_path: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c_{relative_path}",
        project_id="rtl_uvm",
        relative_path=relative_path,
        chunk_type="sv_class",
        text=text,
        line_start=1,
        line_end=40,
        fused_score=1.0,
        signal_scores={},
    )


EVIDENCE = [
    _chunk("tb/i2c_master_seq_lib/i2c_master_read_seq.svh", READ_SEQ),
    _chunk("tb/i2c_master_test_lib/i2c_master_read_test.svh", READ_TEST),
    _chunk("tb/i2c_master_agent/i2c_master_sequencer.svh", SEQUENCER),
    _chunk("tb/i2c_master_agent/i2c_master_driver.svh", DRIVER),
]


def test_fully_grounded_answer_with_citations_is_grounded():
    answer = (
        "The read sequence constrains rd_wr == 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 20-26]. "
        "i2c_master_read_test starts the read sequence on mstr_env.mstr_agnt.sqr "
        "[Source: tb/i2c_master_test_lib/i2c_master_read_test.svh, lines 28-34]."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.fabricated_citations == []


def test_unsupported_class_is_blocked():
    answer = "The i2c_master_test starts the read sequence on the sequencer."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_test" in report.unsupported_entities


def test_unsupported_qualified_method_is_blocked():
    # The exact R4 fabrication: the sequencer has no run_phase.
    answer = "The i2c_master_sequencer::run_phase drives the transaction."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_sequencer::run_phase" in report.unsupported_entities


def test_unsupported_relationship_is_blocked():
    answer = "i2c_master_read_seq extends i2c_master_fake_base_seq."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_fake_base_seq" in report.unsupported_entities


def test_missing_citation_on_source_specific_answer_is_insufficient():
    answer = "i2c_master_read_seq constrains rd_wr to 0 in its body."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "insufficient_evidence"
    assert report.valid_citations == 0
    assert "i2c_master_read_seq" in report.present_source_specific


def test_correct_citation_is_grounded():
    answer = (
        "i2c_master_read_seq constrains rd_wr to 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 20-26]."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.valid_citations == 1


def test_negative_query_does_not_fabricate():
    answer = (
        "There is no class named i2c_master_xyz_sequencer in the retrieved "
        "evidence, and no matching evidence was found."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.unsupported_entities == []


def test_mixed_answer_keeps_supported_portion():
    answer = (
        "i2c_master_read_seq constrains rd_wr == 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 20-26]. "
        "The i2c_master_fake_monitor validates all data."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_fake_monitor" in report.unsupported_entities

    supported = filter_supported_portion(answer, report, EVIDENCE)
    assert "i2c_master_fake_monitor" not in supported
    assert "i2c_master_read_seq constrains rd_wr" in supported
    assert "validates all data" not in supported


def test_fabricated_file_path_is_blocked():
    answer = "See rtl/nonexistent.sv for the driving logic."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert any("nonexistent.sv" in e for e in report.unsupported_entities)


def test_fabricated_citation_is_blocked():
    answer = "The driver drives per [Source: rtl/nonexistent.sv, lines 1-5]."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert report.fabricated_citations == ["rtl/nonexistent.sv"]


def test_generic_explanatory_statement_is_not_blocked():
    answer = "UVM uses sequences to generate transactions for the testbench."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.unsupported_entities == []


def test_case1_multi_symbol_grounded():
    # The DESIRED behavior case from the spec: all symbols are in evidence
    # and cited -> PASS.
    answer = (
        "i2c_master_read_test starts i2c_master_read_seq on "
        "mstr_env.mstr_agnt.sqr [Source: tb/i2c_master_test_lib/i2c_master_read_test.svh, lines 28-34]. "
        "i2c_master_driver drives the interface [Source: tb/i2c_master_agent/i2c_master_driver.svh, lines 20-31]."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"


def test_code_block_with_unsupported_symbol_is_dropped_whole():
    answer = (
        "The sequence is created in the test.\n\n```systemverilog\n"
        "class i2c_master_sequence extends uvm_sequence #(i2c_master_sequence_item);\n"
        "    virtual task body(); endtask\n"
        "endclass :i2c_master_sequence\n```\n\n"
        "The read sequence then constrains rd_wr == 0 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 20-26]."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_sequence" in report.unsupported_entities

    supported = filter_supported_portion(answer, report, EVIDENCE)
    assert "class i2c_master_sequence" not in supported
    assert "rd_wr == 0" in supported


def test_empty_answer_is_grounded_noop():
    report = check_grounding("", EVIDENCE)
    assert report.status == "grounded"


def test_evidence_required_for_symbols_absent_from_retrieved_chunks():
    # A symbol that exists in the repo but was NOT retrieved must still be
    # treated as unsupported (evidence-local rule, CASE 5).
    answer = "i2c_master_scoreboard checks the data."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_scoreboard" in report.unsupported_entities


def test_dot_path_not_in_evidence_is_blocked():
    answer = "The sequence starts on i2c_master_fake_monitor.sqr to reach the driver."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "unsupported"
    assert "i2c_master_fake_monitor.sqr" in report.unsupported_entities


def test_capitalized_prose_and_plural_are_not_flagged():
    # CASE: "Do not over-block generic explanatory language" — sentence
    # starts, headings, and plural forms of whitelisted words are prose.
    answer = (
        "The sequence classes extend the base sequence. "
        "Specific classes then constrain the transaction."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.unsupported_entities == []


def test_hex_literal_fragment_is_not_a_symbol():
    answer = (
        "The address is constrained to 7'h50 "
        "[Source: tb/i2c_master_seq_lib/i2c_master_read_seq.svh, lines 1-40]."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.unsupported_entities == []


def test_negative_answer_with_real_symbols_needs_no_citation():
    # CASE 4: a negative existence answer that names real classes but
    # cites nothing must stay grounded — it verified nothing is invented.
    answer = (
        "There is no class named i2c_master_xyz_sequencer. "
        "The classes defined are i2c_master_read_seq, i2c_master_write1_seq "
        "and i2c_master_monitor."
    )
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "grounded"
    assert report.valid_citations == 0
    assert report.unsupported_entities == []


def test_affirmative_answer_with_real_symbols_still_needs_citation():
    # Same entities, but an affirmative claim with zero citations is
    # insufficient (CASE 3) — the negative exemption must not leak.
    answer = "i2c_master_read_seq constrains rd_wr to 0 in its body."
    report = check_grounding(answer, EVIDENCE)
    assert report.status == "insufficient_evidence"