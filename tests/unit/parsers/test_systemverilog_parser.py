from datetime import datetime, timezone
from pathlib import Path

import pytest

from semrag.ingestion.parsers.systemverilog.parser import SystemVerilogParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _make_record(path: Path, category=FileCategory.RTL_VERILOG) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=f"RTL/{path.name}",
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=category,
        extension=path.suffix,
        is_binary=False,
    )


@pytest.fixture
def parser() -> SystemVerilogParser:
    return SystemVerilogParser()


def test_extracts_module_name_ports_and_parameters(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "i2c_master.sv"
    f.write_text(
        """
        module i2c_master #(
            parameter int CLK_FREQ = 100000000,
            parameter int I2C_FREQ = 400000
        )(
            input  logic clk,
            input  logic rst_n,
            output logic sda,
            input  logic [7:0] data_in
        );
        endmodule
        """
    )
    result = parser.parse(f, _make_record(f))

    assert result.success is True
    assert result.parser_name == "pyslang_syntax"
    modules = result.payload["modules"]
    assert len(modules) == 1
    mod = modules[0]
    assert mod["name"] == "i2c_master"

    port_names = {p["name"]: p for p in mod["ports"]}
    assert port_names["clk"]["direction"] == "input"
    assert port_names["sda"]["direction"] == "output"
    assert port_names["data_in"]["data_type"] == "logic [7:0]"

    param_names = {p["name"]: p for p in mod["parameters"]}
    assert param_names["CLK_FREQ"]["default_value"] == "100000000"
    assert param_names["I2C_FREQ"]["default_value"] == "400000"


def test_extracts_assertions_and_procedural_blocks(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "i2c_master.sv"
    f.write_text(
        """
        module i2c_master(input logic clk, input logic rst_n, output logic scl);
            logic start_condition;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) start_condition <= 1'b0;
            end

            assert property (@(posedge clk) disable iff (!rst_n) start_condition |-> ##[1:5] scl == 1'b0)
                else $error("start condition assertion failed");
        endmodule
        """
    )
    result = parser.parse(f, _make_record(f))
    mod = result.payload["modules"][0]

    assert len(mod["procedural_blocks"]) == 1
    assert mod["procedural_blocks"][0]["kind"] == "always_ff"

    assert len(mod["assertions"]) == 1
    assert mod["assertions"][0]["kind"] == "concurrent"
    assert "start_condition" in mod["assertions"][0]["raw_text"]


def test_extracts_nested_immediate_assertion(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "tb_check.sv"
    f.write_text(
        """
        module tb_check;
            logic clk;
            always_ff @(posedge clk) begin
                if (1) begin
                    assert (clk == 1'b0) else $error("nested check failed");
                end
            end
        endmodule
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    mod = result.payload["modules"][0]
    immediate = [a for a in mod["assertions"] if a["kind"] == "immediate"]
    assert len(immediate) == 1
    assert "nested check failed" in immediate[0]["raw_text"]


def test_extracts_module_instantiation(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "i2c_top.sv"
    f.write_text(
        """
        module i2c_top;
            logic clk;
            i2c_master #(.CLK_FREQ(100)) u_master (.clk(clk));
        endmodule
        """
    )
    result = parser.parse(f, _make_record(f))
    mod = result.payload["modules"][0]
    assert len(mod["instantiations"]) == 1
    assert mod["instantiations"][0]["module_type"] == "i2c_master"
    assert mod["instantiations"][0]["instance_name"] == "u_master"


def test_multiple_modules_in_one_file(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "multi.sv"
    f.write_text(
        """
        module i2c_master; endmodule
        module i2c_slave; endmodule
        """
    )
    result = parser.parse(f, _make_record(f))
    names = {m["name"] for m in result.payload["modules"]}
    assert names == {"i2c_master", "i2c_slave"}


def test_extracts_package(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "i2c_pkg.sv"
    f.write_text(
        """
        package i2c_pkg;
            typedef enum {IDLE, START} state_t;
        endpackage
        """
    )
    result = parser.parse(f, _make_record(f))
    assert len(result.payload["packages"]) == 1
    assert result.payload["packages"][0]["name"] == "i2c_pkg"


def test_missing_include_produces_diagnostic_but_still_succeeds(
    tmp_path: Path, parser: SystemVerilogParser
):
    """Proves the graceful-degradation story from the module docstring:
    an unresolvable `include (which would fail full elaboration) does not
    prevent syntax-level extraction of the module that's still present."""
    f = tmp_path / "i2c_master.sv"
    f.write_text(
        '`include "does_not_exist.svh"\n'
        "module i2c_master; endmodule\n"
    )
    result = parser.parse(f, _make_record(f))

    assert result.success is True
    assert result.has_errors is True
    assert any("CouldNotOpenIncludeFile" in d.message for d in result.diagnostics)
    assert result.payload["modules"][0]["name"] == "i2c_master"


def test_interface_is_flagged(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "i2c_if.sv"
    f.write_text(
        """
        interface i2c_if(input logic clk);
            logic sda;
            logic scl;
        endinterface
        """
    )
    result = parser.parse(f, _make_record(f))
    mod = result.payload["modules"][0]
    assert mod["name"] == "i2c_if"
    assert mod["is_interface"] is True


def test_unreadable_file_fails_gracefully(tmp_path: Path, parser: SystemVerilogParser):
    placeholder = tmp_path / "placeholder.sv"
    placeholder.write_text("module placeholder; endmodule\n")
    record = _make_record(placeholder)  # stat() needs a real file to build the record

    missing = tmp_path / "does_not_exist.sv"
    record = record.model_copy(update={"absolute_path": str(missing)})

    result = parser.parse(missing, record)
    assert result.success is False
    assert result.has_errors is True


def test_extracts_uvm_class_and_methods(tmp_path: Path, parser: SystemVerilogParser):
    """A UVM-style class with `extern` prototypes must be extracted into a
    ``classes`` payload entry carrying its methods (function vs task)."""
    f = tmp_path / "i2c_master_driver.svh"
    f.write_text(
        """
        class i2c_master_driver extends uvm_driver #(i2c_master_sequence_item);
            `uvm_component_utils(i2c_master_driver)

            extern function void build_phase(uvm_phase phase);
            extern task run_phase(uvm_phase phase);
            extern task drive(input i2c_master_sequence_item item);
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    assert result.success is True
    classes = result.payload["classes"]
    assert len(classes) == 1
    cls = classes[0]
    assert cls["name"] == "i2c_master_driver"
    # Parameterized base stripped to bare name.
    assert cls["extends"] == "uvm_driver"

    methods = {m["name"]: m["kind"] for m in cls["methods"]}
    assert methods["build_phase"] == "function"
    assert methods["run_phase"] == "task"
    assert methods["drive"] == "task"


def test_multiple_classes_in_one_file(tmp_path: Path, parser: SystemVerilogParser):
    f = tmp_path / "multi_class.svh"
    f.write_text(
        """
        class i2c_master_sequencer extends uvm_sequencer #(i2c_master_sequence_item);
            extern function new(string name = "i2c_master_sequencer", uvm_component parent);
        endclass

        class i2c_master_monitor extends uvm_monitor;
            extern task run_phase(uvm_phase phase);
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    assert result.success is True
    classes = {c["name"]: c for c in result.payload["classes"]}
    assert set(classes) == {"i2c_master_sequencer", "i2c_master_monitor"}
    assert classes["i2c_master_sequencer"]["methods"][0]["name"] == "new"
    assert classes["i2c_master_monitor"]["methods"][0]["kind"] == "task"


def test_extracts_class_inside_package(tmp_path: Path, parser: SystemVerilogParser):
    """Classes declared inside a ``package ... endpackage`` body (the standard
    UVM testbench style) must be extracted, not only top-level classes."""
    f = tmp_path / "i2c_master_seq_pkg.sv"
    f.write_text(
        """
        package i2c_master_seq_lib;
            import uvm_pkg::*;

            class i2c_master_read_seq extends uvm_sequence #(i2c_master_sequence_item);
                extern task body();
            endclass : i2c_master_read_seq
        endpackage : i2c_master_seq_lib
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    assert result.success is True
    assert [p["name"] for p in result.payload["packages"]] == ["i2c_master_seq_lib"]
    classes = {c["name"]: c for c in result.payload["classes"]}
    assert set(classes) == {"i2c_master_read_seq"}
    assert [m["name"] for m in classes["i2c_master_read_seq"]["methods"]] == ["body"]


def test_extracts_inline_task_and_function_as_methods(tmp_path: Path, parser: SystemVerilogParser):
    """Inline method implementations (``ClassMethodDeclaration``) — not just
    ``extern`` prototypes — must be captured as methods."""
    f = tmp_path / "seq.svh"
    f.write_text(
        """
        class i2c_master_read_seq;
            task body();
                int item;
            endtask : body
            function new(string name = "i2c_master_read_seq");
            endfunction : new
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    cls = result.payload["classes"][0]
    methods = {m["name"]: m["kind"] for m in cls["methods"]}
    assert methods == {"body": "task", "new": "function"}
    assert cls["extends"] is None  # no extends clause


def test_class_with_inline_body_and_run_phase(tmp_path: Path, parser: SystemVerilogParser):
    """A UVM-style class whose ``body()`` / ``run_phase()`` are written inline
    must expose them as methods."""
    f = tmp_path / "i2c_master_read_test.svh"
    f.write_text(
        """
        class i2c_master_read_test extends uvm_test;
            virtual task run_phase(uvm_phase phase);
                phase.raise_objection(this);
                #100;
                phase.drop_objection(this);
            endtask : run_phase
        endclass

        class i2c_master_read_seq extends uvm_sequence #(i2c_master_sequence_item);
            task body();
                i2c_master_sequence_item item;
                start_item(item);
                finish_item(item);
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    classes = {c["name"]: c for c in result.payload["classes"]}
    assert "run_phase" in {m["name"] for m in classes["i2c_master_read_test"]["methods"]}
    assert "body" in {m["name"] for m in classes["i2c_master_read_seq"]["methods"]}


def test_extern_prototype_and_inline_implementation_no_duplicate(
    tmp_path: Path, parser: SystemVerilogParser
):
    """An ``extern`` prototype followed by its inline implementation in the
    same class must yield one method, not two."""
    f = tmp_path / "i2c_master_driver.svh"
    f.write_text(
        """
        class i2c_master_driver;
            extern task body();
            task body();
                int x;
            endtask : body
            extern function new();
            function new();
            endfunction : new
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    cls = result.payload["classes"][0]
    names = [m["name"] for m in cls["methods"]]
    assert names.count("body") == 1
    assert names.count("new") == 1


def test_package_recursion_preserves_top_level_extraction(
    tmp_path: Path, parser: SystemVerilogParser
):
    """A mixed file with top-level module/interface/package plus classes both
    at top level and inside a package extracts everything."""
    f = tmp_path / "mixed.sv"
    f.write_text(
        """
        module i2c_master(input logic clk, output logic scl);
        endmodule
        interface i2c_if(input logic clk);
        endinterface
        package pkg;
            class i2c_master_read_seq;
            endclass
        endpackage
        class i2c_master_top_test;
        endclass
        """
    )
    result = parser.parse(f, _make_record(f))

    modules = {m["name"]: m["is_interface"] for m in result.payload["modules"]}
    assert modules == {"i2c_master": False, "i2c_if": True}
    assert [p["name"] for p in result.payload["packages"]] == ["pkg"]
    classes = {c["name"] for c in result.payload["classes"]}
    assert classes == {"i2c_master_read_seq", "i2c_master_top_test"}


def test_class_type_refs_extracted_from_properties_and_parameterized_types(
    tmp_path: Path, parser: SystemVerilogParser
):
    """Static project-local type references in class properties must be
    captured as ``type_refs`` — simple named types, parameterized named
    types (bare + argument), and parameterized extends-clause arguments."""
    f = tmp_path / "refs.svh"
    f.write_text(
        """
        class i2c_master_read_seq extends i2c_master_base_seq;
            i2c_master_sequence_item item;
            i2c_master_sequencer #(i2c_master_sequence_item) sqr;
            int counter;
            extern task body();
        endclass

        class i2c_master_sequencer extends uvm_sequencer #(i2c_master_sequence_item);
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    classes = {c["name"]: c for c in result.payload["classes"]}

    read_seq = classes["i2c_master_read_seq"]
    # property named type + parameterized property bare type + its arg; builtin int skipped.
    assert read_seq["type_refs"] == ["i2c_master_sequence_item", "i2c_master_sequencer"]

    sequencer = classes["i2c_master_sequencer"]
    # parameterized extends-clause argument captured; uvm_sequencer itself is
    # the extends base (handled separately), not a type_ref.
    assert sequencer["type_refs"] == ["i2c_master_sequence_item"]
    assert sequencer["extends"] == "uvm_sequencer"


def test_simple_function_call_captured(tmp_path: Path, parser: SystemVerilogParser):
    """An unqualified call ``obj.do_something();`` inside a method body must
    be captured with method name and no receiver."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                obj.do_something();
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == [{"method": "do_something", "receiver": "obj", "kind": "method"}]


def test_member_call_preserves_receiver(tmp_path: Path, parser: SystemVerilogParser):
    """``seq.start(sqr);`` must preserve receiver=seq, method=start."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task run();
                seq.start(sqr);
            endtask : run
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    run = result.payload["classes"][0]["methods"][0]
    assert run["calls"] == [{"method": "start", "receiver": "seq", "kind": "method"}]


def test_nested_member_call_captured(tmp_path: Path, parser: SystemVerilogParser):
    """``seq_item_port.get_next_item(item);`` — receiver is the leftmost
    identifier of the member chain."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task run();
                seq_item_port.get_next_item(item);
            endtask : run
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    run = result.payload["classes"][0]["methods"][0]
    assert run["calls"] == [{"method": "get_next_item", "receiver": "seq_item_port", "kind": "method"}]


def test_multiple_calls_in_one_method(tmp_path: Path, parser: SystemVerilogParser):
    """A method with several call sites returns all of them in source order."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                start_item(item);
                finish_item(item);
                seq.start(sqr);
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == [
        {"method": "start_item", "receiver": None, "kind": "method"},
        {"method": "finish_item", "receiver": None, "kind": "method"},
        {"method": "start", "receiver": "seq", "kind": "method"},
    ]


def test_type_id_create_captured(tmp_path: Path, parser: SystemVerilogParser):
    """``i2c_master_sequence_item::type_id::create(\"item\")`` must be captured
    as kind=type_id_create with receiver = the created class."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                i2c_master_sequence_item::type_id::create("item");
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == [
        {"method": "create", "receiver": "i2c_master_sequence_item", "kind": "type_id_create"}
    ]


def test_external_uvm_method_call_captured(tmp_path: Path, parser: SystemVerilogParser):
    """``phase.raise_objection(this);`` — external UVM API call captured
    structurally (receiver=phase, method=raise_objection)."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task run_phase(uvm_phase phase);
                phase.raise_objection(this);
            endtask : run_phase
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    run_phase = result.payload["classes"][0]["methods"][0]
    assert run_phase["calls"] == [
        {"method": "raise_objection", "receiver": "phase", "kind": "method"}
    ]


def test_no_method_calls_produces_empty_calls(tmp_path: Path, parser: SystemVerilogParser):
    """A method with no call sites must have calls == []."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                int x;
                x = 5;
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == []


def test_calls_inside_task_body(tmp_path: Path, parser: SystemVerilogParser):
    """Calls nested inside a ``begin/end`` block within a task body must be
    found."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task run_phase(uvm_phase phase);
                if(active) begin
                    seq.start(sqr);
                end
            endtask : run_phase
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    run_phase = result.payload["classes"][0]["methods"][0]
    assert run_phase["calls"] == [{"method": "start", "receiver": "seq", "kind": "method"}]


def test_calls_inside_function_body(tmp_path: Path, parser: SystemVerilogParser):
    """A call inside a function body must be captured."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            function void build_phase(uvm_phase phase);
                super.build_phase(phase);
            endfunction : build_phase
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    build_phase = result.payload["classes"][0]["methods"][0]
    assert build_phase["calls"] == [
        {"method": "build_phase", "receiver": "super", "kind": "method"}
    ]


def test_duplicate_calls_deduplicated_deterministically(tmp_path: Path, parser: SystemVerilogParser):
    """Repeated identical calls must appear once; identical method on
    different receivers must both be kept."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                start_item(item);
                start_item(item);
                a.start();
                b.start();
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == [
        {"method": "start_item", "receiver": None, "kind": "method"},
        {"method": "start", "receiver": "a", "kind": "method"},
        {"method": "start", "receiver": "b", "kind": "method"},
    ]


def test_system_task_not_captured(tmp_path: Path, parser: SystemVerilogParser):
    """System tasks ($display etc.) are not method calls and must be skipped."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class probe;
            task body();
                $display("hello");
            endtask : body
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    body = result.payload["classes"][0]["methods"][0]
    assert body["calls"] == []


def test_extern_method_definition_calls_merged_into_class_method(
    tmp_path: Path, parser: SystemVerilogParser
):
    """UVM's declare-in-class / define-after pattern: the ``extern`` prototype
    in the class body has no calls, but the top-level ``Class::method``
    definition carries them. After extraction both must be merged into the
    single ParsedMethod."""
    f = tmp_path / "s.svh"
    f.write_text(
        """
        class i2c_master_read_seq extends i2c_master_base_seq;
            i2c_master_sequence_item item;
            extern task body();
        endclass

        task i2c_master_read_seq::body();
            item = i2c_master_sequence_item::type_id::create("item");
            start_item(item);
            finish_item(item);
        endtask : body
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))
    seq = result.payload["classes"][0]
    assert seq["name"] == "i2c_master_read_seq"
    body = next(m for m in seq["methods"] if m["name"] == "body")
    assert body["calls"] == [
        {"method": "create", "receiver": "i2c_master_sequence_item", "kind": "type_id_create"},
        {"method": "start_item", "receiver": None, "kind": "method"},
        {"method": "finish_item", "receiver": None, "kind": "method"},
    ]


def test_class_members_extracted_from_properties(tmp_path: Path, parser: SystemVerilogParser):
    """R2E-A: every property declaration with a named type must be captured as
    a ``members`` entry (name -> bare declared type). Builtin types are
    skipped. Order follows source order."""
    f = tmp_path / "agent.svh"
    f.write_text(
        """
        class i2c_master_agent;
            i2c_master_agent_config cfg;
            i2c_master_sequencer sqr;
            i2c_master_driver drv;
            i2c_master_monitor mon;
            int counter;
            extern task build_phase(uvm_phase phase);
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    cls = result.payload["classes"][0]
    assert cls["members"] == [
        {"name": "cfg", "type": "i2c_master_agent_config"},
        {"name": "sqr", "type": "i2c_master_sequencer"},
        {"name": "drv", "type": "i2c_master_driver"},
        {"name": "mon", "type": "i2c_master_monitor"},
    ]


def test_virtual_interface_member_and_interface_ref_extracted(
    tmp_path: Path, parser: SystemVerilogParser
):
    """R2E-D: a ``virtual i2c_master_if vif;`` property must be captured both as
    a member (name -> interface type) and as an ``interface_refs`` entry."""
    f = tmp_path / "driver.svh"
    f.write_text(
        """
        class i2c_master_driver;
            virtual i2c_master_if vif;
        endclass
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    cls = result.payload["classes"][0]
    assert {"name": "vif", "type": "i2c_master_if"} in cls["members"]
    assert cls["interface_refs"] == ["i2c_master_if"]


def test_connect_captured_from_method_body(tmp_path: Path, parser: SystemVerilogParser):
    """R2E-B: ``drv.seq_item_port.connect(sqr.seq_item_export);`` inside a
    method body must be captured as a ``connects`` entry (receiver = leftmost
    of callee path, target = leftmost of first argument path)."""
    f = tmp_path / "agent.svh"
    f.write_text(
        """
        class i2c_master_agent;
            extern function void connect_phase(uvm_phase phase);
        endclass
        function void i2c_master_agent::connect_phase(uvm_phase phase);
            drv.seq_item_port.connect(sqr.seq_item_export);
        endfunction : connect_phase
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    connect_phase = result.payload["classes"][0]["methods"][0]
    assert connect_phase["connects"] == [{"receiver": "drv", "target": "sqr"}]


def test_connect_with_this_target_skipped(tmp_path: Path, parser: SystemVerilogParser):
    """R2E-B guard: ``mon.mon_port.connect(this.mon2agnt_port);`` — the target
    path's leftmost identifier is the special ``this`` handle, not a member, so
    it must be skipped rather than emitted as a receiver/target pair."""
    f = tmp_path / "agent.svh"
    f.write_text(
        """
        class i2c_master_agent;
            extern function void connect_phase(uvm_phase phase);
        endclass
        function void i2c_master_agent::connect_phase(uvm_phase phase);
            mon.mon_port_2.connect(this.mon2agnt_port_2);
        endfunction : connect_phase
        """
    )
    result = parser.parse(f, _make_record(f, category=FileCategory.TESTBENCH_UVM))

    connect_phase = result.payload["classes"][0]["methods"][0]
    assert connect_phase["connects"] == []
