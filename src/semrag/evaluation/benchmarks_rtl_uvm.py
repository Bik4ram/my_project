"""Retrieval ground-truth benchmarks for the ``rtl_uvm`` project.

Curated for the I2C master UVM testbench corpus at
``/home/bikram/projects/RTL_UVM`` (rtl/, tb/, sim/, doc/). The
``relevant_chunk_ids`` are taken verbatim from the deterministic chunk
IDs produced by a fresh offline scan -> parse -> IR -> chunk run with
the current code (467 chunks total, no embeddings, no retriever). Every
ID was judged against the actual chunk content in that dump; the
retriever was never used to assign labels.

``repo_area`` is ``GENERAL`` for every query because ``RepoArea``
describes the semrag platform's own modules, which do not apply to the
external hardware corpus. ``expected_intent`` carries the meaningful
label (``RTL``, ``VERIFICATION``, ``DEBUG``, ``CONFIG``, ``GENERAL``).

Category targets (guidelines, not quotas): code navigation 5-8, design
explanation 5-8, debugging 5-8, assertion/verification 4-6,
bus/parameter ("register") 4-6, UVM/testbench 4-6, log investigation
3-5, terminology mismatch 3-5.
"""

from __future__ import annotations

from semrag.evaluation.models import ExpectedIntent, GoldenQuery, RepoArea

PROJECT_ID = "rtl_uvm"

RTL_SIG = "rtl_uvm::rtl/i2c_master.sv::sv_module_signature::i2c_master"
RTL_BODY_0 = "rtl_uvm::rtl/i2c_master.sv::sv_body_window::i2c_master::0"
RTL_BODY_1 = "rtl_uvm::rtl/i2c_master.sv::sv_body_window::i2c_master::1"
RTL_BODY_2 = "rtl_uvm::rtl/i2c_master.sv::sv_body_window::i2c_master::2"
README_0 = "rtl_uvm::README.md::markdown_section::0"
README_1 = "rtl_uvm::README.md::markdown_section::1"
README_2 = "rtl_uvm::README.md::markdown_section::2"
README_3 = "rtl_uvm::README.md::markdown_section::3"

CODE_COV_4 = "rtl_uvm::sim/coverage/code_coverage.txt::generic_window::4"
CODE_COV_6 = "rtl_uvm::sim/coverage/code_coverage.txt::generic_window::6"
CODE_COV_7 = "rtl_uvm::sim/coverage/code_coverage.txt::generic_window::7"
FUN_COV_0 = "rtl_uvm::sim/coverage/fun_coverage.txt::generic_window::0"
FUN_COV_1 = "rtl_uvm::sim/coverage/fun_coverage.txt::generic_window::1"
ASSERTIONS_0 = "rtl_uvm::sim/coverage/assertions.txt::generic_window::0"

TOP_SV = "rtl_uvm::tb/i2c_master_top.sv::sv_module::i2c_master_top"
IF_SV = "rtl_uvm::tb/i2c_master_if.sv::sv_module::i2c_master_if"
PARAM_PKG = "rtl_uvm::tb/i2c_master_param_pkg.sv::sv_package::i2c_master_param_pkg"
PKG_SV = "rtl_uvm::tb/i2c_master_pkg.sv::sv_package::i2c_master_pkg"
AGENT_PKG = "rtl_uvm::tb/i2c_master_agent/i2c_master_agent_pkg.sv::sv_package::i2c_master_agent_pkg"
SEQ_PKG = "rtl_uvm::tb/i2c_master_seq_lib/i2c_master_seq_pkg.sv::sv_package::i2c_master_seq_pkg"
TEST_PKG = "rtl_uvm::tb/i2c_master_test_lib/i2c_master_test_pkg.sv::sv_package::i2c_master_test_pkg"

AGENT_0 = "rtl_uvm::tb/i2c_master_agent/i2c_master_agent.svh::generic_window::0"
AGENT_1 = "rtl_uvm::tb/i2c_master_agent/i2c_master_agent.svh::generic_window::1"
AGENT_CFG = "rtl_uvm::tb/i2c_master_agent/i2c_master_agent_config.svh::generic_window::0"
DRIVER_0 = "rtl_uvm::tb/i2c_master_agent/i2c_master_driver.svh::generic_window::0"
DRIVER_1 = "rtl_uvm::tb/i2c_master_agent/i2c_master_driver.svh::generic_window::1"
MONITOR_0 = "rtl_uvm::tb/i2c_master_agent/i2c_master_monitor.svh::generic_window::0"
MONITOR_1 = "rtl_uvm::tb/i2c_master_agent/i2c_master_monitor.svh::generic_window::1"
STAT_MON = "rtl_uvm::tb/i2c_master_agent/i2c_master_stat_monitor.svh::generic_window::0"
SEQ_ITEM = "rtl_uvm::tb/i2c_master_sequence_item.svh::generic_window::0"
ENV_CFG = "rtl_uvm::tb/i2c_master_env_config.svh::generic_window::0"
ENV_0 = "rtl_uvm::tb/i2c_master_environment.svh::generic_window::0"
ENV_1 = "rtl_uvm::tb/i2c_master_environment.svh::generic_window::1"
SCB_0 = "rtl_uvm::tb/i2c_master_scoreboard.svh::generic_window::0"
SCB_1 = "rtl_uvm::tb/i2c_master_scoreboard.svh::generic_window::1"
COV_0 = "rtl_uvm::tb/i2c_master_coverage_collector.svh::generic_window::0"
COV_1 = "rtl_uvm::tb/i2c_master_coverage_collector.svh::generic_window::1"
COV_2 = "rtl_uvm::tb/i2c_master_coverage_collector.svh::generic_window::2"
BASE_SEQ = "rtl_uvm::tb/i2c_master_seq_lib/i2c_master_base_seq.svh::generic_window::0"
WRITE1_SEQ = "rtl_uvm::tb/i2c_master_seq_lib/i2c_master_write1_seq.svh::generic_window::0"
WRITE2_SEQ = "rtl_uvm::tb/i2c_master_seq_lib/i2c_master_write2_seq.svh::generic_window::0"
BASE_TEST = "rtl_uvm::tb/i2c_master_test_lib/i2c_master_base_test.svh::generic_window::0"
WRITE1_TEST = "rtl_uvm::tb/i2c_master_test_lib/i2c_master_write1_test.svh::generic_window::0"
WRITE2_TEST = "rtl_uvm::tb/i2c_master_test_lib/i2c_master_write2_test.svh::generic_window::0"

LOG1_0 = "rtl_uvm::sim/log/write1_test_log.log::log::0"
LOG1_2 = "rtl_uvm::sim/log/write1_test_log.log::log::2"
LOG1_22 = "rtl_uvm::sim/log/write1_test_log.log::log::22"
LOG1_24 = "rtl_uvm::sim/log/write1_test_log.log::log::24"
LOG1_26 = "rtl_uvm::sim/log/write1_test_log.log::log::26"
LOG1_205 = "rtl_uvm::sim/log/write1_test_log.log::log::205"
LOG2_205 = "rtl_uvm::sim/log/write2_test_log.log::log::205"


def _gq(
    query_id: str,
    query_text: str,
    relevant_chunk_ids: set[str],
    primary_files: set[str],
    expected_intent: ExpectedIntent,
    description: str,
    secondary_files: set[str] | None = None,
) -> GoldenQuery:
    return GoldenQuery(
        query_id=query_id,
        project_id=PROJECT_ID,
        query_text=query_text,
        relevant_chunk_ids=relevant_chunk_ids,
        primary_files=primary_files,
        secondary_files=secondary_files or set(),
        expected_intent=expected_intent,
        repo_area=RepoArea.GENERAL,
        description=description,
    )


BENCHMARKS_RTL_UVM: list[GoldenQuery] = [
    # ── Code navigation ──────────────────────────────────────────────────
    _gq(
        "rtluvm_nav_01",
        "Where is the i2c_master module declared and what are its input/output ports?",
        {RTL_SIG},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "Module signature chunk contains the full port list (clk, rst, start, address, rd_wr, din, stop, dout, SDA, SCL).",
    ),
    _gq(
        "rtluvm_nav_02",
        "Where is the i2c_master FSM defined? Show the state enumeration and the clocked always block.",
        {RTL_SIG, RTL_BODY_0},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "State enum (IDLE..STOP) is in the signature chunk; the clocked always block with the reset branch is in body window 0.",
    ),
    _gq(
        "rtluvm_nav_03",
        "Where is the DUT instantiated in the testbench top module?",
        {TOP_SV},
        {"tb/i2c_master_top.sv"},
        ExpectedIntent.VERIFICATION,
        "i2c_master_top instantiates the i2c_master_if and the i2c_master DUT (mstr_dut) and sets the vif via uvm_config_db.",
    ),
    _gq(
        "rtluvm_nav_04",
        "Where is the virtual interface defined? What signals does it carry?",
        {IF_SV, TOP_SV},
        {"tb/i2c_master_if.sv", "tb/i2c_master_top.sv"},
        ExpectedIntent.VERIFICATION,
        "i2c_master_if declares the modport/interface signals (clk, rst, start, address, rd_wr, din, stop, dout, SDA, SCL); the top declares it and binds it to the DUT.",
    ),
    _gq(
        "rtluvm_nav_05",
        "Where is the agent defined and how does it create its sequencer, driver and monitors?",
        {AGENT_0, AGENT_1},
        {"tb/i2c_master_agent/i2c_master_agent.svh"},
        ExpectedIntent.VERIFICATION,
        "Agent class declaration and build_phase (creates sqr, drv, mon, mon2) live in window 0; connect_phase wiring is in window 1.",
    ),
    _gq(
        "rtluvm_nav_06",
        "Where is the driver's drive task that drives SDA/SCL and the reset initialization?",
        {DRIVER_1, DRIVER_0},
        {"tb/i2c_master_agent/i2c_master_driver.svh"},
        ExpectedIntent.VERIFICATION,
        "drive task (start/address/data timing on vif) is in driver window 1; initialize_dut reset stimulus is in window 0.",
    ),
    _gq(
        "rtluvm_nav_07",
        "Where are the write1 and write2 sequence classes defined?",
        {WRITE1_SEQ, WRITE2_SEQ},
        {"tb/i2c_master_seq_lib/i2c_master_write1_seq.svh", "tb/i2c_master_seq_lib/i2c_master_write2_seq.svh"},
        ExpectedIntent.VERIFICATION,
        "Each sequence body sets rd_wr==1 and constrains the address and din ranges.",
    ),
    _gq(
        "rtluvm_nav_08",
        "Where are the write1 and write2 UVM test classes defined?",
        {WRITE1_TEST, WRITE2_TEST},
        {"tb/i2c_master_test_lib/i2c_master_write1_test.svh", "tb/i2c_master_test_lib/i2c_master_write2_test.svh"},
        ExpectedIntent.VERIFICATION,
        "Each test's run_phase starts its sequence repeatedly (40 transactions) with an objection.",
    ),
    # ── Design explanation ───────────────────────────────────────────────
    _gq(
        "rtluvm_des_01",
        "How does the i2c_master generate a START condition on the I2C bus?",
        {README_2, RTL_BODY_0},
        {"README.md", "rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "README section 2 describes the start condition; the RTL START state pulls sda low on the high SCL line.",
    ),
    _gq(
        "rtluvm_des_02",
        "How is an I2C frame formatted (start, address byte, ACK, data, STOP)?",
        {README_1, RTL_BODY_0, RTL_BODY_1, RTL_BODY_2},
        {"README.md", "rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "Frame format diagram (README) plus the ADDR/ADDR_REG, WACK/SACK, WRITE_DATA, and STOP states that implement it.",
    ),
    _gq(
        "rtluvm_des_03",
        "How is the 7-bit slave address combined with the read/write bit and shifted out?",
        {RTL_BODY_0},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "START state latches tx_reg = {address, rd_wr}; ADDR/ADDR_REG shift it out LSB-first on SDA while toggling SCL.",
    ),
    _gq(
        "rtluvm_des_04",
        "How is a byte of write data shifted out bit by bit after the address ACK?",
        {RTL_BODY_1},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "WRITE_DATA/WRITE_DATA_REG states drive sda = tx_reg[~count] on the falling/rising edges of SCL; WACK/SACK follow each byte.",
    ),
    _gq(
        "rtluvm_des_05",
        "How does the master read data back from the slave on SDA into dout?",
        {RTL_BODY_2},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "READ_DATA samples SDA into tx_reg, READ_DATA_2 asserts ACK (sda=0), READ_DATA_REG transfers tx_reg to dout.",
    ),
    _gq(
        "rtluvm_des_06",
        "How is the STOP condition generated and how is the bus released?",
        {README_3, RTL_BODY_2},
        {"README.md", "rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "README section 3 defines STOP (SDA rising while SCL high); WSTOP_2/STOP states raise sda and return to IDLE.",
    ),
    _gq(
        "rtluvm_des_07",
        "How does the master handle ACK/NACK from the slave after transmitting a byte?",
        {RTL_BODY_1},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.RTL,
        "WACK/SACK states sample ack on the SCL rising edge and branch to IDLE (NACK), WSTOP_1 (READ), or WRITE_DATA (WRITE).",
    ),
    _gq(
        "rtluvm_des_08",
        "What is the I2C protocol as documented in the project README?",
        {README_0},
        {"README.md"},
        ExpectedIntent.GENERAL,
        "README section 0 introduces the I2C protocol and the SDA/SCL lines with ACK/NACK semantics.",
    ),
    # ── Debugging ────────────────────────────────────────────────────────
    _gq(
        "rtluvm_dbg_01",
        "Why is FSM coverage only 66.6%? Which states or transitions were never covered?",
        {CODE_COV_6, CODE_COV_7},
        {"sim/coverage/code_coverage.txt"},
        ExpectedIntent.DEBUG,
        "FSM Coverage summary is in code_coverage window 6; the Uncovered States/Transitions lists are in window 7.",
    ),
    _gq(
        "rtluvm_dbg_02",
        "Why was the READ_DATA path never exercised during the write regression?",
        {CODE_COV_4, CODE_COV_7},
        {"sim/coverage/code_coverage.txt"},
        ExpectedIntent.DEBUG,
        "The condition ack==1 && mode==READ (RTL line 78) shows All False in window 4; the uncovered READ_DATA* states are listed in window 7.",
    ),
    _gq(
        "rtluvm_dbg_03",
        "Which RTL branches or conditions were never taken (All False) during coverage?",
        {CODE_COV_4},
        {"sim/coverage/code_coverage.txt"},
        ExpectedIntent.DEBUG,
        "code_coverage window 4 lists branch/condition rows with All False Count, e.g. the ack==1 && mode==READ read-path decision.",
    ),
    _gq(
        "rtluvm_dbg_04",
        "How does the scoreboard detect an address mismatch and report ADDR_FAIL?",
        {SCB_0},
        {"tb/i2c_master_scoreboard.svh"},
        ExpectedIntent.DEBUG,
        "write_addr in scoreboard window 0 compares item.address vs item.address_check and issues uvm_error ADDR_FAIL on mismatch.",
    ),
    _gq(
        "rtluvm_dbg_05",
        "Where would DATA_FAIL and STOP_FAIL be reported, and what data do they compare?",
        {SCB_1},
        {"tb/i2c_master_scoreboard.svh"},
        ExpectedIntent.DEBUG,
        "write_data (din vs data_check) and write_stop (SDA/stop conditions) are in scoreboard window 1.",
    ),
    _gq(
        "rtluvm_dbg_06",
        "Were there any UVM_ERROR or UVM_FATAL in the write1 test run?",
        {LOG1_205},
        {"sim/log/write1_test_log.log"},
        ExpectedIntent.DEBUG,
        "write1 log chunk 205 is the UVM Report Summary: 1885 UVM_INFO, 0 WARNING/ERROR/FATAL.",
    ),
    _gq(
        "rtluvm_dbg_07",
        "How many times did ADDR_PASS, DATA_PASS and STOP_PASS fire in the write1 run?",
        {LOG1_205, LOG1_22, LOG1_24, LOG1_26},
        {"sim/log/write1_test_log.log"},
        ExpectedIntent.DEBUG,
        "Report counts in the summary chunk (40 each) plus representative address/data/stop match event chunks.",
    ),
    _gq(
        "rtluvm_dbg_08",
        "What reset stimulus does the driver apply to the interface before a transaction?",
        {DRIVER_0},
        {"tb/i2c_master_agent/i2c_master_driver.svh"},
        ExpectedIntent.DEBUG,
        "initialize_dut in driver window 0 drives rst=1 and clears the interface signals.",
    ),
    # ── Assertion / verification ─────────────────────────────────────────
    _gq(
        "rtluvm_asr_01",
        "Did any assertions fail during the write1 and write2 regression?",
        {ASSERTIONS_0},
        {"sim/coverage/assertions.txt"},
        ExpectedIntent.VERIFICATION,
        "assertions.txt reports the assertion results: base_rand_ass passes with 0 failures for both sequences.",
    ),
    _gq(
        "rtluvm_asr_02",
        "How many times did the base_rand_ass assertion pass per sequence in the regression?",
        {ASSERTIONS_0, WRITE1_SEQ, WRITE2_SEQ},
        {"sim/coverage/assertions.txt", "tb/i2c_master_seq_lib/i2c_master_write1_seq.svh", "tb/i2c_master_seq_lib/i2c_master_write2_seq.svh"},
        ExpectedIntent.VERIFICATION,
        "assertions.txt shows the pass counts; the sequence bodies contain the base_rand_ass assertion on randomize.",
    ),
    _gq(
        "rtluvm_asr_03",
        "How does the coverage collector define the COVER_STATES covergroup and its bins?",
        {COV_0},
        {"tb/i2c_master_coverage_collector.svh"},
        ExpectedIntent.VERIFICATION,
        "COVER_STATES covergroup with IDLE_to_IDLE etc. transition bins is declared in coverage_collector window 0.",
    ),
    _gq(
        "rtluvm_asr_04",
        "What was the functional coverage result for COVER_STATES in the write regression?",
        {FUN_COV_0, FUN_COV_1},
        {"sim/coverage/fun_coverage.txt"},
        ExpectedIntent.VERIFICATION,
        "fun_coverage window 0 lists the COVER_STATES bin hits (100%); window 1 has the total covergroup coverage.",
    ),
    _gq(
        "rtluvm_asr_05",
        "What assertions are bound inside the UVM sequences and where do their results land?",
        {WRITE1_SEQ, WRITE2_SEQ, ASSERTIONS_0},
        {"tb/i2c_master_seq_lib/i2c_master_write1_seq.svh", "tb/i2c_master_seq_lib/i2c_master_write2_seq.svh", "sim/coverage/assertions.txt"},
        ExpectedIntent.VERIFICATION,
        "base_rand_ass is defined in both sequence bodies; assertions.txt tabulates its pass/fail outcomes.",
    ),
    # ── Bus / parameter ("register") ─────────────────────────────────────
    _gq(
        "rtluvm_bus_01",
        "How is the 7-bit slave address port used to build the first transmitted byte?",
        {RTL_SIG, RTL_BODY_0},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.CONFIG,
        "address [6:0] port in the signature is packed as tx_reg = {address, rd_wr} in the START state.",
    ),
    _gq(
        "rtluvm_bus_02",
        "What are the din and dout data widths and how does data flow from din to dout?",
        {RTL_SIG, RTL_BODY_1, RTL_BODY_2},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.CONFIG,
        "din[7:0]/dout[7:0] in the signature; WRITE_DATA loads tx_reg=din and READ_DATA_REG sets dout=tx_reg.",
    ),
    _gq(
        "rtluvm_bus_03",
        "How does the rd_wr pin select read vs write mode, and what are the READ/WRITE parameters?",
        {RTL_SIG, RTL_BODY_0, RTL_BODY_1},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.CONFIG,
        "READ=0/WRITE=1 parameters in the signature; START latches mode=rd_wr and SACK branches on mode.",
    ),
    _gq(
        "rtluvm_bus_04",
        "What constraints define the address and data ranges for the write1 and write2 sequences?",
        {WRITE1_SEQ, WRITE2_SEQ, SEQ_ITEM},
        {"tb/i2c_master_seq_lib/i2c_master_write1_seq.svh", "tb/i2c_master_seq_lib/i2c_master_write2_seq.svh", "tb/i2c_master_sequence_item.svh"},
        ExpectedIntent.CONFIG,
        "write1: address inside [0:'h3E], din inside ['h80:'hFF]; write2: address inside ['h3F:'h7F], din inside ['h00:'h7F]; constraint classes are on the sequence item.",
    ),
    _gq(
        "rtluvm_bus_05",
        "How many write transactions does each test drive and at what rate?",
        {WRITE1_TEST, WRITE2_TEST, PARAM_PKG},
        {"tb/i2c_master_test_lib/i2c_master_write1_test.svh", "tb/i2c_master_test_lib/i2c_master_write2_test.svh", "tb/i2c_master_param_pkg.sv"},
        ExpectedIntent.CONFIG,
        "Each test starts its sequence 40 times; PARAM_PKG defines PERIOD=10 for the interface clock.",
    ),
    _gq(
        "rtluvm_bus_06",
        "What parameters and enum constants does the i2c_master RTL expose?",
        {RTL_SIG},
        {"rtl/i2c_master.sv"},
        ExpectedIntent.CONFIG,
        "READ/WRITE parameters and the state enum are declared in the module signature chunk.",
    ),
    # ── UVM / testbench ──────────────────────────────────────────────────
    _gq(
        "rtluvm_uvm_01",
        "How is the UVM environment assembled: which components are built and how are they connected?",
        {ENV_0, ENV_1},
        {"tb/i2c_master_environment.svh"},
        ExpectedIntent.VERIFICATION,
        "build_phase in window 0 creates agent, scoreboard, coverage and sets the config; connect_phase in window 1 wires the analysis ports.",
    ),
    _gq(
        "rtluvm_uvm_02",
        "How is the virtual interface passed into the testbench via uvm_config_db?",
        {TOP_SV, BASE_TEST, ENV_CFG},
        {"tb/i2c_master_top.sv", "tb/i2c_master_test_lib/i2c_master_base_test.svh", "tb/i2c_master_env_config.svh"},
        ExpectedIntent.VERIFICATION,
        "top sets the vif; base_test reads it back and builds the env config that carries mstr_vif.",
    ),
    _gq(
        "rtluvm_uvm_03",
        "How is the agent configured (active/passive mode, virtual interface)?",
        {AGENT_CFG, AGENT_0},
        {"tb/i2c_master_agent/i2c_master_agent_config.svh", "tb/i2c_master_agent/i2c_master_agent.svh"},
        ExpectedIntent.VERIFICATION,
        "agent config holds vif_confg and is_active; the agent applies it to driver/monitor creation.",
    ),
    _gq(
        "rtluvm_uvm_04",
        "How is the testbench top assembled and how is the clock generated?",
        {TOP_SV, PARAM_PKG},
        {"tb/i2c_master_top.sv", "tb/i2c_master_param_pkg.sv"},
        ExpectedIntent.VERIFICATION,
        "top instantiates the interface and DUT, sets the vif, runs uvm_test_top and toggles clk with PERIOD/2.",
    ),
    _gq(
        "rtluvm_uvm_05",
        "How do the monitor and stat_monitor observe the interface and emit items?",
        {MONITOR_1, STAT_MON},
        {"tb/i2c_master_agent/i2c_master_monitor.svh", "tb/i2c_master_agent/i2c_master_stat_monitor.svh"},
        ExpectedIntent.VERIFICATION,
        "monitor window 1 runs ADDR/DATA/STOP match checks and emits item_2/3/4; stat_monitor emits item_1 for state coverage.",
    ),
    _gq(
        "rtluvm_uvm_06",
        "What is the package and include structure of the UVM testbench?",
        {PKG_SV, AGENT_PKG, SEQ_PKG, TEST_PKG},
        {"tb/i2c_master_pkg.sv", "tb/i2c_master_agent/i2c_master_agent_pkg.sv", "tb/i2c_master_seq_lib/i2c_master_seq_pkg.sv", "tb/i2c_master_test_lib/i2c_master_test_pkg.sv"},
        ExpectedIntent.VERIFICATION,
        "The top package includes the agent, sequence and test packages; each package includes its component files.",
    ),
    # ── Log investigation ────────────────────────────────────────────────
    _gq(
        "rtluvm_log_01",
        "What UVM test was run in the write1 log and which tool generated it?",
        {LOG1_0},
        {"sim/log/write1_test_log.log"},
        ExpectedIntent.DEBUG,
        "write1 log chunk 0 is the QuestaSim header: vsim command, +UVM_TESTNAME=i2c_master_write1_test, tool banner.",
    ),
    _gq(
        "rtluvm_log_02",
        "What is the UVM testbench topology reported in the write1 log?",
        {LOG1_2},
        {"sim/log/write1_test_log.log"},
        ExpectedIntent.DEBUG,
        "write1 log chunk 2 prints the UVM component hierarchy (write1_test -> mstr_env -> mstr_agnt -> drv/mon/mon2).",
    ),
    _gq(
        "rtluvm_log_03",
        "Show the address-match packet the monitor captured in the write1 log.",
        {LOG1_24},
        {"sim/log/write1_test_log.log"},
        ExpectedIntent.DEBUG,
        "write1 log chunk 24 is an ADDR_MATCH_ITEM UVM_INFO with the item field dump (address 'h10, rd_wr 'b1).",
    ),
    _gq(
        "rtluvm_log_04",
        "What was the final UVM report summary for the write2 test run?",
        {LOG2_205},
        {"sim/log/write2_test_log.log"},
        ExpectedIntent.DEBUG,
        "write2 log chunk 205 contains the UVM Report Summary with severity and per-id counts.",
    ),
    # ── Terminology mismatch ─────────────────────────────────────────────
    _gq(
        "rtluvm_term_01",
        "What is the 'SDA' and 'SCL' the README talks about — where are they in the design?",
        {README_0, RTL_SIG},
        {"README.md", "rtl/i2c_master.sv"},
        ExpectedIntent.GENERAL,
        "README introduces the serial data/clock lines; the RTL signature declares SDA (inout) and SCL (output).",
    ),
    _gq(
        "rtluvm_term_02",
        "What does 'ACK' / 'NACK' mean in this design and where is it handled?",
        {README_0, RTL_BODY_1},
        {"README.md", "rtl/i2c_master.sv"},
        ExpectedIntent.GENERAL,
        "README defines acknowledgment semantics; the WACK/SACK states sample the ack line.",
    ),
    _gq(
        "rtluvm_term_03",
        "Where is the 'transaction' / sequence item class with the random fields and constraints?",
        {SEQ_ITEM},
        {"tb/i2c_master_sequence_item.svh"},
        ExpectedIntent.GENERAL,
        "i2c_master_sequence_item declares the rand fields (address, din, rd_wr, ...) and their constraint classes.",
    ),
    _gq(
        "rtluvm_term_04",
        "What is the 'COVER_STATES' covergroup and which transition bins does it check?",
        {COV_0, FUN_COV_0},
        {"tb/i2c_master_coverage_collector.svh", "sim/coverage/fun_coverage.txt"},
        ExpectedIntent.GENERAL,
        "COVER_STATES is defined in the coverage collector with IDLE_to_IDLE-style bins; fun_coverage.txt tabulates the hits.",
    ),
    _gq(
        "rtluvm_term_05",
        "What is a 'virtual interface' in this testbench and how is it used by the driver?",
        {IF_SV, TOP_SV, DRIVER_1},
        {"tb/i2c_master_if.sv", "tb/i2c_master_top.sv", "tb/i2c_master_agent/i2c_master_driver.svh"},
        ExpectedIntent.GENERAL,
        "The interface is declared in i2c_master_if, bound in the top, and the driver drives it through its vif handle.",
    ),
]


def get_rtl_uvm_benchmarks() -> list[GoldenQuery]:
    """Return a copy of the curated rtl_uvm benchmark set."""
    return list(BENCHMARKS_RTL_UVM)