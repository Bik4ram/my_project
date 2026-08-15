"""A3 multi-file engineering-agent tests.

Extends the A2 coding-agent workflow with the change-surface discipline:
the planner declares which files the change may MODIFY (``change_surface``),
the controller folds those into the inspection set, and the plan must stay
within the identified surface — so a repository-level request like "create a
READ test and integrate it into the active compile path" modifies exactly
the files it must, never every related file.

Uses the same fake toolset / scripted-LLM pattern as test_orchestrator.py.
"""
from __future__ import annotations

import json

from semrag.core.settings import get_settings
from semrag.reasoning.coding.files import ProjectFileError
from semrag.reasoning.coding.models import CodingRequest
from semrag.reasoning.coding.orchestrator import CodingAgent
from semrag.reasoning.llm.interfaces import CompletionResult
from semrag.retrieval.models import RetrievalResult

NEW_TEST = (
    "`ifndef I2C_MASTER_READ_TEST5_SVH\n"
    "`define I2C_MASTER_READ_TEST5_SVH\n"
    "\n"
    "class i2c_master_read_test5 extends i2c_master_base_test;\n"
    "    `uvm_component_utils(i2c_master_read_test5)\n"
    "    function new(string name = \"i2c_master_read_test5\", uvm_component parent);\n"
    "        super.new(name, parent);\n"
    "    endfunction :new\n"
    "    task run_phase(uvm_phase phase);\n"
    "        phase.raise_objection(this);\n"
    "        begin\n"
    "            i2c_master_read_seq seq = i2c_master_read_seq::type_id::create(\"seq\");\n"
    "            seq.start(mstr_env.mstr_agnt.sqr);\n"
    "        end\n"
    "        phase.drop_objection(this);\n"
    "    endtask :run_phase\n"
    "endclass :i2c_master_read_test5\n"
    "\n"
    "`endif\n"
)

PKG_BEFORE = (
    "`ifndef I2C_MASTER_TEST_PKG_SV\n"
    "`define I2C_MASTER_TEST_PKG_SV\n"
    "\n"
    "package i2c_master_test_pkg;\n"
    "    import uvm_pkg::*;\n"
    "    `include \"uvm_macros.svh\"\n"
    "    `include \"i2c_master_base_test.svh\"\n"
    "    `include \"i2c_master_read_test.svh\"\n"
    "endpackage :i2c_master_test_pkg\n"
    "\n"
    "`endif\n"
)

PKG_AFTER = (
    "`ifndef I2C_MASTER_TEST_PKG_SV\n"
    "`define I2C_MASTER_TEST_PKG_SV\n"
    "\n"
    "package i2c_master_test_pkg;\n"
    "    import uvm_pkg::*;\n"
    "    `include \"uvm_macros.svh\"\n"
    "    `include \"i2c_master_base_test.svh\"\n"
    "    `include \"i2c_master_read_test.svh\"\n"
    "    `include \"i2c_master_read_test5.svh\"\n"
    "endpackage :i2c_master_test_pkg\n"
    "\n"
    "`endif\n"
)


def _chunk(path: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c:{path}",
        project_id="rtl_uvm",
        relative_path=path,
        chunk_type="code",
        text="module i2c_master; endmodule\n",
        line_start=1,
        line_end=2,
        fused_score=0.5,
        signal_scores={},
    )


class FakeToolset:
    project_id = "rtl_uvm"

    def __init__(self, files=None, evidence="", validation=None):
        self.files = dict(files or {})
        self.evidence = evidence
        self.validation_queue = list(validation or [])
        self.read_files: set[str] = set()
        self.call_log: list[tuple[str, tuple]] = []
        self.index_called = False

    def _log(self, method, *args):
        self.call_log.append((method, args))

    def list_files(self) -> str:
        return "\n".join(sorted(self.files))

    def search(self, query: str, top_k: int = 10) -> str:
        return self.evidence

    def search_chunks(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return [_chunk(p) for p in self.files]

    def find_symbol(self, symbol: str, limit: int = 20) -> str:
        return f"{symbol}: (symbols in {sorted(self.files)[:2]})"

    def read_file(self, file_path: str, start=None, end=None) -> str:
        self._log("read_file", file_path)
        if file_path not in self.files:
            raise ProjectFileError("not_found", f"File does not exist: {file_path}")
        self.read_files.add(file_path)
        return self.files[file_path]

    def write_file(self, file_path: str, content: str) -> dict:
        self._log("write_file", file_path)
        existed = file_path in self.files
        if existed and file_path not in self.read_files:
            raise AssertionError(f"write without read: {file_path}")
        self.files[file_path] = content
        return {"relative_path": file_path, "created": not existed, "requires_reindex": True}

    def apply_patch(self, file_path: str, old_text: str, new_text: str) -> dict:
        self._log("apply_patch", file_path)
        if file_path not in self.read_files:
            raise AssertionError(f"patch without read: {file_path}")
        current = self.files.get(file_path)
        if current is None:
            raise ProjectFileError("not_found", f"File does not exist: {file_path}")
        if old_text not in current:
            raise ProjectFileError("mismatch", f"old_text not found in {file_path}")
        self.files[file_path] = current.replace(old_text, new_text)
        return {"relative_path": file_path, "replaced_count": 1, "requires_reindex": True}

    def run_validation(self, target: str):
        self._log("run_validation", target)
        status = self.validation_queue.pop(0) if self.validation_queue else "passed"
        from semrag.validation.runner import ValidationResult

        return ValidationResult(
            status=status,
            exit_code=0 if status == "passed" else 1,
            stdout="",
            stderr=f"{status} output",
            command=f"verilator {target}",
            duration_seconds=0.1,
            target=target,
        )

    def validation_targets(self) -> list[str]:
        return ["rtl_lint", "tb_lint", "sv_generated_lint"]

    def index(self) -> dict:
        self._log("index")
        self.index_called = True
        return {"status": "completed"}

    def is_retrievable(self, file_path: str) -> bool:
        return file_path in self.files

    def file_exists(self, file_path: str) -> bool:
        return file_path in self.files

    def snapshot_files(self) -> set[str]:
        return set(self.files)


class ScriptedLLM:
    def __init__(self, plan1=None, plan2=None, diagnose=None):
        self.plan1 = plan1
        self.plan2 = plan2
        self.diagnose = diagnose
        self.calls: list[str] = []

    def complete(self, messages, tools=None) -> CompletionResult:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "PLAN-1" in prompt:
            content = json.dumps(self.plan1)
        elif "PLAN-2" in prompt:
            content = json.dumps(self.plan2)
        elif "DIAGNOSE" in prompt:
            content = json.dumps(self.diagnose)
        else:
            content = "summary"
        return CompletionResult(content=content)


def _make_agent(toolset, llm):
    return CodingAgent(llm, toolset, get_settings(), project_id="rtl_uvm")


def _request(text, max_repair=3, validation_target=None):
    return CodingRequest(
        project_id="rtl_uvm",
        request=text,
        max_repair_attempts=max_repair,
        validation_target=validation_target,
    )


# ---------------------------------------------------------------------------
# A3 change-surface tests
# ---------------------------------------------------------------------------

def test_a3_multi_file_change_within_declared_surface_completes():
    """A create-and-integrate request modifies both the new file and the
    compile-path package, exactly as declared in the change surface."""
    toolset = FakeToolset(
        files={
            "tb/i2c_master_test_lib/i2c_master_base_test.svh": "class base;\nendclass\n",
            "tb/i2c_master_test_lib/i2c_master_read_test.svh": "class read_test;\nendclass\n",
            "tb/i2c_master_test_lib/i2c_master_test_pkg.sv": PKG_BEFORE,
        },
        evidence="READ test pattern [Source: tb/i2c_master_test_lib/i2c_master_read_test.svh]",
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={
            "files_to_read": [
                "tb/i2c_master_test_lib/i2c_master_read_test.svh",
                "tb/i2c_master_test_lib/i2c_master_test_pkg.sv",
            ],
            "change_surface": ["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"],
            "new_file_path": "tb/i2c_master_test_lib/i2c_master_read_test5.svh",
            "validation_target": "tb_lint",
            "pattern_summary": "existing UVM READ test pattern",
        },
        plan2={
            "changes": [
                {
                    "op": "write",
                    "file": "tb/i2c_master_test_lib/i2c_master_read_test5.svh",
                    "new_text": NEW_TEST,
                    "reason": "new READ test",
                },
                {
                    "op": "patch",
                    "file": "tb/i2c_master_test_lib/i2c_master_test_pkg.sv",
                    "old_text": "    `include \"i2c_master_read_test.svh\"\n",
                    "new_text": "    `include \"i2c_master_read_test.svh\"\n    `include \"i2c_master_read_test5.svh\"\n",
                    "reason": "register the new test in the active compile path",
                },
            ],
            "validation_target": "tb_lint",
        },
    )
    result = _make_agent(toolset, llm).run(
        _request("Create a READ test following the existing UVM pattern and integrate it into the active compile path.")
    )

    assert result.final_status == "completed"
    assert "tb/i2c_master_test_lib/i2c_master_read_test5.svh" in result.files_modified
    assert "tb/i2c_master_test_lib/i2c_master_test_pkg.sv" in result.files_modified
    assert "i2c_master_read_test5.svh" in toolset.files["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"]
    assert result.validation_target == "tb_lint"
    assert result.change_surface == ["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"]
    assert "Change surface" in result.final_answer


def test_a3_plan_touching_file_outside_declared_surface_is_rejected():
    """The change surface is a discipline: the planner identified A for
    modification; drifting to B must be rejected with zero changes."""
    toolset = FakeToolset(
        files={
            "rtl/a.sv": "module a;\nendmodule\n",
            "rtl/b.sv": "module b;\nendmodule\n",
        },
        evidence="e",
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={
            "files_to_read": ["rtl/a.sv"],
            "change_surface": ["rtl/a.sv"],
            "new_file_path": "",
            "validation_target": "rtl_lint",
            "pattern_summary": "p",
        },
        plan2={
            "changes": [{"op": "patch", "file": "rtl/b.sv", "old_text": "endmodule", "new_text": "endmodule :b"}],
            "validation_target": "rtl_lint",
        },
    )
    result = _make_agent(toolset, llm).run(_request("Update the module in rtl/a.sv."))

    assert result.final_status == "failed"
    assert "NOT in the declared change surface" in (result.error or "")
    assert result.files_modified == []
    assert toolset.files["rtl/b.sv"] == "module b;\nendmodule\n"


def test_a3_surface_files_are_forced_into_inspection():
    """Every file the change may modify must be read before plan-2, even if
    the planner forgot to list it in files_to_read."""
    toolset = FakeToolset(
        files={
            "tb/pkg.sv": PKG_BEFORE,
            "tb/test.svh": "class t;\nendclass\n",
        },
        evidence="e",
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={
            "files_to_read": ["tb/test.svh"],
            "change_surface": ["tb/pkg.sv"],
            "new_file_path": "",
            "validation_target": "tb_lint",
            "pattern_summary": "p",
        },
        plan2={
            "changes": [
                {"op": "patch", "file": "tb/pkg.sv", "old_text": "    `include \"i2c_master_read_test.svh\"\n", "new_text": "    `include \"i2c_master_read_test.svh\"\n    `include \"i2c_master_read_test5.svh\"\n", "reason": "reg"}
            ],
            "validation_target": "tb_lint",
        },
    )
    result = _make_agent(toolset, llm).run(_request("Add the READ test include to tb/pkg.sv."))

    assert result.final_status == "completed"
    assert "tb/pkg.sv" in toolset.read_files
    order = [m for m, _ in toolset.call_log]
    assert order.index("read_file") < order.index("apply_patch")


def test_a3_compile_path_anchors_surfaced_to_planner():
    """The PLAN-1 prompt must surface the package file so the planner can
    choose it as the integration point (deterministic, no new retrieval)."""
    toolset = FakeToolset(
        files={
            "tb/i2c_master_test_lib/i2c_master_test_pkg.sv": PKG_BEFORE,
            "tb/i2c_master_test_lib/i2c_master_read_test.svh": "class t;\nendclass\n",
            "tb/i2c_master_top.sv": "module i2c_master_top;\nendmodule\n",
            "rtl/i2c_master.sv": "module i2c_master;\nendmodule\n",
        },
        evidence="e",
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={
            "files_to_read": ["tb/i2c_master_test_lib/i2c_master_read_test.svh"],
            "change_surface": [],
            "new_file_path": "",
            "validation_target": "rtl_lint",
            "pattern_summary": "p",
        },
        plan2={"changes": [{"op": "patch", "file": "tb/i2c_master_test_lib/i2c_master_read_test.svh", "old_text": "endclass", "new_text": "endclass :t"}], "validation_target": "rtl_lint"},
    )
    _make_agent(toolset, llm).run(_request("Update the read test."))

    plan1_prompt = llm.calls[0]
    assert "tb/i2c_master_test_lib/i2c_master_test_pkg.sv" in plan1_prompt
    assert "tb/i2c_master_top.sv" in plan1_prompt
    assert "Compile-path anchor" in plan1_prompt


def test_a3_more_than_three_files_allowed_within_surface():
    """Coherent multi-file changes may exceed the old 3-file single-file
    cap when the planner declares a wider change surface."""
    files = {}
    for i in range(4):
        files[f"tb/comp{i}.svh"] = f"class c{i};\nendclass\n"
    toolset = FakeToolset(files=files, evidence="e", validation=["passed"])
    plan2_changes = [
        {"op": "patch", "file": f"tb/comp{i}.svh", "old_text": "endclass", "new_text": f"endclass :c{i}"}
        for i in range(4)
    ]
    llm = ScriptedLLM(
        plan1={
            "files_to_read": [f"tb/comp{i}.svh" for i in range(4)],
            "change_surface": [f"tb/comp{i}.svh" for i in range(4)],
            "new_file_path": "",
            "validation_target": "rtl_lint",
            "pattern_summary": "p",
        },
        plan2={"changes": plan2_changes, "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Update all four component classes to name their endmodule."))

    assert result.final_status == "completed"
    assert len(result.files_modified) == 4
    assert all(toolset.files[f"tb/comp{i}.svh"] == f"class c{i};\nendclass :c{i}\n" for i in range(4))


def test_a3_integration_request_routes_through_changes_contract():
    """A create+integrate request routes through the standard PLAN-2 changes
    contract (single plan emitting BOTH the write and the registration patch),
    so a weak model cannot silently omit the integration step."""
    toolset = FakeToolset(
        files={
            "tb/i2c_master_test_lib/i2c_master_base_test.svh": "class base;\nendclass\n",
            "tb/i2c_master_test_lib/i2c_master_read_test.svh": "class read_test;\nendclass\n",
            "tb/i2c_master_test_lib/i2c_master_test_pkg.sv": PKG_BEFORE,
        },
        evidence="READ test pattern [Source: tb/i2c_master_test_lib/i2c_master_read_test.svh]",
        validation=["passed"],
    )

    class RoutingLLM:
        def __init__(self):
            self.calls: list[str] = []

        def complete(self, messages, tools=None):
            prompt = messages[-1]["content"]
            self.calls.append(prompt)
            if "PLAN-1" in prompt:
                content = json.dumps({
                    "files_to_read": ["tb/i2c_master_test_lib/i2c_master_read_test.svh"],
                    "change_surface": ["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"],
                    "new_file_path": "tb/i2c_master_test_lib/i2c_master_read_test5.svh",
                    "validation_target": "tb_lint",
                    "pattern_summary": "p",
                })
            elif "PLAN-2" in prompt:
                content = json.dumps({
                    "changes": [
                        {"op": "write", "file": "tb/i2c_master_test_lib/i2c_master_read_test5.svh", "new_text": NEW_TEST, "reason": "new READ test"},
                        {
                            "op": "patch",
                            "file": "tb/i2c_master_test_lib/i2c_master_test_pkg.sv",
                            "old_text": "    `include \"i2c_master_read_test.svh\"\n",
                            "new_text": "    `include \"i2c_master_read_test.svh\"\n    `include \"i2c_master_read_test5.svh\"\n",
                            "reason": "register in compile path",
                        },
                    ],
                    "validation_target": "tb_lint",
                })
            else:
                content = "summary"
            return CompletionResult(content=content)

    llm = RoutingLLM()
    result = _make_agent(toolset, llm).run(
        _request("Create a READ test following the existing UVM pattern and integrate it into the active compile path.")
    )

    assert result.final_status == "completed"
    assert "tb/i2c_master_test_lib/i2c_master_read_test5.svh" in result.files_modified
    assert "tb/i2c_master_test_lib/i2c_master_test_pkg.sv" in result.files_modified
    assert "i2c_master_read_test5.svh" in toolset.files["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"]
    plan2_prompt = [p for p in llm.calls if "PLAN-2" in p]
    assert len(plan2_prompt) == 1  # routed through the changes contract, no nudge loop
    assert "Your job (PLAN-2):" in plan2_prompt[0]


def test_a3_surface_is_optional_single_file_change_still_works():
    """No change_surface declared → single-file behaviour is unchanged."""
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e",
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "new_file_path": "", "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule in i2c.sv."))

    assert result.final_status == "completed"
    assert result.files_modified == ["rtl/i2c.sv"]
    assert result.change_surface == []
    assert "(not declared; single-file change)" in result.final_answer