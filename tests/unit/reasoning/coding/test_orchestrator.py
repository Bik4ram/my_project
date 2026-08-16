"""A2 coding agent workflow tests — the 20 required items.

Uses a fake toolset (in-memory files, scripted evidence/validation) and a
scripted LLM (JSON contracts for PLAN-1 / PLAN-2 / DIAGNOSE), so the
deterministic state machine is exercised without PostgreSQL or Ollama.
"""
from __future__ import annotations

import json

from semrag.core.settings import get_settings
from semrag.reasoning.coding.files import ProjectFileError
from semrag.reasoning.coding.models import CodingIntent, CodingRequest
from semrag.reasoning.coding.orchestrator import CodingAgent
from semrag.reasoning.llm.interfaces import CompletionResult
from semrag.retrieval.models import RetrievalResult


def _chunk(path: str, text: str = "module i2c_master; endmodule\n") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c:{path}",
        project_id="rtl_uvm",
        relative_path=path,
        chunk_type="code",
        text=text,
        line_start=1,
        line_end=2,
        fused_score=0.5,
        signal_scores={},
    )


class FakeToolset:
    project_id = "rtl_uvm"

    def __init__(self, files=None, evidence="", chunks=None, symbol_hits="", validation=None):
        self.files = dict(files or {})
        self.evidence = evidence
        self.chunks = chunks or []
        self.symbol_hits = symbol_hits
        self.validation_queue = list(validation or [])
        self.read_files: set[str] = set()
        self.call_log: list[tuple[str, tuple]] = []
        self.index_called = False

    def _log(self, method, *args):
        self.call_log.append((method, args))

    def project_info(self) -> str:
        self._log("project_info")
        return "project_id=rtl_uvm root=/tmp/rtl_uvm"

    def list_files(self) -> str:
        self._log("list_files")
        return "\n".join(sorted(self.files))

    def search(self, query: str, top_k: int = 10) -> str:
        self._log("search", query)
        return self.evidence

    def search_chunks(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._log("search_chunks", query)
        return self.chunks

    def find_symbol(self, symbol: str, limit: int = 20) -> str:
        self._log("find_symbol", symbol)
        return self.symbol_hits

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
        self._log("is_retrievable", file_path)
        return file_path in self.files

    def file_exists(self, file_path: str) -> bool:
        return file_path in self.files

    def snapshot_files(self) -> set[str]:
        return set(self.files)


class ScriptedLLM:
    def __init__(self, plan1=None, plan2=None, diagnose=None, info=None):
        self.plan1 = plan1
        self.plan2 = plan2
        self.diagnose = diagnose
        self.info = info
        self.calls: list[str] = []
        self._diag_index = 0

    def complete(self, messages, tools=None) -> CompletionResult:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "PLAN-1" in prompt:
            content = json.dumps(self.plan1)
        elif "PLAN-2" in prompt:
            content = json.dumps(self.plan2)
        elif "DIAGNOSE" in prompt:
            diag = self.diagnose if not isinstance(self.diagnose, list) else self.diagnose[min(self._diag_index, len(self.diagnose) - 1)]
            self._diag_index += 1
            content = json.dumps(diag)
        else:
            content = self.info
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
# Required test items
# ---------------------------------------------------------------------------

def test_1_information_request_does_not_modify_files():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "module i2c_master_test;\nendmodule\n"},
        evidence="The I2C READ transaction flows SCL SDA [Source: tb/i2c_master_test.sv]",
        chunks=[_chunk("tb/i2c_master_test.sv")],
    )
    llm = ScriptedLLM(info="The I2C READ transaction drives SCL and SDA via vif [Source: tb/i2c_master_test.sv].")
    result = _make_agent(toolset, llm).run(_request("Trace the I2C READ transaction."))

    assert result.intent is CodingIntent.INFORMATION
    assert result.final_status == "completed"
    assert result.changes == []
    assert result.files_modified == []
    calls = [c for c, _ in toolset.call_log]
    assert "write_file" not in calls and "apply_patch" not in calls


def test_2_coding_request_retrieves_before_modifying():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "class i2c_master_test extends uvm_test;\nendclass\n"},
        evidence="existing UVM test pattern [Source: tb/i2c_master_test.sv]",
        chunks=[_chunk("tb/i2c_master_test.sv")],
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.sv"], "validation_target": "rtl_lint", "pattern_summary": "follow existing"},
        plan2={
            "changes": [{
                "op": "patch",
                "file": "tb/i2c_master_test.sv",
                "old_text": "class i2c_master_test extends uvm_test;\nendclass\n",
                "new_text": "class i2c_master_test extends uvm_test;\n  function new(...); endfunction\nendclass\n",
                "reason": "add constructor",
            }],
            "validation_target": "rtl_lint",
        },
    )
    result = _make_agent(toolset, llm).run(_request("Add a constructor to the existing test."))

    assert result.final_status == "completed"
    order = [c for c, _ in toolset.call_log]
    assert "search" in order or "search_chunks" in order
    first_write = next(i for i, m in enumerate(order) if m in ("write_file", "apply_patch"))
    assert order.index("search") < first_write


def test_3_existing_file_read_before_patch():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "class i2c_master_test;\nendclass\n"},
        evidence="evidence",
        chunks=[_chunk("tb/i2c_master_test.sv")],
        validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "tb/i2c_master_test.sv",
                           "old_text": "endclass", "new_text": "endclass : i2c_master_test"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Update the test class name."))

    assert result.final_status == "completed"
    order = [c for c, _ in toolset.call_log]
    assert order.index("read_file") < order.index("apply_patch")


def test_3b_unplanned_existing_file_is_read_before_patch():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "x\n", "tb/other.sv": "y\n"},
        evidence="e", chunks=[_chunk("tb/i2c_master_test.sv")], validation=["passed"],
    )
    # Plan-2 references an existing file the planner never read. The
    # controller deterministically reads it BEFORE the write (so the
    # read-before-write guarantee is controller-enforced), then applies
    # the corrective plan.
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "tb/other.sv", "old_text": "y", "new_text": "z"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Update the file tb/other.sv."))

    assert result.final_status == "completed"
    assert "tb/other.sv" in toolset.read_files
    order = [c for c, _ in toolset.call_log]
    assert order.index("read_file") < order.index("apply_patch")
    assert toolset.files["tb/other.sv"] == "z\n"


def test_3c_patch_mismatch_after_read_is_safe_stop():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "x\n", "tb/other.sv": "REAL_CONTENT\n"},
        evidence="e", chunks=[_chunk("tb/i2c_master_test.sv")],
    )
    # Even after the corrective read, the planner's old_text does not match
    # the actual file → the preflight aborts the WHOLE plan before any
    # write, so the repository stays byte-for-byte untouched.
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "tb/other.sv", "old_text": "y", "new_text": "z"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Update the file tb/other.sv."))

    assert result.final_status == "failed"
    assert "old_text" in (result.error or "")
    assert result.changes == []
    assert toolset.files["tb/other.sv"] == "REAL_CONTENT\n"
    patches = [args for m, args in toolset.call_log if m == "apply_patch"]
    assert not patches


def test_4_minimal_patch_applied():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule in i2c.sv."))

    assert result.final_status == "completed"
    patches = [args for m, args in toolset.call_log if m == "apply_patch"]
    writes = [args for m, args in toolset.call_log if m == "write_file"]
    assert len(patches) == 1 and len(writes) == 0
    assert toolset.files["rtl/i2c.sv"] == "module i2c;\nendmodule : i2c\n"


def test_5_successful_validation_produces_success_state():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.final_status == "completed"
    assert result.validation_attempts and result.validation_attempts[-1].status == "passed"
    assert result.repair_attempts == 0


def test_6_validation_failure_enters_diagnosis():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")],
        validation=["failed", "passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "missing semicolon", "confidence": "high",
                  "changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule : i2c", "new_text": "endmodule : i2c;\n"}], "reason": "add semicolon"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.validation_attempts[0].status == "failed"
    assert any("DIAGNOSE" in c for c in llm.calls)


def test_7_repair_attempted_after_failure():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["failed", "passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "syntax", "confidence": "high",
                  "changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule : i2c", "new_text": "endmodule : i2c;"}]},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.repair_attempts == 1
    assert len(toolset.call_log) >= 2
    patches = [m for m, _ in toolset.call_log if m == "apply_patch"]
    assert len(patches) >= 2  # initial change + repair


def test_8_successful_repair_returns_to_validation():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["failed", "failed", "passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "d", "confidence": "medium",
                  "changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule : i2c", "new_text": "endmodule : i2c;"}]},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.final_status == "completed"
    assert result.repair_attempts == 2
    assert result.validation_attempts[-1].status == "passed"


def test_9_max_repair_attempts_enforced():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")],
        validation=["failed", "failed", "failed", "failed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose=[{"diagnosis": "d", "confidence": "low",
                   "changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule : i2c", "new_text": "endmodule : i2c"}]}],
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.repair_attempts == 3  # bounded
    assert len(result.validation_attempts) == 4
    assert result.final_status == "failed"


def test_10_failed_repair_terminates_safely():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")],
        validation=["failed", "failed", "failed", "failed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "d", "confidence": "low",
                  "changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule : i2c", "new_text": "endmodule : i2c"}]},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.final_status == "failed"
    assert "repair" in (result.error or "").lower()
    assert isinstance(result.final_answer, str) and result.final_answer


def test_11_final_response_reflects_actual_validation_status():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["failed", "failed", "failed", "failed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "d", "confidence": "low", "changes": []},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert "status: failed" in result.final_answer
    assert result.final_status == "failed"


def test_12_no_false_simulation_claim_when_only_lint_ran():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert "rtl_lint" in result.final_answer
    assert "passed" in result.final_answer
    assert "not a full simulation result" in result.final_answer
    assert "simulation" not in result.final_answer.split("not a full simulation result")[0]


def test_13_index_required_modification_handled():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.sv": "class base_test;\nendclass\n"},
        evidence="pattern [Source: tb/i2c_master_test.sv]",
        chunks=[_chunk("tb/i2c_master_test.sv")], validation=["passed"],
    )
    new_content = "class i2c_master_read_test extends base_test;\nendclass\n"
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"new_text": new_content, "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Create a new read test."))

    assert result.final_status == "completed"
    assert toolset.index_called is True
    assert any(m == "is_retrievable" for m, _ in toolset.call_log)
    assert "re-indexed: yes" in result.final_answer
    assert "tb/i2c_master_read_test.sv" in result.final_answer


def test_14_changed_file_verified_after_patch():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.changes and result.changes[0].applied is True
    assert result.changes[0].verified is True


def test_15_unrelated_files_not_modified():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n", "rtl/other.sv": "module other;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule in i2c.sv."))

    assert result.final_status == "completed"
    assert "rtl/other.sv" not in result.files_modified
    assert "rtl/other.sv" not in toolset.read_files


def test_16_ambiguous_request_does_not_modify():
    toolset = FakeToolset(files={"rtl/i2c.sv": "x\n"}, evidence="e", chunks=[_chunk("rtl/i2c.sv")])
    llm = ScriptedLLM()
    result = _make_agent(toolset, llm).run(_request("Hello there."))

    assert result.final_status == "needs_clarification"
    assert result.needs_clarification is not None
    assert result.files_modified == []
    calls = [c for c, _ in toolset.call_log]
    assert "write_file" not in calls and "apply_patch" not in calls


def test_17_insufficient_evidence_safe_stop():
    toolset = FakeToolset(files={}, evidence="", chunks=[], validation=[])
    llm = ScriptedLLM()
    result = _make_agent(toolset, llm).run(_request("Modify the i2c_master_xyz_driver to add a timeout."))

    assert result.final_status == "insufficient_evidence"
    assert result.files_modified == []
    calls = [c for c, _ in toolset.call_log]
    assert "write_file" not in calls and "apply_patch" not in calls


def test_18_a1_grounding_active_on_final_response():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule."))

    assert result.grounding is not None
    assert result.grounding.get("status") in ("grounded", "revised", "insufficient_evidence", "unsupported")


def test_19_20_regression_placeholder():
    """Items 19 and 20 (R4 and A1 suites stay green) are enforced by the
    full-suite run; kept here as explicit documentation of the mapping."""
    assert True


def test_21_unrelated_destructive_deletion_is_rejected():
    toolset = FakeToolset(
        files={"tb/i2c_master_test.svh": "class base_test;\n  bit scb_exist_tst, cov_exist_tst;\nendclass\n"},
        evidence="e", chunks=[_chunk("tb/i2c_master_test.svh")], validation=["passed"],
    )
    # The planner drifts: instead of creating the requested test, it deletes
    # unrelated lines. The deletion guard must stop it with zero changes.
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/i2c_master_test.svh"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [
            {"op": "patch", "file": "tb/i2c_master_test.svh", "old_text": "  bit scb_exist_tst, cov_exist_tst;\n", "new_text": "", "reason": "cleanup"},
        ], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Modify tb/i2c_master_test.svh to add a new field."))

    assert result.final_status == "failed"
    assert "deletion" in (result.error or "")
    assert result.files_modified == []
    assert "  bit scb_exist_tst, cov_exist_tst;\n" in toolset.files["tb/i2c_master_test.svh"]

def test_create_content_accepts_markdown_fenced_sv():
    from semrag.reasoning.coding.prompts import extract_sv_content

    fenced = '```systemverilog\n`ifndef X_SVH\n`define X_SVH\n\nclass x;\nendclass\n\n`endif\n```'
    assert extract_sv_content(fenced) == "`ifndef X_SVH\n`define X_SVH\n\nclass x;\nendclass\n\n`endif"
    raw = '`ifndef X_SVH\n`define X_SVH\nendclass\n`endif\n'
    assert extract_sv_content(raw) == "`ifndef X_SVH\n`define X_SVH\nendclass\n`endif"
    assert extract_sv_content('{"new_text": "module m;\\nendmodule"}') == "module m;\nendmodule"
    assert extract_sv_content("") is None


def test_create_content_unwraps_unescaped_jsonish_wrapper():
    from semrag.reasoning.coding.prompts import extract_sv_content

    raw = '{"new_text": "`ifndef X_SVH\n`define X_SVH\n\nclass x;\n    function new(string name = "x", uvm_component parent);\n    endfunction\nendclass\n\n`endif", "validation_target": "rtl_lint"}'
    out = extract_sv_content(raw)
    assert out is not None
    assert out.startswith("`ifndef X_SVH")
    assert out.endswith("`endif")
    assert 'name = "x"' in out


def test_pinned_validation_target_wins_over_planner_pick():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "endmodule", "new_text": "endmodule : i2c"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Name the endmodule in i2c.sv.", validation_target="tb_lint"))

    assert result.final_status == "completed"
    assert result.validation_target == "tb_lint"
    assert any(a.target == "tb_lint" for a in result.validation_attempts)


def test_explicitly_named_file_is_forced_into_inspection():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\n  int bad syntax\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    # The planner names the wrong file; the controller must still read the
    # file the request explicitly pointed at (rtl/i2c.sv).
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/other.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/i2c.sv", "old_text": "int bad syntax", "new_text": "int good_syntax", "reason": "fix"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Fix the syntax error in rtl/i2c.sv."))

    assert result.final_status == "completed"
    assert "rtl/i2c.sv" in toolset.read_files
    assert "rtl/i2c.sv" in result.files_modified


def test_plan_must_touch_explicitly_named_file():
    toolset = FakeToolset(
        files={"rtl/i2c.sv": "module i2c;\nendmodule\n", "rtl/other.sv": "module other;\nendmodule\n"},
        evidence="e", chunks=[_chunk("rtl/i2c.sv")], validation=["passed"],
    )
    # The planner drifts to a different file; the explicit-file guard stops it.
    llm = ScriptedLLM(
        plan1={"files_to_read": ["rtl/i2c.sv"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "rtl/other.sv", "old_text": "module other;", "new_text": "module other2;", "reason": "drift"}], "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Fix the syntax error in rtl/i2c.sv."))

    assert result.final_status == "failed"
    assert result.files_modified == []
    assert "rtl/i2c.sv" in (result.error or "")


def test_repair_full_file_rewrite_mode():
    good = "`ifndef X_SVH\n`define X_SVH\n\nclass x;\n    `uvm_component_utils(x)\nendclass :x\n\n`endif\n"
    toolset = FakeToolset(
        files={"tb/x.svh": "`ifndef X_SVH\n`define X_SVH\n\nclass x;\n    uvm_component_utils(x)\nendclass :x\n\n`endif\n"},
        evidence="e", chunks=[_chunk("tb/x.svh")], validation=["failed", "passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/x.svh"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"changes": [{"op": "patch", "file": "tb/x.svh", "old_text": "uvm_component_utils(x)", "new_text": "uvm_component_utils(x);", "reason": "fix"}], "validation_target": "rtl_lint"},
        diagnose={"diagnosis": "macro used without backtick", "confidence": "high",
                  "files": {"tb/x.svh": good}},
    )
    result = _make_agent(toolset, llm).run(_request("Fix the syntax error in tb/x.svh.", max_repair=2))

    assert result.final_status == "completed"
    assert result.repair_attempts == 1
    assert toolset.files["tb/x.svh"] == good
    assert "`uvm_component_utils(x)" in toolset.files["tb/x.svh"]


def test_plan2_full_file_rewrite_form():
    good = "`ifndef X_SVH\n`define X_SVH\n\nclass x;\n    `uvm_component_utils(x)\nendclass :x\n\n`endif\n"
    toolset = FakeToolset(
        files={"tb/x.svh": "`ifndef X_SVH\n`define X_SVH\n\nclass x;\n    uvm_component_utils(x)\nendclass :x\n\n`endif\n"},
        evidence="e", chunks=[_chunk("tb/x.svh")], validation=["passed"],
    )
    llm = ScriptedLLM(
        plan1={"files_to_read": ["tb/x.svh"], "validation_target": "rtl_lint", "pattern_summary": "p"},
        plan2={"files": {"tb/x.svh": good}, "validation_target": "rtl_lint"},
    )
    result = _make_agent(toolset, llm).run(_request("Fix the syntax error in tb/x.svh."))

    assert result.final_status == "completed"
    assert toolset.files["tb/x.svh"] == good
    assert "`uvm_component_utils(x)" in toolset.files["tb/x.svh"]
