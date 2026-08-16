"""Prompts and JSON contracts for the A2 coding agent's LLM steps.

The orchestrator only asks the LLM to produce structured JSON for three
narrow steps:

- ``plan_targets`` — which existing files to read and which validation
  target to run (the deterministic inspect step reads exactly these).
- ``plan_patch`` — the exact changes to apply. Must reference files
  that were read during inspection (or brand-new files for writes).
- ``diagnose`` — why validation failed and what to change to fix it,
  when a bounded repair is warranted.

Parsing is deliberately lenient (``extract_json``): a malformed plan is
a safe stop, never a guess.
"""
from __future__ import annotations

import json
import re

CODING_SYSTEM_PROMPT = """You are the SemiRAG Grounded Coding Agent. You plan repository code changes, but you never edit files yourself — a deterministic controller applies only the exact changes you declare. Rules:
- Base every decision on the retrieved evidence and the exact file contents provided. Never invent file paths, symbols, or line ranges.
- Plan minimal, targeted changes. Do not reformat, restructure, or touch unrelated code.
- For a READ test request, follow the existing UVM test patterns in the repository (module name, class, UVM test macros, register/build/run phases).
- Validation honesty: you only know what the controller tells you. Never claim a simulation or compile passed unless the controller reports it.
- When evidence is insufficient to plan a safe change, say so explicitly in your JSON."""


def _plan_targets_prompt(
    request: str,
    evidence: str,
    file_list: str,
    symbol_hits: str,
    registered_targets: str,
    compile_anchors: str,
) -> str:
    return f"""{CODING_SYSTEM_PROMPT}

Coding request:
{request}

CRITICAL — the request above is the ONLY task. Plan THAT request, never a different one:
- If it asks to CREATE a READ/WRITE test, the deliverable is a new test file (+ its registration), NOT a change to coverage/scoreboard code.
- Ignore retrieved chunks that talk about unrelated files (coverage, scoreboard, agent internals) unless the request itself references them.
- If the request creates a file ("create", "add", "write a new test/sequence/module"), new_file_path MUST name a non-existing relative path. An empty new_file_path for a create request is wrong.

Retrieved evidence (indexed chunks):
{evidence if evidence else "(no indexed evidence matched)"}

Indexed files:
{file_list if file_list else "(empty project index)"}

Symbol lookups:
{symbol_hits if symbol_hits else "(none)"}

Compile-path anchor files (the "active compile path": package files that `include test/sequence files, and top modules that drive run_test):
{compile_anchors if compile_anchors else "(none found)"}

Your job (PLAN-1): decide what to read and what to modify before any change. Respond with ONLY JSON of the form:
{{"files_to_read": ["path/to/file1", "..."], "change_surface": ["path/to/file1", "..."], "new_file_path": "path/to/newfile.svh", "validation_target": "rtl_lint", "pattern_summary": "1-2 sentences describing the existing pattern to follow"}}

WORKED EXAMPLE for a "create a READ test and integrate it into the compile path" request:
{{"files_to_read": ["tb/i2c_master_test_lib/i2c_master_base_test.svh", "tb/i2c_master_test_lib/i2c_master_read_test.svh", "tb/i2c_master_test_lib/i2c_master_test_pkg.sv"], "change_surface": ["tb/i2c_master_test_lib/i2c_master_test_pkg.sv"], "new_file_path": "tb/i2c_master_test_lib/i2c_master_read_test5.svh", "validation_target": "tb_lint", "pattern_summary": "existing READ test extends i2c_master_base_test and starts i2c_master_read_seq"}}

- files_to_read must name only files that exist in the repository, using EXACT relative paths from the file list (e.g. "tb/i2c_master_test_lib/i2c_master_read_test.svh"). Never invent absolute paths like C:\\... or /home/... — they will be rejected.
- change_surface: the files this change will actually MODIFY. Decide this from evidence, do NOT assume every related file needs editing. Every change_surface entry must also be in files_to_read (or be the new_file_path). For a single-file change, use [].
- For a test/file creation request, read the ACTUAL pattern source files (.sv/.svh) the new file should follow — e.g. existing test classes, sequences, agents, base tests. NEVER read sim logs, coverage reports, or lint reports as the pattern; those are runtime output, not source.
- If the request asks to CREATE a new file (a test, module, sequence, driver...), set new_file_path to a sensible relative path that does NOT yet exist (e.g. "tb/i2c_master_test_lib/i2c_master_read_test3.svh"). If the request does not create a file, use "".
- If the request asks to INTEGRATE a new file into the active compile path (e.g. "integrate it into the compile path"), add the compile-path anchor package that `include-s such files to the change_surface (e.g. "tb/i2c_master_test_lib/i2c_master_test_pkg.sv") so the new file gets registered there.
- validation_target must be one of the registered targets for this project: {registered_targets}. Pick the one that actually exercises the change: rtl-only changes → rtl_lint; generated TB test files → sv_generated_lint; a new test integrated into the full UVM testbench → tb_lint. Default "rtl_lint".
- If the evidence is insufficient to plan a safe change, respond:
{{"files_to_read": [], "change_surface": [], "new_file_path": "", "validation_target": "rtl_lint", "insufficient": "why"}}"""


def _plan_patch_prompt(
    request: str,
    evidence: str,
    inspected_files: str,
    plan_summary: str,
    change_surface: str,
) -> str:
    return f"""{CODING_SYSTEM_PROMPT}

Coding request:
{request}

Retrieved evidence:
{evidence if evidence else "(none)"}

Files already read during inspection (you may ONLY patch these):
{inspected_files if inspected_files else "(none read yet)"}

Declared change surface — the ONLY files this request may modify:
{change_surface if change_surface else "(none declared; a single-file change)"}

Plan summary: {plan_summary}

Your job (PLAN-2): declare the exact minimal changes. Respond with ONLY JSON, either:
A) targeted patches:
{{"changes": [{{"op": "patch", "file": "path", "old_text": "exact text that exists in the file", "new_text": "replacement text", "reason": "short"}}], "validation_target": "rtl_lint"}}
B) a full-file rewrite when the change pervades a small file (bad include guard, wrong macro usage, or the user asserts a defect you cannot fix with a clean small patch):
{{"files": {{"path/to/file": "<complete corrected SystemVerilog content of that file>"}}, "validation_target": "rtl_lint"}}
- Form A requires old_text that appears exactly once in the read file. Form B replaces the WHOLE file, so reproduce everything (guard, class, all methods) — only fix the defect, change nothing else.
- Every change MUST target a file in the declared change surface (or the new_file_path for a write). Do NOT touch files outside it.
- op "patch" requires old_text that appears exactly once in the read file.
- op "write" creates a NEW file (the path from the plan summary); file must not already exist.
- WORKED EXAMPLE for creating a new UVM test file:
{{"changes": [{{"op": "write", "file": "tb/i2c_master_test_lib/i2c_master_read_test3.svh", "new_text": "`ifndef I2C_MASTER_READ_TEST3_SVH\n`define I2C_MASTER_READ_TEST3_SVH\n\nclass i2c_master_read_test3 extends i2c_master_base_test;\n    `uvm_component_utils(i2c_master_read_test3)\n\n    function new(string name = \"i2c_master_read_test3\", uvm_component parent);\n        super.new(name, parent);\n    endfunction :new\n\n    task run_phase(uvm_phase phase);\n        phase.raise_objection(this);\n        begin\n            i2c_master_read_seq seq = i2c_master_read_seq::type_id::create(\"seq\");\n            seq.start(mstr_env.mstr_agnt.sqr);\n        end\n        phase.drop_objection(this);\n    endtask :run_phase\nendclass :i2c_master_read_test3\n\n`endif\n", "reason": "new READ test following the existing pattern"}}], "validation_target": "rtl_lint"}}
- WORKED EXAMPLE for integrating a new test into the active compile path (registering it in the test package):
{{"changes": [{{"op": "write", "file": "tb/i2c_master_test_lib/i2c_master_read_test5.svh", "new_text": "`ifndef I2C_MASTER_READ_TEST5_SVH\n`define I2C_MASTER_READ_TEST5_SVH\n\nclass i2c_master_read_test5 extends i2c_master_base_test;\n    `uvm_component_utils(i2c_master_read_test5)\n    function new(string name = \"i2c_master_read_test5\", uvm_component parent); super.new(name, parent); endfunction\n    task run_phase(uvm_phase phase);\n        phase.raise_objection(this);\n        begin\n            i2c_master_read_seq seq = i2c_master_read_seq::type_id::create(\"seq\");\n            seq.start(mstr_env.mstr_agnt.sqr);\n        end\n        phase.drop_objection(this);\n    endtask :run_phase\nendclass :i2c_master_read_test5\n\n`endif\n", "reason": "new READ test following the existing pattern"}}, {{"op": "patch", "file": "tb/i2c_master_test_lib/i2c_master_test_pkg.sv", "old_text": "    `include \"i2c_master_read_test.svh\"\n", "new_text": "    `include \"i2c_master_read_test.svh\"\n    `include \"i2c_master_read_test5.svh\"\n", "reason": "register the new test in the active compile path"}}], "validation_target": "tb_lint"}}
- When the request creates a new file, your change MUST be an op "write" to the new_file_path from the plan summary (or a tiny patch that registers it). Do NOT invent placeholder old_text that is not in the file.
- Never use empty new_text (deleting working code) unless the request is explicitly to remove something.
- Every patched file MUST appear in the inspected files list above, using the EXACT relative path shown (e.g. "tb/i2c_master_test_lib/i2c_master_read_test.svh"). NEVER invent an absolute path like C:\\... or /home/... — that will be rejected.
- Provide complete SystemVerilog for any new file (module + UVM test, `include uvm_macros.svh, etc.).
- If the change is not safe or evidence is insufficient, respond:
{{"changes": [], "validation_target": "rtl_lint", "insufficient": "why"}}"""


def _plan_patch_retry_prompt(
    request: str,
    evidence: str,
    inspected_files: str,
    plan_summary: str,
    violation: str,
    change_surface: str,
) -> str:
    return f"""{CODING_SYSTEM_PROMPT}

Your previous change plan was REJECTED by the controller:
{violation}

Coding request:
{request}

Retrieved evidence:
{evidence if evidence else "(none)"}

Files already read during inspection (you may ONLY reference these, by EXACT relative path):
{inspected_files if inspected_files else "(none read yet)"}

Declared change surface — the ONLY files this request may modify:
{change_surface if change_surface else "(none declared; a single-file change)"}

Plan summary: {plan_summary}

Your job (PLAN-2 retry): produce a corrected minimal change plan. Respond with ONLY JSON of the form:
{{"changes": [{{"op": "patch", "file": "path", "old_text": "exact text in the file", "new_text": "replacement text", "reason": "short"}}], "validation_target": "rtl_lint"}}
HARD CONSTRAINTS for this retry:
- Stay on the ORIGINAL request. Do not drift into unrelated cleanup, refactoring, or removal of working code.
- Do NOT delete or remove existing code: new_text must never be empty for a patch.
- Every patched file MUST use the EXACT relative path from the inspected list above. Never invent absolute paths.
- Every change MUST target a file in the declared change surface (or the new_file_path). Do NOT touch files outside it.
- For a "create <something>" request, the plan should be ONE write of a new file, or at most a tiny patch to register it.
- If you cannot produce a safe corrected plan, respond:
{{"changes": [], "validation_target": "rtl_lint", "insufficient": "why"}}"""


def _diagnose_prompt(
    request: str,
    plan_summary: str,
    validation: str,
    changes_applied: str,
    current_files: str,
    repair_attempts_left: int,
) -> str:
    return f"""{CODING_SYSTEM_PROMPT}

Coding request:
{request}

Plan summary: {plan_summary}

Validation output (actual, observed):
{validation}

Changes applied so far:
{changes_applied if changes_applied else "(none)"}

Current content of the files being changed:
{current_files if current_files else "(none)"}

Repair attempts remaining: {repair_attempts_left}

Your job (DIAGNOSE): determine why validation failed and what minimal change fixes it. Respond with ONLY JSON, either:
A) a targeted patch:
{{"diagnosis": "root-cause statement", "confidence": "high|medium|low", "changes": [{{"op": "patch", "file": "path", "old_text": "exact text currently in the file", "new_text": "replacement text", "reason": "..."}}]}}
B) a full-file rewrite when the failure pervades the file (broken include guard, wrong macro usage, bad structure):
{{"diagnosis": "root-cause statement", "confidence": "high|medium|low", "files": {{"path/to/file": "<complete corrected SystemVerilog content of that file>"}}}}
- Use form B when the file is small and a wholesale rewrite is cleaner; the controller replaces the whole file content atomically.
- Every preprocessor directive (`ifndef, `define, `endif, `uvm_component_utils, `include) MUST start with a literal backtick.
- changes may be empty / files may be empty if you cannot determine a repair.
- Never invent file contents: only reference files from the current content above."""


def _plan_create_prompt(
    request: str,
    evidence: str,
    inspected_files: str,
    target_path: str,
    plan_summary: str,
    change_surface: str,
) -> str:
    return f"""{CODING_SYSTEM_PROMPT}

Coding request:
{request}

Retrieved evidence:
{evidence if evidence else "(none)"}

Pattern files already read during inspection:
{inspected_files if inspected_files else "(none read yet)"}

Declared change surface — the ONLY files this request may modify:
{change_surface if change_surface else f"(none declared; the new file {target_path})"}

Plan summary: {plan_summary}

The controller will create the new file at: {target_path}
The path is fixed by the controller — do NOT propose a different path.

Your job (PLAN-2 create): write the COMPLETE content for the new file at {target_path}. Respond with ONLY JSON of the form:
{{"new_text": "<complete SystemVerilog file content, JSON-escaped>", "validation_target": "rtl_lint"}}

WORKED EXAMPLE (the new_text VALUE is shown; note it starts directly with the `ifndef line — no extra quote characters):
{{"new_text": "`ifndef I2C_MASTER_READ_TEST3_SVH\n`define I2C_MASTER_READ_TEST3_SVH\n\nclass i2c_master_read_test3 extends i2c_master_base_test;\n    `uvm_component_utils(i2c_master_read_test3)\n\n    function new(string name = \"i2c_master_read_test3\", uvm_component parent);\n        super.new(name, parent);\n    endfunction :new\n\n    task run_phase(uvm_phase phase);\n        phase.raise_objection(this);\n        begin\n            i2c_master_read_seq seq = i2c_master_read_seq::type_id::create(\"seq\");\n            seq.start(mstr_env.mstr_agnt.sqr);\n        end\n        phase.drop_objection(this);\n    endtask :run_phase\nendclass :i2c_master_read_test3\n\n`endif\n", "validation_target": "rtl_lint"}}

Requirements:
- The new_text VALUE must begin directly with `ifndef and end with `endif — do NOT wrap it in double-quote characters of your own; the controller handles JSON quoting. Every preprocessor directive (`ifndef, `define, `endif, `include) MUST start with a literal backtick.
- The file is a new UVM test. Model it EXACTLY on the pattern files above: use the base test class, the existing sequence, and the member/agent names that actually appear in those patterns (e.g. mstr_env.mstr_agnt.sqr, i2c_master_read_seq).
- Include exactly ONE `ifndef guard named after the target file basename (e.g. I2C_MASTER_READ_TEST3_SVH for i2c_master_read_test3.svh), ONE `define, and ONE `endif at the end.
- Use only SystemVerilog: class declaration ends with `;`, NOT `{{`. No markdown fence inside the JSON string.
- Do NOT register the file anywhere; the controller only creates the file.
- If the evidence is insufficient to write a correct new file, respond:
{{"new_text": "", "validation_target": "rtl_lint", "insufficient": "why"}}"""


def _strip_wrapping_quotes(text: str) -> str:
    """Some small models wrap the generated file in literal double quotes
    (a JSON-escaping artifact). Remove ONE balanced leading/trailing quote
    pair when the rest of the content still looks like code."""
    stripped = text
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        inner = stripped[1:-1]
        if "\n" in inner or "`if" in inner or "module " in inner or "class " in inner:
            return inner
    return stripped


def _unwrap_jsonish_new_text(text: str) -> str | None:
    """Some small models emit ``{{"new_text": "...", ...}}`` where the
    inner quotes are not JSON-escaped, so ``json.loads`` fails. If the
    text still has the recognisable wrapper shape, peel it off without
    full JSON parsing."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}") and '"new_text"' in stripped):
        return None
    body = stripped
    m = re.search(r'"validation_target"\s*:\s*"[^"]*"\s*\}\s*$', body)
    if m:
        body = body[: m.start()].rstrip().rstrip(",").rstrip()
    if '"new_text"' not in body:
        return None
    head = body.index('"new_text"') + len('"new_text"')
    rest = body[head:].strip()
    if not rest.startswith(":"):
        return None
    rest = rest[1:].strip()
    if not (rest.startswith('"') and rest.endswith('"')):
        return None
    inner = _unescape_json_body(rest[1:-1])
    return _strip_wrapping_quotes(inner) or None


def _unescape_json_body(s: str) -> str:
    """Decode the JSON-style escapes the model may have emitted inside an
    unparseable ``new_text`` wrapper. ``\\`` is decoded last so escaped
    backslashes are not re-interpreted as escapes."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def extract_sv_content(text: str) -> str | None:
    """Extract SystemVerilog content from a raw LLM response. Accepts:
    - a JSON object with ``new_text`` (or a ``changes`` list whose first
      write carries ``new_text``), or
    - an unparseable ``{"new_text": ...}`` wrapper (inner quotes not
      escaped), or
    - plain text, optionally wrapped in a markdown code fence.
    Returns the trimmed content or None."""
    if not text:
        return None
    parsed = extract_json(text)
    if parsed is not None:
        new_text = parsed.get("new_text")
        if isinstance(new_text, str) and new_text.strip():
            return _strip_wrapping_quotes(new_text.strip())
        changes = parsed.get("changes")
        if isinstance(changes, list) and changes:
            first = changes[0]
            if isinstance(first, dict) and isinstance(first.get("new_text"), str):
                return _strip_wrapping_quotes(first["new_text"].strip())
        return None
    unwrapped = _unwrap_jsonish_new_text(text)
    if unwrapped is not None:
        return unwrapped
    stripped = text.strip()
    fenced = re.match(r"^\s*```[A-Za-z0-9_\-]*\s*\n(.*?)\n?\s*```\s*$", stripped, re.DOTALL)
    if fenced:
        return _strip_wrapping_quotes(fenced.group(1).strip())
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_\-]*\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return _strip_wrapping_quotes(stripped.strip()) or None


def extract_json(text: str) -> dict | None:
    """Leniently parse the first balanced JSON object in ``text``."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None