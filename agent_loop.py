"""Bounded coding-agent feedback loop for SemiRAG.

Drives a bounded retrieve → generate/modify → validate → diagnose →
patch → revalidate loop against an LLM agent that uses the SemiRAG MCP
tools. The loop is bounded (MAX_ITERATIONS, default 3) and never runs in
the background.

Each iteration:
  1. Instruct the agent (via opencode) to work the task using the MCP
     write/patch tools.
  2. Run the project validator (Verilator lint) through the FastAPI
     endpoint — this is the *ground truth* gate, not the model's opinion.
  3. Classify: passed → stop. blocked → stop (environment/toolchain
     problem, not patchable here). failed → feed the validation output
     back and ask the agent to retrieve the relevant source, diagnose,
     and patch.

The model is told to retrieve source before modifying — the fix prompt
requires reading the relevant file(s) via semrag_read_file and only then
patching with semrag_apply_patch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

SEMRAG_URL = "http://127.0.0.1:8000"
SEMRAG_API_KEY = "dev-local-key"
DEFAULT_MODEL = "opencode/nemotron-3.5-lightning-free"
DEFAULT_PROJECT = "rtl_uvm"
DEFAULT_TARGET = "tb_lint"
MAX_ITERATIONS = 3
OPENCODE_TIMEOUT = 600

_SYSTEM_RULES = """You are a Verilog/SystemVerilog and UVM engineer working through SemiRAG.

Rules:
- Use semrag_semrag_search / semrag_semrag_find_symbol / semrag_semrag_read_file
  to retrieve the ACTUAL project source before changing anything. Never patch
  based only on an error message.
- Create or modify files ONLY through semrag_semrag_write_file or
  semrag_semrag_apply_patch. Never use a generic filesystem tool.
- Do NOT use semrag_semrag_ask.
- Do NOT delete files, install packages, modify system config, or run arbitrary
  shell commands.
- Validation is run for you after your turn; report what you changed and why.
"""


def _headers() -> dict:
    return {"Authorization": f"Bearer {SEMRAG_API_KEY}", "Content-Type": "application/json"}


def run_validation(project_id: str, target: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{SEMRAG_URL}/v1/projects/{project_id}/validate",
            headers=_headers(),
            json={"target": target},
        )
    if resp.status_code != 200:
        return {"status": "error", "http_status": resp.status_code, "body": resp.text}
    return resp.json()


def run_agent(prompt: str) -> tuple[int, str, bool]:
    """Run opencode with a prompt; returns (returncode, stdout, timed_out)."""
    try:
        result = subprocess.run(
            ["opencode", "run", "-m", DEFAULT_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=OPENCODE_TIMEOUT,
        )
        return result.returncode, result.stdout, False
    except subprocess.TimeoutExpired:
        return -1, "", True


def _validation_summary(result: dict) -> str:
    status = result.get("status")
    if status == "error":
        return f"Validation call failed (HTTP {result.get('http_status')}): {result.get('body')}"
    lines = [f"status: {status}"]
    if result.get("exit_code") is not None:
        lines.append(f"exit_code: {result['exit_code']}")
    if result.get("blocked_reason"):
        lines.append(f"blocked_reason: {result['blocked_reason']}")
    if result.get("stderr"):
        lines.append(f"stderr:\n{result['stderr']}")
    if result.get("stdout"):
        lines.append(f"stdout:\n{result['stdout']}")
    if result.get("duration_seconds") is not None:
        lines.append(f"duration_seconds: {result['duration_seconds']}")
    return "\n".join(lines)


def build_task_prompt(task: str, project_id: str, target: str) -> str:
    return f"""{_SYSTEM_RULES}

project_id={project_id}
validation target to use: {target}

TASK:
{task}

Work the task now. After you finish, stop and report. Validation will be
run by the harness immediately afterwards.
"""


def build_fix_prompt(
    project_id: str,
    target: str,
    task: str,
    validation_result: dict,
    iteration: int,
) -> str:
    summary = _validation_summary(validation_result)
    return f"""{_SYSTEM_RULES}

project_id={project_id}
validation target: {target}
Iteration {iteration} of the loop.

TASK (unchanged):
{task}

The project validator just ran and it FAILED. Validation output:
----------------------------------------
{summary}
----------------------------------------

Diagnose the failure:
1. semrag_semrag_search / semrag_semrag_read_file the relevant source
   files (follow the file paths in the validation output).
2. Find the root cause in the code — never guess from the message alone.
3. Fix it with semrag_semrag_apply_patch (single exact replacement) or
   semrag_semrag_write_file if a new file is genuinely needed.
4. If the validation output says the failure is BLOCKED (missing
   dependency/toolchain, e.g. UVM), do NOT try to patch code — report the
   exact missing dependency and stop.

Do not touch unrelated code. Do not use semrag_semrag_ask. Stop and
report when done; the harness re-runs validation afterwards.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded SemiRAG coding-agent loop")
    parser.add_argument("--task", default="Create the I2C READ testcase.", help="Task for the agent")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Validation target name")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    if args.max_iterations < 1:
        print("max-iterations must be >= 1", file=sys.stderr)
        return 2

    # Verify the validation endpoint works up front (fail fast).
    probe = run_validation(args.project, args.target)
    if probe.get("status") == "error":
        print(f"Pre-flight validation failed: {probe.get('body')}", file=sys.stderr)
        return 2

    history = []
    final = None
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'=' * 70}\nITERATION {iteration}/{args.max_iterations}\n{'=' * 70}")

        if iteration == 1:
            prompt = build_task_prompt(args.task, args.project, args.target)
        else:
            prompt = build_fix_prompt(
                args.project, args.target, args.task, final, iteration
            )

        rc, stdout, timed_out = run_agent(prompt)
        if timed_out:
            print(f"[opencode] TIMED OUT after {OPENCODE_TIMEOUT}s")
        else:
            print(f"[opencode] return code: {rc}")

        print("\n--- validation ---")
        result = run_validation(args.project, args.target)
        print(_validation_summary(result))

        history.append(
            {
                "iteration": iteration,
                "agent_return_code": rc,
                "agent_timed_out": timed_out,
                "validation": result,
            }
        )

        status = result.get("status")
        final = result
        if status == "passed":
            print(f"\nVALIDATION PASSED at iteration {iteration}. Stopping loop.")
            break
        if status == "blocked":
            print("\nVALIDATION BLOCKED (environment/toolchain). Stopping loop — no code patch can fix this.")
            break
        if status == "failed":
            if iteration == args.max_iterations:
                print("\nMax iterations reached with validation still failing. Loop exhausted.")
            else:
                print(f"\nValidation failed. Iteration {iteration + 1} will diagnose + patch.")
        else:
            print(f"\nUnexpected validation status: {status!r}")

    report = {
        "model": DEFAULT_MODEL,
        "project_id": args.project,
        "target": args.target,
        "task": args.task,
        "max_iterations": args.max_iterations,
        "iterations_used": len(history),
        "final_status": final.get("status") if final else None,
        "history": history,
    }
    Path("agent_loop_report.json").write_text(json.dumps(report, indent=2))
    print("\nReport written to agent_loop_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())