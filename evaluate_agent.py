import json
import subprocess
import time
from pathlib import Path


MODEL = "opencode/nemotron-3.5-lightning-free"
PROJECT_ID = "rtl_uvm"

QUESTIONS_FILE = Path("agent_eval_questions.json")
REPORT_FILE = Path("agent_eval_report.json")


def run_question(question):
    prompt = f"""You have access to SemiRAG MCP tools.

project_id={PROJECT_ID}

Question:
{question}

Use SemiRAG MCP tools to retrieve the actual project evidence.

Rules:
- Use semrag_semrag_search first.
- Use semrag_semrag_find_symbol when symbol lookup is appropriate.
- Use semrag_semrag_read_file when exact source context is required.
- Use semrag_semrag_project_info when project-level information is required.
- Do NOT use semrag_semrag_ask.
- Do NOT use built-in read, grep, glob, or bash to inspect the RTL/UVM project.
- Do not answer from your own knowledge.
- When giving file paths, copy them exactly from SemiRAG.
"""

    start = time.time()

    result = subprocess.run(
        [
            "opencode",
            "run",
            "-m",
            MODEL,
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.time() - start

    return {
        "return_code": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())

    report = {
        "model": MODEL,
        "project_id": PROJECT_ID,
        "total_questions": len(questions),
        "results": [],
    }

    for i, item in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(questions)}] {item['id']}")
        print(item["question"])
        print("=" * 70)

        try:
            execution = run_question(item["question"])

            result = {
                **item,
                "execution": execution,
            }

        except subprocess.TimeoutExpired:
            result = {
                **item,
                "execution": {
                    "return_code": None,
                    "elapsed_seconds": 300,
                    "timeout": True,
                    "stdout": "",
                    "stderr": "OpenCode execution timed out",
                },
            }

        report["results"].append(result)

        if execution.get("stdout"):
            print(execution["stdout"])

    REPORT_FILE.write_text(
        json.dumps(report, indent=2)
    )

    print(f"\nReport written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
