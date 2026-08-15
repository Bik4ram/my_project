#!/usr/bin/env python3

import subprocess
import time
import json
from pathlib import Path

PROJECT_ID = "rtl_uvm"
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

QUESTIONS = [

# ======================================================
# Repository Overview (1-10)
# ======================================================

"What is this repository about?",
"Give me an overview of this repository.",
"What are the major directories?",
"What are the important source files?",
"Describe the project architecture.",
"What is the purpose of this project?",
"List the major RTL files.",
"List the major UVM files.",
"What verification environment is implemented?",
"How is this repository organized?",

# ======================================================
# RTL (11-20)
# ======================================================

"What is the top RTL module?",
"Explain the RTL architecture.",
"Which modules instantiate the DUT?",
"What interfaces does the RTL expose?",
"How is reset handled?",
"How is clocking implemented?",
"Explain the state machine.",
"How are registers updated?",
"What signals control transactions?",
"What parameters are configurable?",

# ======================================================
# Testbench (21-30)
# ======================================================

"Describe the UVM testbench.",
"What components are inside the environment?",
"What agents exist?",
"What monitors exist?",
"What drivers exist?",
"What scoreboards exist?",
"What sequencers exist?",
"What virtual interfaces are used?",
"How is configuration passed?",
"What does the top testbench instantiate?",

# ======================================================
# Sequences (31-40)
# ======================================================

"What sequences exist?",
"What tests exist?",
"Explain the write1 sequence.",
"Explain the write2 sequence.",
"What is the base sequence?",
"What transactions are generated?",
"How are sequence items randomized?",
"What constraints exist?",
"How is stimulus generated?",
"What is the sequence hierarchy?",

# ======================================================
# Verification (41-50)
# ======================================================

"How does the scoreboard work?",
"How is coverage collected?",
"What covergroups exist?",
"What assertions exist?",
"How are failures detected?",
"What is checked during write transactions?",
"How are stop conditions verified?",
"How is address checking performed?",
"How is data checking implemented?",
"What functional coverage exists?",

# ======================================================
# Architecture (51-60)
# ======================================================

"Describe the verification architecture.",
"What packages exist?",
"What classes exist?",
"What inheritance hierarchy exists?",
"What factory registrations exist?",
"How are components connected?",
"What TLM connections exist?",
"What analysis ports exist?",
"What analysis exports exist?",
"What FIFOs are used?",

# ======================================================
# Repository Search (61-70)
# ======================================================

"Where is the driver implemented?",
"Where is the monitor implemented?",
"Where is the scoreboard implemented?",
"Where is the interface defined?",
"Where is the DUT instantiated?",
"Where is run_phase implemented?",
"Where is build_phase implemented?",
"Where is connect_phase implemented?",
"Where is main_phase implemented?",
"Where is the environment created?",

# ======================================================
# Code Understanding (71-80)
# ======================================================

"Explain the transaction flow.",
"Explain the verification flow.",
"Explain the packet lifecycle.",
"Trace a write transaction.",
"How does a packet move through the testbench?",
"What happens after reset?",
"What happens after start?",
"What happens after stop?",
"What happens when ACK is received?",
"What happens during data transmission?",

# ======================================================
# Debug (81-90)
# ======================================================

"What files should I inspect if a write transaction fails?",
"What files control address generation?",
"What files control data generation?",
"Where should I debug assertion failures?",
"What logs are generated?",
"What coverage reports exist?",
"What simulation scripts exist?",
"What compile scripts exist?",
"What filelists exist?",
"What documentation exists?",

# ======================================================
# Deep Understanding (91-100)
# ======================================================

"Explain the complete verification flow.",
"Explain the complete transaction lifecycle.",
"Explain how this repository verifies the DUT.",
"What are the strengths of this verification environment?",
"What are the important design patterns?",
"What reusable components exist?",
"What coding style is followed?",
"What are the dependencies between modules?",
"What are the key takeaways from this repository?",
"If I join this project tomorrow, where should I start reading?"
]


results = []

for idx, question in enumerate(QUESTIONS, start=1):
    print("=" * 80)
    print(f"[{idx}/{len(QUESTIONS)}]")
    print(question)

    cmd = [
        "python",
        "run_local_qa.py",
        "ask",
        "--project-id",
        PROJECT_ID,
        "--question",
        question,
    ]

    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        latency = time.time() - start

        results.append({
            "id": idx,
            "question": question,
            "latency_sec": round(latency, 2),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })

        print(f"Completed in {latency:.2f}s")

    except subprocess.TimeoutExpired:

        results.append({
            "id": idx,
            "question": question,
            "timeout": True
        })

        print("Timed out")

    with open(OUTPUT_DIR / f"{idx:03d}.json", "w") as f:
        json.dump(results[-1], f, indent=2)

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone.")
print(f"Questions executed: {len(results)}")
print(f"Results saved in: {OUTPUT_DIR}")
