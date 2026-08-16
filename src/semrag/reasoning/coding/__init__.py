"""A2 — grounded coding agent.

Deterministic, bounded end-to-end code-change workflow on top of the
frozen SemiRAG core:

- intent classification (information / code change / debug / review /
  ambiguous) never modifies on ambiguous or information-only requests
- retrieve-first: evidence is gathered before any planning
- read-before-write: a patch may only target a file that was read
  during inspection, and new-file writes never clobber an unread
  existing file
- minimal change: only the exact changes the planner declared are
  applied; no reformatting or unrelated edits
- validation honesty: the final response only reports the validation
  that actually ran (target + status + command)
- bounded repair: at most ``coding_max_repair_attempts`` diagnose →
  repair → validate iterations before stopping safely
- safe failure: insufficient evidence, ambiguous requests, patch
  application failures, and parse failures stop before or at the
  modification boundary
- index freshness: after a change, the project is re-indexed and each
  changed file is verified retrievable
- A1 grounding remains the final answer gate on the workflow summary
"""
from semrag.reasoning.coding.models import (
    ChangeOp,
    CodingIntent,
    CodingPhase,
    CodingRequest,
    CodingResult,
    CodingState,
    FileChange,
    PlannedChange,
    ValidationAttempt,
)
from semrag.reasoning.coding.orchestrator import CodingAgent

__all__ = [
    "ChangeOp",
    "CodingAgent",
    "CodingIntent",
    "CodingPhase",
    "CodingRequest",
    "CodingResult",
    "CodingState",
    "FileChange",
    "PlannedChange",
    "ValidationAttempt",
]