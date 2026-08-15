"""The A2 coding agent state machine.

``CodingAgent`` is the deterministic orchestrator. The LLM only fills
three narrow JSON contracts (plan targets, plan patch, diagnose); every
safety-relevant step — retrieval, read-before-write, applying changes,
validation, bounded repair, verification, index freshness, grounding —
is deterministic controller code.

State machine (CodingPhase):
  INTAKE → classify_intent
    INFORMATION/REVIEW → RETRIEVE → FINAL (grounded answer, no modify)
    AMBIGUOUS           → STOPPED (needs_clarification, no modify)
    CODE_CHANGE/DEBUG   → RETRIEVE → PLAN-1 → INSPECT (read) → PLAN-2
                        → MODIFY → VALIDATE
                            passed  → VERIFY → index → FINAL (completed)
                            blocked → FINAL (blocked)
                            failed  → DIAGNOSE → REPAIR → VALIDATE (bounded)
                                      exhausted → STOPPED (failed)
"""
from __future__ import annotations

import os
import re

from semrag.reasoning.coding.files import ProjectFileError
from semrag.reasoning.coding.intent import (
    clarify_reason,
    classify_intent,
    is_create_request,
    request_target_noun,
)
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
from semrag.reasoning.coding.prompts import (
    _diagnose_prompt,
    _plan_create_prompt,
    _plan_patch_prompt,
    _plan_patch_retry_prompt,
    _plan_targets_prompt,
    extract_json,
    extract_sv_content,
)
from semrag.reasoning.coding.tools import CodingToolset
from semrag.retrieval.models import RetrievalResult
from semrag.services.services import _apply_grounding_policy

_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_EXPLICIT_PATH_RE = re.compile(
    r"\b(?:tb|rtl)/[A-Za-z0-9_./\-]+\.(?:sv|svh|v|vh|py|c|h|cpp|hpp)\b"
)


def _explicit_paths_from_request(request_text: str) -> list[str]:
    """Paths the caller named verbatim in the request (e.g.
    ``tb/i2c_master_test_lib/i2c_master_read_test4.svh``). The controller
    forces these into the inspection set so the planner cannot skip the
    file the user actually asked about."""
    return list(dict.fromkeys(_EXPLICIT_PATH_RE.findall(request_text or "")))


def _compile_path_anchors(file_list: str) -> str:
    """Deterministic 'active compile path' discovery from the indexed file
    list: package files (``*_pkg.sv``/``*_pkg.svh``) and top modules
    (``*_top.sv``/``*_top.svh``). Reuses ``list_files()`` — no new
    retrieval phase. These are the files that ``include``-register tests
    and sequences, so a 'integrate into the compile path' request must
    touch the right one."""
    anchors: list[str] = []
    for line in (file_list or "").splitlines():
        path = line.strip().replace("\\", "/")
        if path.endswith(("_pkg.sv", "_pkg.svh", "_top.sv", "_top.svh")):
            anchors.append(path)
    return ", ".join(anchors)


def _extract_identifiers(text: str, limit: int = 4) -> list[str]:
    """Deterministically pull snake_case identifiers out of the request
    for the symbol index lookup (e.g. ``i2c_master_read_seq``)."""
    return list(dict.fromkeys(_IDENTIFIER_RE.findall(text)))[:limit]


def _to_retrieval_result(path: str, text: str, project_id: str) -> RetrievalResult:
    line_count = text.count("\n") + 1
    return RetrievalResult(
        chunk_id=f"code-read:{path}",
        project_id=project_id,
        relative_path=path,
        chunk_type="code",
        text=text,
        line_start=1,
        line_end=line_count,
        fused_score=0.0,
        signal_scores={},
    )


def _parse_plan_targets(raw: dict, state: CodingState) -> str | None:
    """Returns an error string (safe stop) or None on success."""
    if raw.get("insufficient"):
        return f"Planner found insufficient evidence: {raw['insufficient']}"
    files = raw.get("files_to_read") or []
    if not isinstance(files, list) or not files:
        return "Planner declared no files to read; cannot make a safe change."
    state.files_to_read = [f for f in files if isinstance(f, str)]
    surface = raw.get("change_surface") or []
    if isinstance(surface, list):
        state.change_surface = [f for f in surface if isinstance(f, str)]
    new_path = raw.get("new_file_path") or ""
    if isinstance(new_path, str):
        state.new_file_path = new_path.strip()
    target = raw.get("validation_target")
    if isinstance(target, str) and target:
        state.validation_target = target
    return None


def _parse_changes(raw: dict, state: CodingState) -> tuple[list[PlannedChange], str | None]:
    """Parse the PLAN-2 / DIAGNOSE changes list. Returns (changes, error)."""
    if raw.get("insufficient"):
        return [], f"Planner stopped: {raw['insufficient']}"
    raw_changes = raw.get("changes") or []
    if not isinstance(raw_changes, list) or not raw_changes:
        return [], "Planner declared no changes; nothing to do."
    changes: list[PlannedChange] = []
    for i, item in enumerate(raw_changes):
        if not isinstance(item, dict):
            return [], f"Change #{i + 1} is not an object."
        op = item.get("op")
        file_path = item.get("file")
        new_text = item.get("new_text")
        if op not in ("patch", "write") or not isinstance(file_path, str) or not isinstance(new_text, str):
            return [], f"Change #{i + 1} must have op (patch|write), file, and new_text."
        if not file_path.strip():
            return [], f"Change #{i + 1} has an empty file path."
        if op == "patch" and not isinstance(item.get("old_text"), str):
            return [], f"Patch change #{i + 1} is missing old_text."
        changes.append(
            PlannedChange(
                op=ChangeOp(op),
                file_path=file_path,
                new_text=new_text,
                old_text=item.get("old_text") if op == "patch" else None,
                reason=item.get("reason") or "",
            )
        )
    return changes, None


class CodingAgent:
    def __init__(self, llm, toolset: CodingToolset, settings, project_id: str | None = None):
        self._llm = llm
        self._toolset = toolset
        self._settings = settings
        self._project_id = project_id or getattr(toolset, "project_id", "")

    def run(self, request: CodingRequest) -> CodingResult:
        state = CodingState(request_text=request.request)
        state.intent = classify_intent(request.request)
        state.phase = CodingPhase.INTAKE
        state.validation_target = request.validation_target or "rtl_lint"

        result = CodingResult(
            request=request,
            intent=state.intent,
            validation_target=state.validation_target,
        )

        if state.intent in (CodingIntent.INFORMATION, CodingIntent.REVIEW):
            return self._handle_information(request, state, result)
        if state.intent is CodingIntent.AMBIGUOUS:
            return self._handle_ambiguous(request, state, result)
        return self._handle_change(request, state, result)

    # ------------------------------------------------------------------
    # Non-modifying paths
    # ------------------------------------------------------------------

    def _handle_information(self, request, state, result) -> CodingResult:
        state.phase = CodingPhase.RETRIEVE
        evidence = self._toolset.search(request.request, top_k=10)
        chunks = self._toolset.search_chunks(request.request, top_k=10)
        answer = self._generate_information_answer(request.request, evidence)
        final_answer, grounding = _apply_grounding_policy(
            request.request,
            answer,
            chunks,
            None,
            self._llm,
            self._settings,
        )
        result.final_answer = final_answer
        result.grounding = grounding
        result.final_status = "completed"
        state.phase = CodingPhase.FINAL
        return result

    def _generate_information_answer(self, request_text: str, evidence: str) -> str:
        prompt = (
            "You are a semiconductor RAG assistant grounded strictly in the "
            "indexed source. Answer the request using ONLY the retrieved "
            "evidence. Cite files inline as [Source: path]. If the evidence "
            "is insufficient, say so and do not invent files or symbols.\n\n"
            f"Request: {request_text}\n\nEvidence:\n{evidence or '(none)'}"
        )
        try:
            completion = self._llm.complete([{"role": "user", "content": prompt}])
            return (completion.content or "").strip()
        except Exception as exc:  # network path
            return (
                "I could not retrieve a grounded answer: the model call failed "
                f"({exc}). No files were modified."
            )

    def _handle_ambiguous(self, request, state, result) -> CodingResult:
        state.phase = CodingPhase.STOPPED
        result.final_status = "needs_clarification"
        result.needs_clarification = clarify_reason(state.intent, request.request)
        result.final_answer = result.needs_clarification
        state.note = "no modification: ambiguous request"
        return result

    # ------------------------------------------------------------------
    # Modifying path
    # ------------------------------------------------------------------

    def _handle_change(self, request, state, result) -> CodingResult:
        # ---- RETRIEVE (deterministic, evidence first) -----------------
        state.phase = CodingPhase.RETRIEVE
        # The planner only needs the STRONGEST source patterns to plan
        # against; the low-score tail of a top-10 pull routinely contains
        # coverage/scoreboard internals that distract a small model from a
        # test-creation request. top_k=8 keeps every strong pattern file
        # while dropping the noisy tail (retrieval core untouched).
        evidence = self._toolset.search(request.request, top_k=8)
        chunks = self._toolset.search_chunks(request.request, top_k=10)
        file_list = self._toolset.list_files()

        symbol_hits: list[str] = []
        for ident in _extract_identifiers(request.request):
            hit = self._toolset.find_symbol(ident)
            symbol_hits.append(f"{ident}: {hit}")
        symbol_text = "\n".join(symbol_hits)

        if not chunks and not file_list.strip():
            result.final_status = "insufficient_evidence"
            result.error = "No indexed evidence or files for this project."
            result.final_answer = (
                "I could not find any indexed evidence for this repository, so I "
                "stopped before making any change. No files were modified."
            )
            state.phase = CodingPhase.STOPPED
            return result

        # ---- PLAN-1: targets to read + change surface + validation -----
        state.phase = CodingPhase.PLAN
        registered = self._toolset.validation_targets()
        anchors = _compile_path_anchors(file_list)
        plan1 = self._llm_json(
            _plan_targets_prompt(
                request.request, evidence, file_list, symbol_text, ", ".join(registered), anchors
            )
        )
        if plan1 is None:
            return self._safe_stop(result, state, "Could not parse the reading plan.")
        plan_error = _parse_plan_targets(plan1, state)
        explicit = _explicit_paths_from_request(request.request)
        if plan_error is not None:
            if not explicit:
                return self._safe_stop(result, state, plan_error)
            # The request names concrete files: fall back to them as the
            # inspection set instead of stopping on a timid planner.
            state.files_to_read = [p for p in explicit if self._toolset.file_exists(p)]
            state.note = "planner declared insufficient evidence; fell back to the explicitly named file(s)"
            if not state.files_to_read:
                return self._safe_stop(result, state, plan_error)
        # Files the caller named in the request are always inspected: the
        # planner must not skip the file the user actually pointed at.
        for path in explicit:
            if path not in state.files_to_read and self._toolset.file_exists(path):
                state.files_to_read.append(path)
        # The declared change surface (files this change will modify) is
        # part of the inspection set: every file a plan may touch must be
        # read before any plan-2. This is the A3 'INSPECT ALL RELEVANT
        # FILES' step — the surface is identified, then fully inspected.
        for path in list(state.change_surface):
            if path not in state.files_to_read and self._toolset.file_exists(path):
                state.files_to_read.append(path)
        result.change_surface = list(state.change_surface)
        surface_text = ", ".join(state.change_surface) if state.change_surface else ""
        # A target explicitly pinned by the caller wins over the planner's pick.
        if request.validation_target:
            state.validation_target = request.validation_target
            result.validation_target = request.validation_target
        plan_summary = str(plan1.get("pattern_summary") or "no pattern summary")
        if state.new_file_path:
            plan_summary += f" | new_file_path: {state.new_file_path}"

        # ---- INSPECT: read exactly the planned files -------------------
        state.phase = CodingPhase.INSPECT
        inspected: dict[str, str] = {}
        existing_reads: list[str] = []
        for path in state.files_to_read:
            if not self._toolset.file_exists(path):
                continue
            try:
                content = self._toolset.read_file(path)
            except ProjectFileError as exc:
                return self._safe_stop(result, state, f"Failed to read {path}: {exc}")
            inspected[path] = content
            state.files_inspected.append(path)
            existing_reads.append(path)
        if not inspected:
            return self._safe_stop(
                result, state, "None of the planned files exist; cannot make a safe change."
            )
        state.files_to_read = existing_reads
        inspected_text = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in inspected.items()
        )

        # ---- PLAN-2: exact changes --------------------------------------
        state.phase = CodingPhase.PLAN
        create = is_create_request(request.request)
        # An explicit "integrate into the active compile path" request must
        # register the new file. The pure-create contract's optional
        # "changes" proved unreliable on the small model, so create+integrate
        # routes through the standard changes contract, where the planner
        # emits BOTH the write and the registration patch in one plan.
        integration_requested = (
            "integrat" in request.request.lower()
            or "compile path" in request.request.lower()
        )
        if create and integration_requested:
            target_path = self._derive_new_file_path(request, state)
            if not target_path:
                return self._safe_stop(
                    result, state,
                    "This looks like a file-creation request but I could not "
                    "determine a safe new-file path; stopping before any change.",
                )
            state.new_file_path = target_path
            plan2 = self._llm_json(
                _plan_patch_prompt(request.request, evidence, inspected_text, plan_summary, surface_text)
            )
            if plan2 is None:
                return self._safe_stop(result, state, "Could not parse the change plan.")
            changes, change_error = self._parse_plan2(plan2, state)
            if change_error is not None:
                return self._safe_stop(result, state, change_error)
            target = plan2.get("validation_target") or state.validation_target
            if not request.validation_target and isinstance(target, str):
                state.validation_target = target
                result.validation_target = target
        elif create:
            target_path = self._derive_new_file_path(request, state)
            if not target_path:
                return self._safe_stop(
                    result, state,
                    "This looks like a file-creation request but I could not "
                    "determine a safe new-file path; stopping before any change.",
                )
            state.new_file_path = target_path
            raw = self._llm_text(
                _plan_create_prompt(request.request, evidence, inspected_text, target_path, plan_summary, surface_text)
            )
            content = extract_sv_content(raw)
            if not content:
                return self._safe_stop(result, state, "Could not parse the new-file content from the planner.")
            changes = [
                PlannedChange(
                    op=ChangeOp.WRITE,
                    file_path=target_path,
                    new_text=content,
                    reason="file-creation request",
                )
            ]
            parsed = extract_json(raw)
            if parsed is not None:
                target = parsed.get("validation_target")
                if not request.validation_target and isinstance(target, str) and target:
                    state.validation_target = target
                    result.validation_target = target
        else:
            plan2 = self._llm_json(
                _plan_patch_prompt(request.request, evidence, inspected_text, plan_summary, surface_text)
            )
            if plan2 is None:
                return self._safe_stop(result, state, "Could not parse the change plan.")
            changes, change_error = self._parse_plan2(plan2, state)
            if change_error is not None:
                return self._safe_stop(result, state, change_error)
            target = plan2.get("validation_target") or state.validation_target
            if not request.validation_target and isinstance(target, str):
                state.validation_target = target
                result.validation_target = target

        # ---- read-before-write enforcement (with bounded corrective read) ----
        violation = self._plan_violation(changes, state)
        if violation is not None:
            # Deterministically read any referenced EXISTING file that was
            # not yet inspected (reading is safe and keeps the guarantee:
            # the controller reads every file before any write). Then one
            # corrective replan with the expanded content.
            unread: list[str] = []
            for c in changes:
                targets_existing = (
                    c.op is ChangeOp.PATCH
                    or (c.op is ChangeOp.WRITE and self._toolset.file_exists(c.file_path))
                )
                if targets_existing and c.file_path not in state.files_to_read:
                    unread.append(c.file_path)
            for path in dict.fromkeys(unread):
                if not self._toolset.file_exists(path):
                    continue
                try:
                    inspected[path] = self._toolset.read_file(path)
                except ProjectFileError as exc:
                    return self._safe_stop(result, state, f"Failed to read {path}: {exc}")
                state.files_to_read.append(path)
                state.files_inspected.append(path)
            inspected_text = "\n\n".join(
                f"--- {path} ---\n{content}" for path, content in inspected.items()
            )
            retry = self._llm_json(
                _plan_patch_retry_prompt(request.request, evidence, inspected_text, plan_summary, violation, surface_text)
            )
            if retry is not None:
                changes, change_error = _parse_changes(retry, state)
                if change_error is None:
                    violation = self._plan_violation(changes, state)
            if change_error is not None or violation is not None:
                return self._safe_stop(
                    result, state, f"Change plan rejected: {violation or change_error}"
                )

        # A write targeting an existing (read) file becomes a safe full
        # replacement patch against the inspected content.
        changes = [
            PlannedChange(
                op=ChangeOp.PATCH if (
                    c.op is ChangeOp.WRITE and self._toolset.file_exists(c.file_path)
                ) else c.op,
                file_path=c.file_path,
                new_text=c.new_text,
                old_text=inspected.get(c.file_path) if (
                    c.op is ChangeOp.WRITE and self._toolset.file_exists(c.file_path)
                ) else c.old_text,
                reason=c.reason,
            )
            for c in changes
        ]

        # ---- MODIFY: apply exactly the declared changes ----------------
        state.phase = CodingPhase.MODIFY
        preflight = self._preflight_changes(changes, state)
        if preflight is not None:
            return self._safe_stop(result, state, f"Change plan would not apply atomically: {preflight}")
        state.snapshot_before = self._toolset.snapshot_files()
        self._apply_changes(changes, inspected, state)
        result.changes = list(state.changes)
        result.files_modified = list(state.files_modified)
        if not state.changes or not all(c.applied for c in state.changes):
            result.final_status = "failed"
            applied_count = sum(1 for c in state.changes if c.applied)
            if applied_count:
                result.error = (
                    "Change application was not atomic: some declared changes applied, "
                    "some failed. See 'changes' for the exact per-file state."
                )
            else:
                result.error = "Change application failed; no declared change was applied."
            state.phase = CodingPhase.STOPPED
            result.final_answer = self._build_summary(request, state, None, repaired=False)
            return result

        # ---- VALIDATE --------------------------------------------------
        state.phase = CodingPhase.VALIDATE
        self._run_validation(state, result)
        last_validation = state.validation_attempts[-1]

        if last_validation.status == "passed":
            return self._verify_and_finalize(request, state, result, repaired=False)
        if last_validation.status == "blocked":
            return self._finalize_blocked(request, state, result)

        # ---- REPAIR loop (bounded) -------------------------------------
        repaired = False
        while state.repair_attempts < request.max_repair_attempts:
            state.phase = CodingPhase.DIAGNOSE
            diag = self._llm_json(
                _diagnose_prompt(
                    request.request,
                    plan_summary,
                    self._format_validation(last_validation),
                    self._format_applied(state.changes),
                    self._format_current_files(state),
                    request.max_repair_attempts - state.repair_attempts,
                )
            )
            if diag is None:
                state.note = "Repair planner output could not be parsed; stopping safely."
                break
            repair_changes, repair_error = self._parse_repair(diag, state)
            if repair_error is not None:
                state.note = repair_error
                break
            if not repair_changes:
                state.note = "Repair planner did not produce a change; stopping safely."
                break
            # Repairs may only touch files that were inspected or already modified.
            allowed = set(state.files_to_read) | set(state.files_modified)
            if any(
                c.file_path not in allowed and not (
                    c.op is ChangeOp.WRITE and not self._toolset.file_exists(c.file_path)
                )
                for c in repair_changes
            ):
                state.note = "Repair targeted a file never read nor modified; refusing."
                break

            state.phase = CodingPhase.REPAIR
            self._apply_changes(repair_changes, inspected, state)
            state.repair_attempts += 1
            result.repair_attempts = state.repair_attempts
            result.changes = list(state.changes)
            result.files_modified = list(state.files_modified)
            if not state.changes or not all(c.applied for c in state.changes):
                break

            state.phase = CodingPhase.VALIDATE
            self._run_validation(state, result)
            last_validation = state.validation_attempts[-1]
            if last_validation.status == "passed":
                repaired = True
                break
            if last_validation.status == "blocked":
                break

        if last_validation.status == "blocked":
            return self._finalize_blocked(request, state, result)
        if last_validation.status == "passed":
            return self._verify_and_finalize(request, state, result, repaired=repaired)

        # Repair exhausted → safe stop with honest reporting.
        result.final_status = "failed"
        result.error = (
            f"Validation still failing after {state.repair_attempts} repair "
            f"attempt(s) (max {request.max_repair_attempts})."
        )
        state.phase = CodingPhase.STOPPED
        result.final_answer = self._build_summary(request, state, last_validation, repaired=False)
        return result

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _llm_text(self, prompt: str) -> str:
        completion = None
        try:
            completion = self._llm.complete([{"role": "user", "content": prompt}])
        except Exception:  # pragma: no cover - network path
            pass
        content = getattr(completion, "content", None) or ""
        if os.environ.get("SEMRAG_CODING_TRACE"):
            with open("/tmp/opencode/a2_llm_trace.log", "a", encoding="utf-8") as fh:
                fh.write("===== PROMPT =====\n" + prompt[:2000] + "\n===== RESPONSE =====\n" + content[:3000] + "\n\n")
        return content

    def _llm_json(self, prompt: str) -> dict | None:
        return extract_json(self._llm_text(prompt))

    def _pattern_dir(self, state: CodingState) -> str | None:
        """Pick the directory for a new file deterministically from the
        inspected files: prefer a test/seq library, then any tb/ path,
        then the first file's directory."""
        for path in state.files_inspected:
            p = path.replace("\\", "/")
            if "test_lib" in p or "seq_lib" in p:
                return p.rsplit("/", 1)[0]
        for path in state.files_inspected:
            p = path.replace("\\", "/")
            if p.startswith("tb/") or "/tb/" in p:
                return p.rsplit("/", 1)[0]
        for path in state.files_inspected:
            p = path.replace("\\", "/")
            if "/" in p:
                return p.rsplit("/", 1)[0]
        return None

    def _derive_new_file_path(self, request: CodingRequest, state: CodingState) -> str | None:
        """Deterministically choose a non-existing path for a new file.
        Uses plan-1's proposed path if it does not exist; otherwise builds
        ``i2c_master_<noun>`` in the pattern directory, deduplicated."""
        if state.new_file_path and not self._toolset.file_exists(state.new_file_path):
            return state.new_file_path
        noun = request_target_noun(request.request)
        if not noun:
            return None
        directory = self._pattern_dir(state)
        if not directory:
            return None
        base = f"i2c_master_{noun}"
        ext = ".svh" if "test_lib" in directory or "seq_lib" in directory else ".sv"
        existing = {
            f.strip().replace("\\", "/")
            for f in self._toolset.list_files().splitlines()
            if f.strip()
        }
        n = 1
        while True:
            name = f"{base}{n if n > 1 else ''}{ext}"
            path = f"{directory}/{name}"
            if path not in existing and not self._toolset.file_exists(path):
                return path
            n += 1

    def _read_before_write_violation(self, changes: list[PlannedChange], state: CodingState) -> str | None:
        """Deterministic read-before-write check. Returns an error string
        (with the exact violation) or None if every change is legal."""
        for c in changes:
            if c.op is ChangeOp.PATCH and c.file_path not in state.files_to_read:
                return (
                    f"Patch targets {c.file_path} which was not read during inspection "
                    "(read-before-write violated)."
                )
            if c.op is ChangeOp.WRITE and self._toolset.file_exists(c.file_path):
                if c.file_path not in state.files_to_read:
                    return (
                        f"Write targets existing unread file {c.file_path}; refusing to clobber."
                    )
        return None

    def _parse_repair(self, diag: dict | None, state: CodingState) -> tuple[list[PlannedChange], str | None]:
        """Parse the DIAGNOSE repair. Accepts either a patch list (``changes``)
        or a full-file rewrite (``files`` → path → content). Full rewrites
        become atomic full-replacement patches against the CURRENT content."""
        if diag is None:
            return [], "Repair output could not be parsed; stopping safely."
        files = diag.get("files")
        if isinstance(files, dict):
            changes: list[PlannedChange] = []
            # Full-file rewrites may only target files the workflow already
            # modified — never an inspected-but-untouched pattern file.
            modifiable = set(state.files_modified)
            for path, content in files.items():
                if not isinstance(path, str) or not isinstance(content, str) or not content.strip():
                    continue
                if path not in modifiable and self._toolset.file_exists(path):
                    return [], f"Repair rewrite targets {path} which the workflow never modified."
                try:
                    current = self._toolset.read_file(path)
                except ProjectFileError:
                    current = None
                if current is None:
                    return [], f"Repair rewrite targets {path} which does not exist."
                changes.append(
                    PlannedChange(
                        op=ChangeOp.PATCH,
                        file_path=path,
                        new_text=content,
                        old_text=current,
                        reason="full-file rewrite from repair planner",
                    )
                )
            if changes:
                return changes, None
        return _parse_changes(diag or {}, state)

    def _parse_plan2(self, diag: dict | None, state: CodingState) -> tuple[list[PlannedChange], str | None]:
        """Parse the PLAN-2 output. Accepts the patch list (``changes``) or a
        full-file rewrite (``files`` → path → content) for inspected files."""
        if diag is None:
            return [], "Could not parse the change plan."
        files = diag.get("files")
        if isinstance(files, dict) and files:
            changes: list[PlannedChange] = []
            for path, content in files.items():
                if not isinstance(path, str) or not isinstance(content, str) or not content.strip():
                    continue
                if path not in state.files_to_read:
                    return [], (
                        f"Rewrite targets {path} which was not read during inspection "
                        "(read-before-write violated)."
                    )
                try:
                    current = self._toolset.read_file(path)
                except ProjectFileError:
                    current = None
                if current is None:
                    return [], f"Rewrite targets {path} which does not exist."
                changes.append(
                    PlannedChange(
                        op=ChangeOp.PATCH,
                        file_path=path,
                        new_text=content,
                        old_text=current,
                        reason="full-file rewrite from plan",
                    )
                )
            if changes:
                return changes, None
        return _parse_changes(diag or {}, state)

    def _plan_violation(self, changes: list[PlannedChange], state: CodingState) -> str | None:
        """Combined plan-2 discipline check (read-before-write + minimal
        change): no unrelated destructive deletions, no touching files
        beyond the declared change surface, bounded scope, and creation
        requests must create the declared new file."""
        violation = self._read_before_write_violation(changes, state)
        if violation is not None:
            return violation
        for c in changes:
            if c.op is ChangeOp.PATCH and not (c.new_text or "").strip():
                return (
                    f"Change to {c.file_path} is a pure deletion (empty new_text); "
                    "unrelated destructive changes are not allowed in a plan."
                )
        # A3 change-surface discipline: when the planner identified the
        # files the change may modify, the plan must stay within them
        # (plus the caller's explicitly named files and the new file).
        surface = [p for p in state.change_surface if p]
        if surface:
            surface_set = set(surface)
            explicit_set = set(_explicit_paths_from_request(state.request_text))
            for c in changes:
                if (
                    c.file_path not in surface_set
                    and c.file_path not in explicit_set
                    and c.file_path != state.new_file_path
                ):
                    return (
                        f"Change targets {c.file_path}, which is NOT in the declared "
                        f"change surface ({', '.join(sorted(surface_set)) or '(empty)'}); "
                        "the plan must stay within the files identified for modification."
                    )
        distinct = {c.file_path for c in changes}
        max_files = int(getattr(self._settings, "coding_max_plan_files", 6))
        if len(distinct) > max_files:
            return (
                f"Plan touches {len(distinct)} distinct files; a minimal change "
                f"must stay within {max_files} files."
            )
        if state.new_file_path and not any(
            c.op is ChangeOp.WRITE and c.file_path == state.new_file_path for c in changes
        ):
            return (
                f"The request creates a file but the plan does not write the declared "
                f"new file {state.new_file_path}; add an op 'write' to that path."
            )
        explicit = _explicit_paths_from_request(state.request_text)
        if explicit and not any(c.file_path in explicit for c in changes):
            return (
                "The request names specific file(s) but the plan does not touch any of "
                f"them ({', '.join(explicit)}); stay on the files the request points at."
            )
        return None

    def _preflight_changes(self, changes: list[PlannedChange], state: CodingState) -> str | None:
        """Before ANY write: verify every patch's old_text matches exactly
        once in the CURRENT file content and every target is legal. A
        single failure aborts the whole plan with zero modifications —
        the atomicity that lets a rejected plan leave the repo untouched."""
        for c in changes:
            if c.op is ChangeOp.PATCH:
                try:
                    content = self._toolset.read_file(c.file_path)
                except ProjectFileError as exc:
                    return f"cannot preflight {c.file_path}: {exc}"
                matches = content.count(c.old_text or "")
                if matches != 1:
                    return (
                        f"old_text for {c.file_path} matches {matches} times "
                        "(exactly 1 required); no change applied."
                    )
            else:  # WRITE
                if self._toolset.file_exists(c.file_path) and c.file_path not in state.files_to_read:
                    return f"write targets existing unread file {c.file_path}"
        return None

    def _apply_changes(self, changes: list[PlannedChange], inspected: dict[str, str], state: CodingState) -> None:
        """Apply changes through the toolset, recording FileChange records."""
        for c in changes:
            record = FileChange(op=c.op, file_path=c.file_path, expected_text=c.new_text)
            try:
                if c.op is ChangeOp.PATCH:
                    self._toolset.apply_patch(c.file_path, c.old_text or "", c.new_text)
                else:
                    self._toolset.write_file(c.file_path, c.new_text)
                record.applied = True
                state.files_modified.append(c.file_path)
            except ProjectFileError as exc:
                record.error = str(exc)
            except Exception as exc:  # noqa: BLE001 - surface as safe stop
                record.error = str(exc)
            state.changes.append(record)

    def _run_validation(self, state: CodingState, result: CodingResult) -> ValidationAttempt:
        v = self._toolset.run_validation(state.validation_target)
        attempt = ValidationAttempt(
            target=v.target or state.validation_target,
            status=v.status,
            exit_code=v.exit_code,
            stdout=v.stdout,
            stderr=v.stderr,
            command=v.command,
            blocked_reason=v.blocked_reason,
            duration_seconds=v.duration_seconds,
        )
        state.validation_attempts.append(attempt)
        result.validation_attempts = list(state.validation_attempts)
        return attempt

    def _verify_and_finalize(self, request, state, result, repaired: bool) -> CodingResult:
        state.phase = CodingPhase.VERIFY
        # Read-back each changed file against the exact text we wrote.
        for record in state.changes:
            if not record.applied:
                continue
            try:
                content = self._toolset.read_file(record.file_path)
                expected = record.expected_text or ""
                record.verified = content == expected if record.op is ChangeOp.WRITE else expected in content
            except ProjectFileError:
                record.verified = False

        # No unrelated files may have changed.
        after = self._toolset.snapshot_files()
        modified_set = set(state.files_modified)
        unrelated = (after - state.snapshot_before) - modified_set
        if unrelated:
            state.note = f"Unrelated files changed during workflow: {sorted(unrelated)}"

        # Index freshness.
        reindexed = False
        retrievable: list[str] = []
        if state.changes and any(c.requires_reindex for c in state.changes):
            idx = self._toolset.index()
            reindexed = idx.get("status") == "completed"
            for path in state.files_modified:
                if self._toolset.is_retrievable(path):
                    retrievable.append(path)

        result.final_status = "completed"
        state.phase = CodingPhase.FINAL
        result.final_answer = self._build_summary(
            request, state, state.validation_attempts[-1], repaired=repaired,
            reindexed=reindexed, retrievable=retrievable,
        )
        result.grounding = self._ground_summary(request, state, result.final_answer)
        return result

    def _finalize_blocked(self, request, state, result) -> CodingResult:
        result.final_status = "blocked"
        state.phase = CodingPhase.FINAL
        result.final_answer = self._build_summary(request, state, state.validation_attempts[-1], repaired=False)
        result.grounding = self._ground_summary(request, state, result.final_answer)
        return result

    def _ground_summary(self, request: CodingRequest, state: CodingState, final_answer: str) -> dict | None:
        """Run the A1 gate on the workflow summary using the read/changed
        files as evidence. Regeneration is disabled: the summary is a
        deterministic record, so the gate only checks it, never rewrites."""
        evidence: list[RetrievalResult] = []
        for path in list(state.files_inspected) + list(state.files_modified):
            try:
                evidence.append(_to_retrieval_result(path, self._toolset.read_file(path), request.project_id))
            except ProjectFileError:
                continue
        if not evidence:
            return None
        _final, grounding = _apply_grounding_policy(
            request.request,
            final_answer,
            evidence,
            None,
            self._llm,
            self._settings.model_copy(update={"grounding_gate_regenerate": False}),
        )
        return grounding

    def _safe_stop(self, result: CodingResult, state: CodingState, message: str) -> CodingResult:
        result.final_status = "failed"
        result.error = message
        state.phase = CodingPhase.STOPPED
        state.note = message
        result.final_answer = (
            f"I stopped before modifying any files: {message}\n\n"
            "No changes were applied."
        )
        return result

    # ------------------------------------------------------------------
    # Honest summary rendering
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        request: CodingRequest,
        state: CodingState,
        validation: ValidationAttempt | None,
        repaired: bool,
        reindexed: bool | None = None,
        retrievable: list[str] | None = None,
    ) -> str:
        lines: list[str] = ["## Request", request.request, "", "## Intent", state.intent.value, ""]
        lines.append("## Change surface")
        if state.change_surface:
            lines.append(f"- identified for modification: {', '.join(state.change_surface)}")
        else:
            lines.append("- (not declared; single-file change)")
        lines.append("")
        lines.append("## Changes applied")
        if state.changes:
            for c in state.changes:
                lines.append(f"- {c.file_path} ({c.op.value}): applied={c.applied} verified={c.verified}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Validation")
        if validation is not None:
            lines.append(f"- target: {validation.target}")
            lines.append(f"- status: {validation.status}")
            lines.append(f"- command: {validation.command or '(not run)'}")
        else:
            lines.append("- (no validation ran)")
        lines.append("")
        lines.append("## Repair attempts")
        if state.repair_attempts:
            lines.append(f"- {state.repair_attempts} repair attempt(s) applied")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Indexing")
        if reindexed is not None:
            lines.append(f"- re-indexed: {'yes' if reindexed else 'no'}")
            lines.append(f"- changed file retrievable: {', '.join(retrievable or []) or 'none verified'}")
        else:
            lines.append("- not performed")
        lines.append("")
        lines.append("## Not done / caveats")
        lines.append(f"- {state.note or 'none'}")
        if validation is not None and validation.status == "passed":
            lines.append(
                f"- Note: only {validation.target} was run; this is not a full "
                "simulation result."
            )
        lines.append("")
        lines.append("## Sources")
        seen: set[str] = set()
        for path in list(state.files_modified) + list(state.files_inspected):
            if path in seen:
                continue
            seen.add(path)
            lines.append(f"- [Source: {path}]")
        return "\n".join(lines)

    def _format_validation(self, v: ValidationAttempt) -> str:
        return (
            f"target={v.target} status={v.status} command={v.command}\n"
            f"stdout:\n{v.stdout[:2000]}\nstderr:\n{v.stderr[:2000]}"
        )

    def _format_applied(self, changes: list[FileChange]) -> str:
        return "\n".join(
            f"- {c.op.value} {c.file_path} applied={c.applied} error={c.error or 'none'}"
            for c in changes
        )

    def _format_current_files(self, state: CodingState) -> str:
        """Current on-disk content of every file the workflow has modified,
        for the repair planner to rewrite or patch precisely."""
        blocks: list[str] = []
        for path in dict.fromkeys(state.files_modified):
            try:
                blocks.append(f"--- {path} ---\n{self._toolset.read_file(path)}")
            except ProjectFileError:
                continue
        return "\n\n".join(blocks)