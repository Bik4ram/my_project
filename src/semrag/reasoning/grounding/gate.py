"""Grounding gate (A1).

A deterministic, evidence-local, post-generation check. It answers one
question:

    "Does the generated answer make source-specific claims that are
    supported by the evidence that was supplied to generation?"

Why this exists: the local 7B LLM occasionally emits plausible
SystemVerilog or architecture claims that are NOT present in the
retrieved context (the R4 trace-query fabrication of ``i2c_master_test``
and ``sequencer::run_phase``). The existing citation validator only
checks paths inside ``[Source: ...]`` citations, so an answer that simply
never cites anything — and invents symbols instead — passes with a
fabrication rate of 0.0. This gate closes that gap.

Design rules (A1 spec):

- Operates strictly AFTER retrieval and AFTER LLM answer generation.
  It never touches retrieval; its evidence is exactly the chunks that
  were included in the prompt (plus the optional FILE INDEX).
- Prefers deterministic verification over LLM judgement. There is no
  second LLM call inside this module.
- Distinguishes *strong* (repository-specific) entities from *weak*
  generic UVM/SystemVerilog/English terms, so ordinary explanatory
  language ("UVM uses sequences to generate transactions") is never
  blocked.
- A reference to an entity that is absent from the evidence is an
  unsupported claim *unless* the surrounding sentence is a negation
  ("does not exist", "not found"), in which case it is a legitimate
  negative-query reference, not a fabricated fact.
- A source-specific answer that carries no valid citation at all is
  treated as insufficiently grounded rather than trusted on face value.
- File paths / symbols / methods that are not in the evidence are never
  accepted as grounded — no arbitrary filesystem probing.

The module is pure (no I/O, no LLM, no DB); the answer policy that acts
on a ``GroundingReport`` (regenerate / return supported portion /
controlled response) lives in the service layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from semrag.evaluation.citation_validator import _CITATION_PATTERN, validate_citations
from semrag.reasoning.prompting.citation import format_chunk_with_citation
from semrag.retrieval.models import RetrievalResult

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# A file path, e.g. tb/i2c_master_agent/i2c_master_driver.svh or
# [Source: ...] content is handled by the citation validator separately.
_SLASH_PATH_RE = re.compile(r"\b(?:[A-Za-z0-9_]+/)+[A-Za-z0-9_.-]+\b")
# A dotted member path, e.g. mstr_env.mstr_agnt.sqr or item.rd_wr.
_DOT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}(?:\.[A-Za-z_][A-Za-z0-9_]{1,}){1,3}\b")
# A SystemVerilog qualified symbol, e.g. i2c_master_sequencer::run_phase.
_QUALIFIED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}::[A-Za-z_][A-Za-z0-9_]{2,}\b")
# A file-like suffix on a path component (e.g. "rtl/nonexistent.sv").
_PATH_EXT_RE = re.compile(r"\.(?:sv|svh|v|vh|vhd|vhdl|c|cc|cpp|h|hpp|py|md|txt|json|yaml|yml|toml)\b")
# Any bare identifier.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# SystemVerilog literal fragments: the alphanumeric part of ``7'h50``,
# ``3'd7``, ``2'b01``, ``4'o7``, or a bare run of digits.
_LITERAL_RE = re.compile(r"(?:[hH][0-9a-fA-F]+|[dD][0-9]+|[bB][01]+|[oO][0-7]+|[0-9]+)")
# Fenced code blocks are treated atomically by the answer policy.
_CODE_FENCE_RE = re.compile(r"```")
# Sentence boundary: newline, whitespace after . ! ?, or the "]. " of a
# finished inline citation ("... lines 20-26]. The read sequence...").
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=\].)\s+|\n+")

# Words that are generic UVM / SystemVerilog / English vocabulary rather
# than repository-specific symbols. A token in this set is *weak*: it can
# never by itself make an answer insufficiently grounded, and a bare
# reference to it is never treated as an unsupported claim. Repository
# symbols (e.g. ``i2c_master_read_seq``) never appear here.
_GENERIC_TERMS = frozenset(
    """
    uvm_test uvm_sequence uvm_sequencer uvm_driver uvm_monitor uvm_agent uvm_env
    uvm_environment uvm_scoreboard uvm_component uvm_object uvm_sequence_item
    uvm_transaction uvm_config_db uvm_factory uvm_phase uvm_sequence_lib
    uvm_object_registry uvm_component_registry uvm_analysis_port uvm_analysis_imp
    uvm_analysis_export uvm_tlm_analysis_fifo uvm_analysis_fifo
    uvm_seq_item_pull_port uvm_seq_item_pull_imp uvm_blocking_put_port
    uvm_nonblocking_put_port uvm_blocking_get_port uvm_put_port uvm_get_port
    uvm_tlm_fifo uvm_reg uvm_reg_block uvm_reg_map uvm_reg_field uvm_reg_model
    uvm_root uvm_top uvm_report_server uvm_status_e uvm_reg_adapter
    uvm_bus_reg_adapter uvm_callback uvm_callbacks uvm_object_wrapper
    uvm_sequence_base uvm_sequencer_base uvm_driver_base uvm_monitor_base
    uvm_component_base uvm_object_base uvm_agent_base uvm_env_base uvm_test_base

    run_phase build_phase connect_phase end_of_elaboration_phase
    start_of_simulation_phase reset_phase configure_phase main_phase
    shutdown_phase pre_reset_phase post_reset_phase pre_configure_phase
    post_configure_phase pre_main_phase post_main_phase pre_shutdown_phase
    post_shutdown_phase report_phase extract_phase check_phase final_phase
    body new create print copy clone compare pack unpack convert2string
    get_type_name get_name get_full_name get_parent set_name get_object
    start_item finish_item wait_for_grant pre_do mid_do post_do
    get_next_item try_next_item item_done get_response peek get put send write
    read seq_item_port seq_item_export rsp_port rsp_export raise_objection
    drop_objection phase_ready_to_end set_drain_time set_report_verbosity_level
    set_config_object set_config_int set_config_string get_config_object
    get_config_int get_config_string uvm_config_int uvm_config_string
    uvm_config_object uvm_object_utils uvm_component_utils
    uvm_object_utils_begin uvm_object_utils_end uvm_component_utils_begin
    uvm_component_utils_end uvm_field_int uvm_field_enum uvm_field_string
    uvm_field_object uvm_fatal uvm_error uvm_info uvm_warning
    uvm_report_fatal uvm_report_error uvm_report_info uvm_report_warning
    randomize randomize_with srandom get_randstate set_randstate type_id

    module endmodule interface endinterface program endprogram package
    endpackage class endclass function endfunction task endtask typedef struct
    union enum logic wire reg bit byte int integer shortint longint time real
    event virtual input output inout parameter localparam generate endgenerate
    always always_comb always_ff always_latch initial begin end if else case
    casex casez endcase for foreach while do repeat forever return break
    continue assign deassign wait fork join join_any join_none disable posedge
    negedge default inside within rand randc constraint solve dist soft unique
    priority automatic static extern pure protected covergroup coverpoint cross
    bins illegal_bins ignore_bins binsof iff import export include define ifdef
    ifndef endif timescale default_nettype undef resetall standard assert
    assume cover property sequence of at least until throughout

    display monitor write fwrite readmemh readmemb random urandom sformatf
    timeformat realtime finish stop cast typename onehot onehot0 countones
    isunknown stable rose fell past sample key ungetc readline fopen fclose
    fscanf fgets ftell rewind urandom_range fatal error warning info

    testbench test sequence sequencer driver monitor agent environment scoreboard
    transaction interface vif config coverage covergroup coverpoint bin bins
    packet address data clock reset bus signal port phase task function class
    module method response request item callback factory registry seed constraint
    byte word register frame transaction queue mailbox semaphore field value bit
    state idle start stop ack nack ackstop read write ready valid enable clock
    resetn rst rst_n clk sda scl din dout addr sel en we re cs oe irq
    req drv mon agt sqr cfg env vif

    source lines line sentence answer question context file files index path
    paths above below section sections code snippet snippets excerpt evidence
    this that these those there here with from into onto within over under and or
    not no none never without cannot cant does doesnot is not was were has have
    had been be being but however also only just then then now when where which
    what who whom whose why how if then else the a an of to in for on at as by
    it its he she they them their our your my his her me we us you one two three
    four five six seven eight nine ten zero
    use uses used using provide provides provided providing perform performs
    performing run runs running called calls calling define defines defined
    defining contain contains containing include includes including generate
    generates generating create creates creating handle handles handling
    support supports supported start starts starting stop stops stopped
    send sends sending receive receives receiving write writes written writing
    read reads reading drive drives driving driven control controls controlling
    manage manages managing return returns returning set sets setting wait
    waits waiting check checks checking verify verifies verifying test tests
    testing simulate simulates simulating build builds building compile compiles
    compiling execute executes executing process processes processing apply
    applies applied assign assigns assigned declare declares declared
    implement implements implementing construct constructs constructing
    instantiate instantiates instantiating extend extends extending derived
    override overrides overriding connect connects connecting flow flows
    hierarchy hierarchical component components instance instances
    operation operations behavior behaviors value values
    constrain constrains constrained cause causes caused occur occurs occurring
    describe describes describing represent represents representing indicate
    indicates indicated indicating relate relates relating produce produces
    producing send sends sending fetch fetches fetching query queries queried
    trace tracing traced let initialize initializing initialized example steps
    step step-by-step through during after before because since while following
    """.split()
)

# Phrases that mark a reference as a *negative* mention (a claim of
# absence) rather than an affirmative claim about the entity. Applied to
# the sentence containing the entity.
_NEGATION_PATTERNS = re.compile(
    r"(?:\b(?:no|none|not|never|without|cannot|can't|doesn't|don't|isn't|"
    r"aren't|wasn't|weren't|unable|missing|absent|nowhere)\b|"
    r"\b(?:does|is|are|was|were|has|have|do)\s+not\b|"
    r"\bnot\s+(?:found|present|exist|supported|mentioned|referenced|listed|"
    r"shown|defined|declared|indicated|found|matched|located)\b|"
    r"\bno\s+(?:evidence|indication|mention|trace|match|result|such|matching|"
    r"corresponding|direct|indication)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Evidence model (request-local; no schema change)
# ---------------------------------------------------------------------------


def build_evidence(
    included_chunks: list[RetrievalResult],
    file_index_text: str | None = None,
) -> dict:
    """Build the request-local evidence vocabulary from the chunks that
    were actually included in the generation prompt plus the optional
    FILE INDEX block. Everything here is "what the model was given".
    """
    paths: set[str] = set()
    symbols: set[str] = set()
    texts: list[str] = []

    for chunk in included_chunks:
        paths.add(chunk.relative_path)
        if chunk.symbol_name:
            symbols.add(chunk.symbol_name)
        texts.append(chunk.text)

    text = "\n".join(texts)

    if file_index_text:
        for line in file_index_text.splitlines():
            stripped = line.strip()
            if "/" in stripped and not stripped.startswith("category:"):
                paths.add(stripped)

    for p in paths:
        try:
            symbols.add(PurePosixPath(p).stem)
        except ValueError:
            pass

    symbols.update(_IDENT_RE.findall(text))

    return {
        "paths": paths,
        "symbols": symbols,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class EntityReference:
    kind: str  # "path" | "qualified" | "dot" | "symbol"
    token: str
    status: str = "pending"  # present | unsupported | negated
    sentence: str = ""

    @property
    def strong(self) -> bool:
        return self.kind in ("path", "qualified", "dot")


def _overlaps(span: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    start, end = span
    for s, e in used:
        if start < e and s < end:
            return True
    return False


def _is_weak_term(token: str) -> bool:
    """True for generic vocabulary. Handles plural forms ("classes" ->
    "class"), sentence-start capitalization ("The", "Classes"),
    SystemVerilog literal fragments (``7'h50`` -> ``h50``), and plain
    lowercase prose. Repository-flavored tokens (containing ``_`` or a
    digit, e.g. ``i2c_master_driver``) are never weakened by the
    case-insensitive fallback; PascalCase classes (``FakeMonitor``)
    are not weakened either (only a first-letter-capitalized prose
    word like "Base" or "The" is)."""
    if token in _GENERIC_TERMS:
        return True
    if _LITERAL_RE.fullmatch(token):
        return True
    low = token.lower()
    if low in _GENERIC_TERMS:
        return True
    if low.endswith("s"):
        base = low[:-1]
        if base in _GENERIC_TERMS:
            return True
        if base.endswith("s") and base[:-1] in _GENERIC_TERMS:
            return True
    if "_" in token or any(ch.isdigit() for ch in token):
        return False
    # No underscore/digit: repository symbols here are snake_case or
    # PascalCase. A plain-lowercase word, or a single-capitalized prose
    # word ("Base", "The", "Tracing"), is generic.
    return token.islower() or token == token.capitalize()


def _classify_bare(token: str) -> bool:
    """True if ``token`` is a *strong* (repository-specific) symbol that
    the gate cares about. Plain lowercase English words (no underscore,
    no digit, e.g. "validates", "all") are prose, not repository symbols,
    and are never treated as unsupported claims on their own — repo
    symbols in this stack are snake_case (``i2c_master_driver``) or
    PascalCase (``FakeMonitor``) or qualified (``A::B``)."""
    if _is_weak_term(token):
        return False
    if token.isupper():
        return False
    if token.islower() and "_" not in token and not any(ch.isdigit() for ch in token):
        return False
    return True


def _dot_is_weak(token: str) -> bool:
    """``e.g.``, ``i.e.`` and similar dot-abbreviations are generic; a
    real member path (``mstr_env.mstr_agnt.sqr``) is not."""
    return all(_is_weak_term(seg) or len(seg) <= 2 for seg in token.split("."))


def _slash_is_weak(token: str) -> bool:
    """``read/write`` in prose is not a file path; a real relative path
    (``rtl/nonexistent.sv``, ``tb/.../i2c_master_driver.svh``) has a file
    extension, a dotted component, or at least one non-generic segment."""
    if _PATH_EXT_RE.search(token):
        return False
    if "." in token:
        return False
    return all(_is_weak_term(seg) for seg in token.split("/"))


def _sentence_of(text: str, span: tuple[int, int]) -> str:
    """Return the line containing the span (truncated), or empty."""
    lines = text.splitlines()
    if not lines:
        return ""
    line_start = text.count("\n", 0, span[0])
    return lines[min(line_start, len(lines) - 1)].strip()[:200]


def _is_negated(sentence: str) -> bool:
    return bool(_NEGATION_PATTERNS.search(sentence))


# Whole-answer absence markers: an answer whose overall claim is "no such
# thing exists". CASE 4 (negative query) must not be forced into a
# citation-gated controlled response. Broadened to common phrasings the
# 7B model produces ("did not find", "does not contain", "nowhere").
_NEGATIVE_ANSWER_PATTERNS = re.compile(
    r"(?:\b(?:no|not|none|never|nothing|nowhere|without)\b|"
    r"\b(?:does|do|did|is|are|was|were|has|have|could|would|can|will)\s+not\b|"
    r"\b(?:doesn't|don't|didn't|isn't|aren't|wasn't|weren't|cannot|can't|"
    r"couldn't|won't)\b|"
    r"\bnot\s+(?:found|present|exist|appear|seem|defined|declared|supported|"
    r"matched|mentioned|located|listed|included|contain|indicate)\b|"
    r"\bno\s+(?:mention|evidence|instance|sequencer|class|file|such|trace|"
    r"occurrence|indication|sign|match|record)\b)",
    re.IGNORECASE,
)


def _is_negative_answer(answer: str) -> bool:
    return bool(_NEGATIVE_ANSWER_PATTERNS.search(answer))


def extract_entities(text: str) -> list[EntityReference]:
    """Deterministically pull source-specific references out of ``text``
    (file paths, qualified ``A::B`` symbols, dotted member paths, and
    bare identifiers), skipping tokens that live inside fenced code."""
    refs: list[EntityReference] = []
    used: set[tuple[int, int]] = set()

    # Skip everything inside [Source: ...] citations — those are already
    # validated by citation_validator.
    citation_spans = [(m.start(), m.end()) for m in _CITATION_PATTERN.finditer(text)]

    def _usable(match: re.Match) -> bool:
        span = (match.start(), match.end())
        if _overlaps(span, used):
            return False
        if _overlaps(span, citation_spans):
            return False
        return True

    def _add(kind: str, token: str, span: tuple[int, int], sentence: str) -> None:
        used.add(span)
        refs.append(EntityReference(kind=kind, token=token, sentence=sentence))

    for pattern, kind in ((_QUALIFIED_RE, "qualified"), (_DOT_RE, "dot")):
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if not _usable(m):
                continue
            token = m.group(0)
            if kind == "dot" and _dot_is_weak(token):
                continue
            _add(kind, token, span, _sentence_of(text, span))

    for m in _SLASH_PATH_RE.finditer(text):
        span = (m.start(), m.end())
        if not _usable(m):
            continue
        token = m.group(0)
        if _slash_is_weak(token):
            continue
        _add("path", token, span, _sentence_of(text, span))

    for m in _IDENT_RE.finditer(text):
        span = (m.start(), m.end())
        if not _usable(m):
            continue
        token = m.group(0)
        if not _classify_bare(token):
            continue
        _add("symbol", token, span, _sentence_of(text, span))

    return refs


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass
class GroundingReport:
    status: str = "grounded"  # grounded | insufficient_evidence | unsupported
    unsupported_entities: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    valid_citations: int = 0
    fabricated_citations: list[str] = field(default_factory=list)
    present_source_specific: list[str] = field(default_factory=list)


def classify_refs(answer: str, evidence: dict) -> list[EntityReference]:
    """Extract and classify every source-specific reference in ``answer``
    against the evidence vocabulary. Sets each reference's ``status`` to
    ``present``, ``negated``, or ``unsupported``."""
    refs = extract_entities(answer)
    for ref in refs:
        if ref.kind == "path":
            present = ref.token in evidence["paths"]
            if not present and "/" in ref.token:
                present = any(ref.token.endswith(p) for p in evidence["paths"])
        elif ref.kind in ("qualified", "dot"):
            # A qualified ``A::B`` or dotted ``a.b.c`` reference is only
            # supported when the exact token appears in the evidence text
            # (SystemVerilog always writes ``task A::B`` / ``obj.a.b`` in
            # the source). No root-only fallback: ``A::run_phase`` must
            # not pass just because ``A`` is a known class.
            present = ref.token in evidence["text"]
        else:  # symbol
            present = ref.token in evidence["symbols"] or ref.token in evidence["paths"]

        if present:
            ref.status = "present"
        elif _is_negated(ref.sentence):
            ref.status = "negated"
        else:
            ref.status = "unsupported"
    return refs


def check_grounding(
    answer: str,
    included_chunks: list[RetrievalResult],
    file_index_text: str | None = None,
    question: str | None = None,
) -> GroundingReport:
    """Run the grounding gate on a generated answer.

    Returns a ``GroundingReport`` with one of three statuses:

    - ``grounded``: every source-specific reference is backed by the
      evidence, and source-specific answers carry at least one valid
      citation.
    - ``insufficient_evidence``: the answer makes source-specific claims
      but cites nothing (nothing to verify against), or the evidence
      itself is empty for a source-specific answer.
    - ``unsupported``: the answer affirmatively references entities that
      are NOT in the retrieved evidence.
    """
    report = GroundingReport()
    if not answer:
        return report

    evidence = build_evidence(included_chunks, file_index_text)

    # Citation check — reuse the existing validator.
    citation_validation = validate_citations(answer, included_chunks)
    report.valid_citations = len(citation_validation.valid_citations)
    report.fabricated_citations = sorted(citation_validation.fabricated_citations)

    # Entity check.
    source_specific_present: list[str] = []
    unsupported: list[EntityReference] = []

    for ref in classify_refs(answer, evidence):
        if ref.status == "present":
            source_specific_present.append(ref.token)
        elif ref.status == "unsupported":
            unsupported.append(ref)

    # Fabricated citations are themselves unsupported references.
    for path in report.fabricated_citations:
        unsupported.append(EntityReference(kind="path", token=path, status="unsupported", sentence="[Source: ...] citation"))

    report.present_source_specific = sorted(set(source_specific_present))

    if unsupported:
        report.status = "unsupported"
        seen: set[str] = set()
        for ref in unsupported:
            if ref.token in seen:
                continue
            seen.add(ref.token)
            report.unsupported_entities.append(ref.token)
            if ref.sentence:
                report.unsupported_claims.append(
                    f"{ref.token} (in: {ref.sentence[:120]})"
                )
            else:
                report.unsupported_claims.append(ref.token)
        return report

    # No unsupported entities. Decide between grounded and
    # insufficient_evidence.
    if report.fabricated_citations:
        report.status = "unsupported"
        report.unsupported_entities.extend(report.fabricated_citations)
        return report

    # Source-specific claims that cite nothing are not trusted (CASE 3) —
    # unless the whole answer is a negative existence claim, which CASE 4
    # explicitly allows to pass without citations ("No, such a class does
    # not exist" is verified by the entity check itself).
    if (
        source_specific_present
        and report.valid_citations == 0
        and not _is_negative_answer(answer)
    ):
        report.status = "insufficient_evidence"
        return report

    # Empty evidence for a source-specific question is also weak.
    if not evidence["paths"] and not evidence["text"]:
        report.status = "insufficient_evidence"
        return report

    return report


# ---------------------------------------------------------------------------
# Answer policy helpers (pure; the LLM call itself lives in the service)
# ---------------------------------------------------------------------------


def split_blocks(text: str) -> list[tuple[bool, str]]:
    """Split ``text`` into ``(is_code, segment)`` blocks. Fenced code
    blocks (``` ... ```) are atomic; everything else is prose."""
    blocks: list[tuple[bool, str]] = []
    parts = _CODE_FENCE_RE.split(text)
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        blocks.append((i % 2 == 1, part))
    return blocks


def filter_supported_portion(
    answer: str,
    report: GroundingReport,
    included_chunks: list[RetrievalResult] | None = None,
    file_index_text: str | None = None,
) -> str:
    """Keep only the parts of the answer that reference no unsupported
    entity (and no fabricated citation). Code blocks are dropped whole if
    any unsupported entity appears inside them; prose is filtered by
    sentence. ``included_chunks`` may be omitted when the caller only
    needs to drop references whose tokens are listed as unsupported in
    the report."""
    unsupported_tokens = set(report.unsupported_entities)
    kept: list[str] = []

    def _unsupported_segment(segment: str) -> bool:
        if included_chunks is None:
            refs = extract_entities(segment)
            return any(r.token in unsupported_tokens for r in refs)
        evidence = build_evidence(included_chunks, file_index_text)
        return any(r.status == "unsupported" for r in classify_refs(segment, evidence))

    for is_code, segment in split_blocks(answer):
        if is_code:
            # A code block is atomic: if any fabricated symbol appears
            # inside it, drop the whole block (a half-reproduced invented
            # function is worse than none).
            if _unsupported_segment(segment):
                continue
            kept.append(segment.strip())
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(segment):
            if not sentence.strip():
                continue
            if _unsupported_segment(sentence):
                continue
            kept.append(sentence.strip())

    return "\n".join(kept)


def build_revision_messages(
    question: str,
    original_answer: str,
    report: GroundingReport,
    included_chunks: list[RetrievalResult],
    system_prompt: str,
    file_index_text: str | None = None,
) -> list[dict]:
    """Build the messages for a constrained regeneration that keeps only
    evidence-supported claims. Uses the same SYSTEM_PROMPT the original
    answer was generated under, plus the unsupported-claims list."""
    context_pieces = [format_chunk_with_citation(c) for c in included_chunks]
    if file_index_text:
        context_pieces.insert(0, f"[FILE INDEX]\n{file_index_text}")

    unsupported_desc = report.unsupported_claims or report.unsupported_entities or [
        "the answer made source-specific claims without any supporting citation"
    ]
    user = (
        "The answer below was checked against the retrieved evidence. "
        "The following claims are NOT supported by that evidence and must be removed:\n"
        + "\n".join(f"- {c}" for c in unsupported_desc)
        + "\n\nRewrite the answer using ONLY the provided CONTEXT. "
        "Remove every unsupported claim entirely — do not paraphrase, repair, or "
        "reintroduce it. Do not invent any class, method, symbol, file path, or code. "
        "If removing the unsupported claims leaves too little to answer the question, "
        "say explicitly that the retrieved evidence does not support a complete answer. "
        "Every remaining factual claim must carry its own inline [Source: path, lines X-Y] "
        "citation taken from the CONTEXT."
        + "\n\nCONTEXT:\n" + "\n\n".join(context_pieces)
        + "\n\nQUESTION:\n" + question
        + "\n\nORIGINAL ANSWER (rewrite this, removing unsupported claims):\n" + original_answer
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


__all__ = [
    "EntityReference",
    "GroundingReport",
    "build_evidence",
    "build_revision_messages",
    "check_grounding",
    "classify_refs",
    "extract_entities",
    "filter_supported_portion",
    "split_blocks",
]