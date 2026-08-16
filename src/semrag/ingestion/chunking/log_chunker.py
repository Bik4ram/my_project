"""
Event-aware log chunker for ``FileCategory.LOG``.

Why this exists
---------------

The generic sliding-window chunker slices a log into fixed 60-line windows,
so a multi-line event (a UVM check message followed by its packet table and
its PASS/FAIL verdict, or an ``ERROR`` line followed by its property
continuation) can be split across two windows and diluted by unrelated
repetitive lines. In the repository's real corpus
(``sim/log/write1_test_log.log``, a QuestaSim UVM-1.0p1 regression run)
roughly 41% of the file is a single repeated ``[STATE_COV_ITEM]`` message;
the meaningful content is the ``[X_MATCH_ITEM]`` message + packet table +
``[X_PASS]`` verdict groups. That split-vs-noise problem is what this
chunker addresses.

Strategy
--------

1. **Event boundaries** — a line starts a new event when it carries a
   semantic marker, derived from the existing codebase and the real corpus:

   - ``UVM_INFO`` / ``UVM_WARNING`` / ``UVM_ERROR`` / ``UVM_FATAL`` — the
      exact severity vocabulary the regression-log parser
      (``parsers/logs/regression_log_parser.py``) already recognizes, and
      every meaningful line of the corpus. UVM's per-macro occurrence tail
      (``UVM_ERROR at time 500 ns : 1 occurrences``) is a continuation,
      not a boundary — it carries no message body and is the closing line
      of the preceding event.
   - generic severity / result markers at the start of a line:
     ``ERROR``, ``FATAL``, ``EXCEPTION``, ``WARNING``, ``WARN``,
     ``CRITICAL``, ``ASSERTION FAILURE``/``assertion failed``,
     simulation/test start/end, and coverage/report summary lines. These
     cover logs that don't use the ``UVM_`` prefix.
   - A line is deliberately **not** a boundary just because it contains
     ``PASS``/``FAIL`` or ``INFO`` — those are the ordinary/repetitive
     lines the task says to group, not split on.

2. **Continuation retention** — every line after a boundary, until the next
   boundary, belongs to that event: the packet tables and ``Name/Type/
   Size/Value`` rows, indented continuation lines, separator rows, and the
   ``UVM Report Summary`` counts. This is what keeps a multi-line event
   (an ``ERROR`` line + its ``property/expected/actual`` block, or a check
   message + its table) together in one chunk.

3. **Verdict absorption** — a single-line event immediately following a
   multi-line event is absorbed into it when it carries a verdict marker
   (``PASS(ED)``/``FAIL(ED)``/``ERROR``/``FATAL``). In the corpus every
   ``[X_MATCH_ITEM]`` table is concluded by its ``[X_PASS]`` line; keeping
   them together preserves the complete check as one retrievable unit.

4. **Repetitive-run grouping** — consecutive single-line events (the
   ``[STATE_COV_ITEM]`` noise) are grouped into contiguous windows of up to
   ``window_lines`` lines. This turns ~1600 repetitive lines into a bounded
   number of chunks instead of one chunk per line, without deleting any
   content — every line survives in some chunk.

5. **Bounded chunks** — a single event ≤ ``window_lines`` lines becomes one
   chunk. A larger event (a giant table, a long stack trace) is split with
   the same overlapping windowing as ``GenericWindowChunker``, reusing the
   existing chunk-size convention (``window_lines``/``overlap_lines``).
   A log with no recognizable markers degrades to the same windowing
   behaviour as the generic chunker.

Metadata contract
-----------------

Same as ``GenericWindowChunker``: ``chunk_type="log"`` (the reserved
vocabulary ``services._NOISE_CHUNK_TYPES`` and
``evaluation.retrieval_analytics.categorize_document`` already use),
``source_node_id=None`` (the IR builder creates no nodes for logs),
deterministic content-derived ``chunk_id`` with a sequential
disambiguator, and 1-indexed inclusive ``line_start``/``line_end``.
"""
from __future__ import annotations

import re

from semrag.ingestion.chunking.generic_window_chunker import (
    DEFAULT_OVERLAP_LINES,
    DEFAULT_WINDOW_LINES,
)
from semrag.ingestion.chunking.id_scheme import chunk_id
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileRecord

_UVM_SEVERITY = re.compile(r"\bUVM_(INFO|WARNING|ERROR|FATAL)\b", re.IGNORECASE)

# UVM's reporter prints a per-macro occurrence tail after the message block
# ("UVM_ERROR at time 500 ns : 1 occurrences"). It carries no message body
# and is the tail of the preceding event, so it must NOT re-trigger a
# boundary — otherwise a FATAL stack trace is split by its own summary line.
_UVM_OCCURRENCE_TAIL = re.compile(
    r"\bUVM_(?:INFO|WARNING|ERROR|FATAL)\s+at time\b", re.IGNORECASE
)

_GENERIC_BOUNDARY = re.compile(
    r"^\s*#?\s*(?:"
    r"ERROR|FATAL|EXCEPTION|WARNING|WARN|CRITICAL"
    r"|ASSERTION\s+(?:FAILURE|FAILED|FAIL)"
    r"|simulation\s+(?:started|start|ended|end|complete)"
    r"|test\s+(?:started|start|ended|end|complete|done)"
    r"|coverage\s+summary"
    r"|report\s+(?:summary|counts)"
    r")\b",
    re.IGNORECASE,
)

_SUMMARY_MARKER = re.compile(
    r"(?:UVM\s+Report\s+Summary|Report\s+counts\s+by\s+severity)",
    re.IGNORECASE,
)

# Verdict markers: PASS/FAIL (with ED suffix) plus ERROR/FATAL, bounded by
# non-alphanumeric characters so bracketed tags (``[ADDR_PASS]``,
# ``[X_FAIL]``) and bare tokens match, while embedded uses
# (``PASSAGE``) do not. `\b` alone would miss ``ADDR_PASS`` because `_` is
# a word character and there is no word boundary before ``PASS``.
_VERDICT = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:PASS(?:ED)?|FAIL(?:ED)?|ERROR|FATAL)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def _is_event_start(line: str) -> bool:
    """True when a line starts a new log event (UVM severity line anywhere,
    generic severity/result marker at line start). UVM occurrence-tail lines
    (``UVM_ERROR at time ...``) are continuations, not boundaries."""
    stripped = line.strip()
    if _UVM_SEVERITY.search(stripped) and not _UVM_OCCURRENCE_TAIL.search(stripped):
        return True
    return bool(_GENERIC_BOUNDARY.match(stripped))


def _build_regions(lines: list[str]) -> list[list[int]]:
    """Split into chronological regions. Each region starts with a boundary
    line (an event start or the report-summary marker) except the optional
    preamble; everything after a summary marker is swallowed to EOF so the
    count lines (``UVM_INFO : 1885``) don't re-trigger splitting."""
    regions: list[list[int]] = []
    current: list[int] = []
    summary_mode = False
    for i, line in enumerate(lines):
        if summary_mode:
            current.append(i)
            continue
        if _SUMMARY_MARKER.search(line):
            summary_mode = True
            if current:
                regions.append(current)
            current = [i]
            continue
        if _is_event_start(line):
            if current:
                regions.append(current)
            current = [i]
            continue
        current.append(i)
    if current:
        regions.append(current)
    return regions


def _absorb_verdicts(regions: list[list[int]], lines: list[str]) -> list[list[int]]:
    """Absorb one trailing single-line verdict event into the preceding
    multi-line event, keeping a check event and its PASS/FAIL conclusion in
    the same chunk. Only genuine events (not the preamble) absorb, and only
    one line, so this never chains."""
    result: list[list[int]] = []
    i = 0
    while i < len(regions):
        region = regions[i]
        is_multi_event = len(region) > 1 and _is_event_start(lines[region[0]])
        next_is_verdict = (
            i + 1 < len(regions)
            and len(regions[i + 1]) == 1
            and bool(_VERDICT.search(lines[regions[i + 1][0]]))
        )
        if is_multi_event and next_is_verdict:
            result.append(region + regions[i + 1])
            i += 2
        else:
            result.append(region)
            i += 1
    return result


def _group_singles(regions: list[list[int]], window_lines: int) -> list[list[int]]:
    """Merge consecutive single-line event regions into contiguous windows
    of up to ``window_lines`` lines. Multi-line events break the run so a
    table-owning event is never merged into the surrounding noise."""
    result: list[list[int]] = []
    run: list[int] = []
    for region in regions:
        if len(region) == 1:
            run.extend(region)
            while len(run) >= window_lines:
                result.append(run[:window_lines])
                run = run[window_lines:]
        else:
            if run:
                result.append(run)
                run = []
            result.append(region)
    if run:
        result.append(run)
    return result


def _split_region(region: list[int], window_lines: int, overlap_lines: int) -> list[list[int]]:
    """Overlapping-window split for a single oversized event region, mirroring
    ``GenericWindowChunker`` so very long events stay bounded."""
    step = window_lines - overlap_lines
    out: list[list[int]] = []
    start = 0
    n = len(region)
    while start < n:
        end = min(start + window_lines, n)
        out.append(region[start:end])
        if end == n:
            break
        start += step
    return out


class LogChunker:
    """Event-aware chunker for ``FileCategory.LOG`` (see module docstring)."""

    chunker_name = "log"

    def __init__(self, window_lines: int = DEFAULT_WINDOW_LINES, overlap_lines: int = DEFAULT_OVERLAP_LINES):
        if overlap_lines >= window_lines:
            raise ValueError("overlap_lines must be smaller than window_lines")
        self._window = window_lines
        self._overlap = overlap_lines

    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        del parse_result, ir_nodes  # logs carry no IR nodes; parser payload is metadata-only

        lines = file_text.splitlines()
        if not lines:
            return []

        regions = _group_singles(
            _absorb_verdicts(_build_regions(lines), lines),
            self._window,
        )

        chunks: list[Chunk] = []
        index = 0
        for region in regions:
            regions_out = (
                [region]
                if len(region) <= self._window
                else _split_region(region, self._window, self._overlap)
            )
            for part in regions_out:
                text = "\n".join(lines[i] for i in part)
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(
                            file_record.project_id,
                            file_record.relative_path,
                            "log",
                            str(index),
                        ),
                        project_id=file_record.project_id,
                        relative_path=file_record.relative_path,
                        chunk_type="log",
                        source_node_id=None,
                        text=text,
                        line_start=part[0] + 1,
                        line_end=part[-1] + 1,
                    )
                )
                index += 1

        return chunks