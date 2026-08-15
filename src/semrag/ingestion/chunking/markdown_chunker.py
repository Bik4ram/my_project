"""
Markdown chunker.

Unlike the SystemVerilog/Python chunkers, this one never receives IR
nodes to work from — Part 4's IR builder only creates nodes for the SV/
Python/TCL categories (see Part 4's design doc, Section 1), so `ir_nodes`
is always empty here. This chunker instead reads directly from Part 3's
`MarkdownFileStructure` payload (`headings`), which is exactly the
information needed: chunk boundaries are section boundaries.

**Boundary rule, and why it isn't "same-or-shallower level":** the first
version of this chunker ended a section at the next heading of the same
or shallower level — standard "table of contents" nesting semantics. That
turned out to be the wrong rule for chunking, caught by a test: a
document's single H1 title has no sibling H1 to stop it, so its "section"
extended to the end of the file and its chunk silently duplicated the
entire rest of the document — every H2/H3 subsection's content, again,
inside the H1 chunk. That's a real content-duplication bug, not just a
test artifact; every retrieval hit under that H1 would come back
redundant with a more specific subsection hit. The fix: a section's
content is everything up to the **next heading of ANY level** — each
chunk holds only its own leaf content, never a child section's content
too. Hierarchy context (this is a subsection of X, which is a subsection
of Y) is preserved instead as a breadcrumb line prefixed onto the chunk's
text, not by duplicating the parent's prose — cheap, keeps each chunk
self-contained for citation, and avoids the duplication the level-based
rule produced.

Any content before the first heading (a preamble/title block) becomes its
own chunk rather than being silently dropped.
"""
from __future__ import annotations

from semrag.ingestion.chunking.id_scheme import chunk_id, extract_lines
from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.ir.models import IRNode
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileRecord


class MarkdownChunker:
    chunker_name = "markdown"

    def chunk(
        self,
        file_record: FileRecord,
        parse_result: ParseResult | None,
        file_text: str,
        ir_nodes: list[IRNode],
    ) -> list[Chunk]:
        if parse_result is None or not parse_result.success:
            return []

        headings = parse_result.payload.get("headings", [])
        total_lines = len(file_text.splitlines())
        if not headings:
            # No headings at all — whole file is one chunk rather than
            # producing nothing for a plain-prose markdown file.
            if not file_text.strip():
                return []
            return [
                Chunk(
                    chunk_id=chunk_id(file_record.project_id, file_record.relative_path, "markdown_section", "0"),
                    project_id=file_record.project_id,
                    relative_path=file_record.relative_path,
                    chunk_type="markdown_section",
                    source_node_id=None,
                    text=file_text,
                    line_start=1,
                    line_end=total_lines,
                )
            ]

        chunks: list[Chunk] = []

        if headings[0]["line"] > 1:
            preamble = extract_lines(file_text, 1, headings[0]["line"] - 1)
            if preamble.strip():
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(file_record.project_id, file_record.relative_path, "markdown_preamble", "0"),
                        project_id=file_record.project_id,
                        relative_path=file_record.relative_path,
                        chunk_type="markdown_preamble",
                        source_node_id=None,
                        text=preamble,
                        line_start=1,
                        line_end=headings[0]["line"] - 1,
                    )
                )

        breadcrumb_stack: list[tuple[int, str]] = []  # (level, text)

        for i, heading in enumerate(headings):
            # Leaf content only — bounded by the very next heading,
            # regardless of its level. See module docstring for why.
            section_end = (headings[i + 1]["line"] - 1) if i + 1 < len(headings) else total_lines
            section_start = heading["line"]
            body = extract_lines(file_text, section_start, section_end)
            if not body.strip():
                continue

            while breadcrumb_stack and breadcrumb_stack[-1][0] >= heading["level"]:
                breadcrumb_stack.pop()
            breadcrumb_stack.append((heading["level"], heading["text"]))

            breadcrumb = " > ".join(text for _, text in breadcrumb_stack)
            text = f"{breadcrumb}\n\n{body}" if len(breadcrumb_stack) > 1 else body

            chunks.append(
                Chunk(
                    chunk_id=chunk_id(
                        file_record.project_id, file_record.relative_path, "markdown_section", str(i)
                    ),
                    project_id=file_record.project_id,
                    relative_path=file_record.relative_path,
                    chunk_type="markdown_section",
                    source_node_id=None,
                    text=text,
                    line_start=section_start,
                    line_end=section_end,
                )
            )

        return chunks
