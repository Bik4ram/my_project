"""
Markdown parser.

Extracts heading hierarchy (ATX `#`-style headings, the overwhelming
majority in engineering docs/READMEs) and the line range each section
spans, using stdlib regex rather than a full CommonMark implementation.
This is deliberately structure-only — no rendering, no link resolution —
because what Part 8 (chunking) actually needs from Markdown is section
boundaries to chunk along, not a faithful document render. Setext-style
headings (`===`/`---` underlines) are not handled; flagged as a gap since
they're occasionally used, notably for H1/H2 in older READMEs.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import ParseDiagnostic, DiagnosticSeverity, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_CODE_FENCE_PATTERN = re.compile(r"^```(\w*)\s*$", re.MULTILINE)


class ParsedHeading(BaseModel):
    level: int
    text: str
    line: int


class MarkdownFileStructure(BaseModel):
    headings: list[ParsedHeading] = Field(default_factory=list)
    code_block_languages: list[str] = Field(
        default_factory=list,
        description="Language tag of each fenced code block, in document order ('' if untagged).",
    )


class MarkdownParser:
    name = "markdown_regex_structural"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        try:
            text = absolute_path.read_text(errors="replace")
        except (FileNotFoundError, PermissionError) as exc:
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=f"Could not read file: {exc}")
                ],
            )

        headings = [
            ParsedHeading(
                level=len(m.group(1)),
                text=m.group(2).strip(),
                line=text.count("\n", 0, m.start()) + 1,
            )
            for m in _HEADING_PATTERN.finditer(text)
        ]
        code_langs = [m.group(1) for m in _CODE_FENCE_PATTERN.finditer(text)]
        # Fences come in open/close pairs; only every other match is an
        # "opening" fence with a meaningful language tag.
        code_langs = code_langs[::2]

        structure = MarkdownFileStructure(headings=headings, code_block_languages=code_langs)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )
