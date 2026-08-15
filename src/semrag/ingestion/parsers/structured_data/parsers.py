"""
Structured data parsers: JSON, YAML, CSV, XML.

Unlike source-code parsers, these formats are already machine-structured —
there's no grammar-ambiguity problem to solve. What these parsers extract
is *shape*, not full content: top-level keys and nesting depth for
JSON/YAML (register maps, testbench config, coverage exclusion lists are
common uses in this domain), column headers and row count for CSV
(frequently register maps or test lists exported from a spreadsheet), and
tag inventory for XML (common for waveform-viewer config and some IP-XACT
style deliverables).

Full content is deliberately not embedded into the payload for large
files — a 50,000-row CSV register map's *shape* (column names) is what a
retrieval query like "what columns does the register map have" needs;
the actual row content becomes retrievable chunks in Part 8, not
duplicated here. Row/record sampling (a handful of example rows) is
included since it materially helps a retrieval system explain what a file
contains without embedding the whole thing twice.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import FileRecord

_MAX_SAMPLE_ROWS = 5
_MAX_TOP_LEVEL_KEYS = 200


class JsonYamlStructure(BaseModel):
    root_type: str  # "object" | "array" | "scalar"
    top_level_keys: list[str] = Field(default_factory=list)
    array_length: int | None = None


class CsvStructure(BaseModel):
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    sample_rows: list[dict[str, str]] = Field(default_factory=list)


class XmlStructure(BaseModel):
    root_tag: str
    distinct_tags: list[str] = Field(default_factory=list)
    max_depth: int = 0


def _read_text(absolute_path: Path) -> str | None:
    try:
        return absolute_path.read_text(errors="replace")
    except (FileNotFoundError, PermissionError):
        return None


def _fail(file_record: FileRecord, parser_name: str, message: str) -> ParseResult:
    return ParseResult(
        project_id=file_record.project_id,
        relative_path=file_record.relative_path,
        parser_name=parser_name,
        success=False,
        diagnostics=[ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message=message)],
    )


class JsonParser:
    name = "json_shape"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        text = _read_text(absolute_path)
        if text is None:
            return _fail(file_record, self.name, "Could not read file")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return _fail(file_record, self.name, f"Invalid JSON: {exc}")

        structure = _shape_of(data)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )


class YamlParser:
    name = "yaml_shape"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        text = _read_text(absolute_path)
        if text is None:
            return _fail(file_record, self.name, "Could not read file")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return _fail(file_record, self.name, f"Invalid YAML: {exc}")

        structure = _shape_of(data)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )


def _shape_of(data) -> JsonYamlStructure:
    if isinstance(data, dict):
        return JsonYamlStructure(
            root_type="object", top_level_keys=list(data.keys())[:_MAX_TOP_LEVEL_KEYS]
        )
    if isinstance(data, list):
        return JsonYamlStructure(root_type="array", array_length=len(data))
    return JsonYamlStructure(root_type="scalar")


class CsvParser:
    name = "csv_shape"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        text = _read_text(absolute_path)
        if text is None:
            return _fail(file_record, self.name, "Could not read file")
        try:
            reader = csv.DictReader(text.splitlines())
            columns = reader.fieldnames or []
            rows = list(reader)
        except csv.Error as exc:
            return _fail(file_record, self.name, f"Invalid CSV: {exc}")

        structure = CsvStructure(
            columns=list(columns),
            row_count=len(rows),
            sample_rows=[dict(r) for r in rows[:_MAX_SAMPLE_ROWS]],
        )
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )


class XmlParser:
    name = "xml_shape"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        try:
            tree = ET.parse(absolute_path)
        except (ET.ParseError, FileNotFoundError, PermissionError) as exc:
            return _fail(file_record, self.name, f"Invalid XML: {exc}")

        root = tree.getroot()
        tags: set[str] = set()
        max_depth = 0

        def walk(node, depth):
            nonlocal max_depth
            tags.add(node.tag)
            max_depth = max(max_depth, depth)
            for child in node:
                walk(child, depth + 1)

        walk(root, 1)

        structure = XmlStructure(root_tag=root.tag, distinct_tags=sorted(tags), max_depth=max_depth)
        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=structure.model_dump(),
        )
