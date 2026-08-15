"""
End-to-end convenience wiring: scan -> parse every scannable file -> build
IR. This is what Part 2's future Celery worker entrypoint calls; exposed
here as a plain function so it's testable without touching Celery at all.

Deliberately re-parses every file every call rather than only the
`ChangeSet` — see builder.py's docstring for why: incremental *parsing*
(only re-parsing changed files and reusing cached ParseResults for the
rest) is a caching concern for the worker layer, not something this
convenience function or the builder itself need to solve to be correct.
This function is the simple, obviously-correct baseline; an incremental
variant is listed in this part's future-work section.
"""
from __future__ import annotations

from pathlib import Path

from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.ir.models import IRGraph
from semrag.ingestion.parsers.interfaces import ParserRegistry
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.interfaces import ManifestStore
from semrag.ingestion.scanner.models import ScanManifest
from semrag.ingestion.scanner.orchestrator import Scanner


def scan_parse_and_build_ir(
    project_id: str,
    root_path: str,
    manifest_store: ManifestStore,
    registry: ParserRegistry | None = None,
) -> tuple[ScanManifest, IRGraph]:
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _change_set = scanner.scan(project_id, root_path)

    registry = registry or build_default_registry()
    root = Path(root_path).resolve()

    parse_results: dict[str, ParseResult] = {}
    for relative_path, record in manifest.files.items():
        absolute_path = root / relative_path
        result = registry.parse_file(absolute_path, record)
        if result is not None:
            parse_results[relative_path] = result

    graph = IRBuilder().build(project_id, manifest.files, parse_results)
    return manifest, graph
