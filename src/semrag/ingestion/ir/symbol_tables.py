"""
Project-wide symbol tables.

Built in one pass over every node before any edge resolution happens —
edges can reference nodes defined anywhere in the project, so the full
symbol table must exist before resolution starts (this is the one place
in the ingestion pipeline that genuinely needs whole-project state, not
just the changed files; see the design doc's incremental-update section
for how this is reconciled with Part 2's incremental philosophy).

Every index maps a lookup key to a **list** of candidate node ids, not a
single id — name collisions are a real possibility in a large multi-repo
system (two modules named `fifo` in different subsystems) and must be
surfaced as `ambiguous`, not silently resolved to whichever happened to be
indexed first.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from semrag.ingestion.ir.models import IRNode, IRNodeKind


class SymbolTables:
    def __init__(self) -> None:
        self.sv_file_by_basename: dict[str, list[str]] = defaultdict(list)
        self.sv_module_by_name: dict[str, list[str]] = defaultdict(list)
        self.sv_interface_by_name: dict[str, list[str]] = defaultdict(list)
        self.sv_package_by_name: dict[str, list[str]] = defaultdict(list)
        self.sv_class_by_name: dict[str, list[str]] = defaultdict(list)
        self.sv_method_by_name: dict[str, list[str]] = defaultdict(list)
        self.tcl_by_basename: dict[str, list[str]] = defaultdict(list)
        # Python: indexed by every dotted-path *suffix*, so both a fully
        # qualified import and a shorter relative-feeling one can resolve.
        # See design doc for why this is a heuristic, not exact.
        self.py_module_by_dotted_suffix: dict[str, list[str]] = defaultdict(list)

    def index_node(self, node: IRNode) -> None:
        basename = PurePosixPath(node.relative_path).name
        if node.kind == IRNodeKind.SV_FILE:
            self.sv_file_by_basename[basename].append(node.id)
        elif node.kind == IRNodeKind.SV_MODULE or node.kind == IRNodeKind.SV_INTERFACE:
            self.sv_module_by_name[node.name].append(node.id)
            if node.kind == IRNodeKind.SV_INTERFACE:
                self.sv_interface_by_name[node.name].append(node.id)
        elif node.kind == IRNodeKind.SV_PACKAGE:
            self.sv_package_by_name[node.name].append(node.id)
        elif node.kind == IRNodeKind.SV_CLASS:
            self.sv_class_by_name[node.name].append(node.id)
        elif node.kind == IRNodeKind.SV_METHOD:
            self.sv_method_by_name[node.name].append(node.id)
        elif node.kind == IRNodeKind.TCL_SCRIPT:
            self.tcl_by_basename[basename].append(node.id)
        elif node.kind == IRNodeKind.PY_MODULE:
            for suffix in _dotted_path_suffixes(node.relative_path):
                self.py_module_by_dotted_suffix[suffix].append(node.id)

    def resolve(self, index: dict[str, list[str]], key: str) -> tuple[str | None, bool, list[str]]:
        """Returns (resolved_id, ambiguous, candidates). resolved_id is
        None if there were zero or more-than-one candidates; ambiguous is
        True only in the more-than-one case (zero matches is simply
        unresolved, not ambiguous)."""
        candidates = index.get(key, [])
        if len(candidates) == 1:
            return candidates[0], False, candidates
        if len(candidates) > 1:
            return None, True, candidates
        return None, False, []


def _dotted_path_suffixes(relative_path: str) -> list[str]:
    """`src/semrag/ingestion/scanner/hashing.py` ->
    ['semrag.ingestion.scanner.hashing', 'ingestion.scanner.hashing',
     'scanner.hashing', 'hashing']

    The leading `src/` (or any single conventional package-root directory
    literally named `src`) is stripped first since Python import
    statements reference the package name, not the repository layout
    directory — this is a heuristic tied to the "src layout" convention
    used in Part 1's directory structure, not a general solution to
    Python package-root discovery (which would require reading
    pyproject.toml / setup.cfg, out of scope for this structural pass).
    """
    path = PurePosixPath(relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return []
    return [".".join(parts[i:]) for i in range(len(parts))]
