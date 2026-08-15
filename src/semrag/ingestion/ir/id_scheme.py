"""
Deterministic ID generation.

Every ID is a pure function of stable inputs (project_id, relative_path,
name/kind) — never a random UUID or auto-increment counter — specifically
so that re-running the IR builder on an unchanged file produces the exact
same node ID every time. Part 6's storage layer relies on this for
upsert-not-duplicate semantics on incremental re-indexing (only files in a
`ChangeSet` get re-processed; their node IDs must land on the same rows,
not create new ones alongside stale copies).

Edge IDs additionally hash in `target_name` so that if a file's raw
dependency refs change (an `include added or removed), the edge set
changes deterministically too, without colliding with edges from an
unrelated ref sharing the same source node.
"""
from __future__ import annotations

import hashlib


def _slug(*parts: str) -> str:
    return "::".join(parts)


def sv_file_id(project_id: str, relative_path: str) -> str:
    return _slug(project_id, relative_path, "sv_file")


def sv_module_id(project_id: str, relative_path: str, module_name: str) -> str:
    return _slug(project_id, relative_path, "sv_module", module_name)


def sv_package_id(project_id: str, relative_path: str, package_name: str) -> str:
    return _slug(project_id, relative_path, "sv_package", package_name)


def sv_class_id(project_id: str, relative_path: str, class_name: str) -> str:
    return _slug(project_id, relative_path, "sv_class", class_name)


def sv_method_id(project_id: str, relative_path: str, class_name: str, method_name: str) -> str:
    return _slug(project_id, relative_path, "sv_method", class_name, method_name)


def py_module_id(project_id: str, relative_path: str) -> str:
    return _slug(project_id, relative_path, "py_module")


def tcl_script_id(project_id: str, relative_path: str) -> str:
    return _slug(project_id, relative_path, "tcl_script")


def edge_id(kind: str, source_id: str, target_name: str) -> str:
    digest = hashlib.blake2b(f"{source_id}->{target_name}".encode(), digest_size=8).hexdigest()
    return _slug(kind, digest)
