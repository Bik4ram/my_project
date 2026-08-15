"""
Intermediate Representation (IR) data models.

Scope of this IR, stated precisely because "IR" is an overloaded word and
Part 1's file-type list (Module, Signal, Register, FSM, Covergroup, Spec
Section...) is broader than what this module builds: this is a
**structural linking IR** — it takes the per-file facts Part 3's parsers
already extracted (module names, ports, parameters, assertions, imports)
and links them into a cross-file graph wherever that linking is resolvable
from names alone (module instantiation -> module definition, `include ->
file, import -> module/package).

It deliberately does NOT attempt **semantic** linking that requires
understanding what an assertion's expression actually references (e.g.
"this assertion covers register X" requires signal-level analysis: is
`scl` in `start_condition |-> scl == 1'b0` the same `scl` declared as a
port of some other module three files away, or a local signal). That is a
distinct, harder analysis pass — likely layered on top of this IR rather
than folded into it — and is explicitly out of scope here. See the design
doc's Section on scope for why drawing the line here, rather than
attempting a shallow version of semantic linking, was the right call.

Node/edge identity is content-derived (a deterministic function of
project_id + relative_path + name/kind), not an auto-increment ID, so that
re-parsing an unchanged file produces byte-identical node IDs across runs
— required for Part 6's storage layer to do upserts instead of
duplicate-inserts on every re-index.
"""
from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class IRNodeKind(str, enum.Enum):
    SV_FILE = "sv_file"          # any .sv/.svh file, whether or not it defines a module
    SV_MODULE = "sv_module"
    SV_INTERFACE = "sv_interface"
    SV_PACKAGE = "sv_package"
    SV_CLASS = "sv_class"        # a SystemVerilog class (e.g. a UVM component/sequence)
    SV_METHOD = "sv_method"      # a function/task inside an SV class
    PY_MODULE = "py_module"      # one node per .py file
    TCL_SCRIPT = "tcl_script"    # one node per .tcl file


class IREdgeKind(str, enum.Enum):
    INSTANTIATES = "instantiates"       # SV_MODULE -> SV_MODULE
    INCLUDES = "includes"               # SV_FILE -> SV_FILE
    IMPORTS_PACKAGE = "imports_package"  # SV_FILE -> SV_PACKAGE
    EXTENDS = "extends"                 # SV_CLASS -> SV_CLASS (class inheritance)
    REFERENCES = "references"           # SV_CLASS -> SV_CLASS (project-local type/class reference)
    CALLS = "calls"                     # SV_METHOD -> SV_METHOD (project-local method call)
    PY_IMPORTS = "py_imports"           # PY_MODULE -> PY_MODULE
    TCL_SOURCES = "tcl_sources"         # TCL_SCRIPT -> TCL_SCRIPT


class IRNode(BaseModel):
    id: str
    kind: IRNodeKind
    project_id: str
    relative_path: str
    name: str = Field(..., description="Module/package/file-stem name; not necessarily unique project-wide.")
    parent_file_id: str | None = Field(
        default=None,
        description="For SV_MODULE/SV_INTERFACE/SV_PACKAGE nodes, the SV_FILE node they were defined in.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The parser payload subset relevant to this node (ports, parameters, "
            "assertions, functions, classes, procs, etc.) carried through unchanged "
            "from Part 3's ParseResult.payload, not re-derived here."
        ),
    )


class IREdge(BaseModel):
    id: str
    kind: IREdgeKind
    source_id: str
    target_name: str = Field(..., description="The raw referenced name/path before resolution.")
    target_id: str | None = Field(default=None, description="Resolved node id, or None if unresolved.")
    resolved: bool = False
    ambiguous: bool = False
    candidate_ids: list[str] = Field(
        default_factory=list,
        description="Populated only when ambiguous=True — more than one node matched target_name.",
    )


class IRGraph(BaseModel):
    project_id: str
    nodes: dict[str, IRNode] = Field(default_factory=dict)
    edges: list[IREdge] = Field(default_factory=list)

    def nodes_of_kind(self, kind: IRNodeKind) -> list[IRNode]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def edges_from(self, node_id: str) -> list[IREdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def unresolved_edges(self) -> list[IREdge]:
        return [e for e in self.edges if not e.resolved]
