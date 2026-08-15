"""
IR builder.

Two-pass algorithm, and it must stay two-pass: Pass 1 creates every node
and populates the symbol tables; Pass 2 resolves every edge against the
now-complete symbol tables. A single-pass approach would make edge
resolution order-dependent on file processing order (a module instantiated
before its definition is processed would incorrectly resolve as
unresolved) — this is the same class of bug as a forward-reference problem
in a one-pass compiler, and the fix is the same: build the full symbol
table before resolving anything against it.

Input is deliberately the *whole project's* parse results, not just a
`ChangeSet` from Part 2 — explained in the design doc's incremental-update
section: this builder computes a full IRGraph from complete inputs every
time it runs, but Part 4's caller is responsible for deciding how much of
that work was actually necessary (e.g. only re-running Part 3's parsers
on changed files, then re-running this builder's cheap in-memory
graph-construction pass over the full — mostly cached — ParseResult set).
The graph construction itself, given parsed data already in memory, is
fast; it's the parsing that's expensive and that's what Part 2's
ChangeSet already protects.
"""
from __future__ import annotations

from semrag.ingestion.ir.id_scheme import (
    edge_id,
    py_module_id,
    sv_class_id,
    sv_file_id,
    sv_method_id,
    sv_module_id,
    sv_package_id,
    tcl_script_id,
)
from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRGraph, IRNode, IRNodeKind
from semrag.ingestion.ir.symbol_tables import SymbolTables
from semrag.ingestion.parsers.models import ParseResult
from semrag.ingestion.scanner.models import FileCategory, FileRecord

_SV_FILE_EXTENSIONS = {".sv", ".svh", ".v", ".vh"}


class IRBuilder:
    def build(
        self,
        project_id: str,
        file_records: dict[str, FileRecord],
        parse_results: dict[str, ParseResult],
    ) -> IRGraph:
        graph = IRGraph(project_id=project_id)
        symbols = SymbolTables()

        # --- Pass 1: nodes ---
        for relative_path, record in file_records.items():
            result = parse_results.get(relative_path)
            self._create_nodes_for_file(graph, symbols, project_id, record, result)

        # --- Pass 2: edges ---
        for relative_path, record in file_records.items():
            result = parse_results.get(relative_path)
            self._create_edges_for_file(graph, symbols, project_id, record, result)

        return graph

    # ------------------------------------------------------------------
    # Pass 1
    # ------------------------------------------------------------------

    def _create_nodes_for_file(
        self,
        graph: IRGraph,
        symbols: SymbolTables,
        project_id: str,
        record: FileRecord,
        result: ParseResult | None,
    ) -> None:
        if record.category in (
            FileCategory.RTL_VERILOG,
            FileCategory.TESTBENCH_UVM,
            FileCategory.ASSERTION,
        ):
            self._create_sv_nodes(graph, symbols, project_id, record, result)
        elif record.category == FileCategory.PYTHON:
            self._create_py_node(graph, symbols, project_id, record, result)
        elif record.category == FileCategory.TCL:
            self._create_tcl_node(graph, symbols, project_id, record, result)

    def _create_sv_nodes(self, graph, symbols, project_id, record, result) -> None:
        file_node = IRNode(
            id=sv_file_id(project_id, record.relative_path),
            kind=IRNodeKind.SV_FILE,
            project_id=project_id,
            relative_path=record.relative_path,
            name=record.relative_path.rsplit("/", 1)[-1],
        )
        graph.nodes[file_node.id] = file_node
        symbols.index_node(file_node)

        if result is None or not result.success:
            return

        for module_payload in result.payload.get("modules", []):
            kind = IRNodeKind.SV_INTERFACE if module_payload.get("is_interface") else IRNodeKind.SV_MODULE
            node = IRNode(
                id=sv_module_id(project_id, record.relative_path, module_payload["name"]),
                kind=kind,
                project_id=project_id,
                relative_path=record.relative_path,
                name=module_payload["name"],
                parent_file_id=file_node.id,
                attributes=module_payload,
            )
            graph.nodes[node.id] = node
            symbols.index_node(node)

        for package_payload in result.payload.get("packages", []):
            node = IRNode(
                id=sv_package_id(project_id, record.relative_path, package_payload["name"]),
                kind=IRNodeKind.SV_PACKAGE,
                project_id=project_id,
                relative_path=record.relative_path,
                name=package_payload["name"],
                parent_file_id=file_node.id,
                attributes=package_payload,
            )
            graph.nodes[node.id] = node
            symbols.index_node(node)

        for class_payload in result.payload.get("classes", []):
            class_node = IRNode(
                id=sv_class_id(project_id, record.relative_path, class_payload["name"]),
                kind=IRNodeKind.SV_CLASS,
                project_id=project_id,
                relative_path=record.relative_path,
                name=class_payload["name"],
                parent_file_id=file_node.id,
                attributes=class_payload,
            )
            graph.nodes[class_node.id] = class_node
            symbols.index_node(class_node)

            for method_payload in class_payload.get("methods", []):
                method_node = IRNode(
                    id=sv_method_id(
                        project_id,
                        record.relative_path,
                        class_payload["name"],
                        method_payload["name"],
                    ),
                    kind=IRNodeKind.SV_METHOD,
                    project_id=project_id,
                    relative_path=record.relative_path,
                    name=method_payload["name"],
                    parent_file_id=class_node.id,
                    attributes=method_payload,
                )
                graph.nodes[method_node.id] = method_node
                symbols.index_node(method_node)

    def _create_py_node(self, graph, symbols, project_id, record, result) -> None:
        node = IRNode(
            id=py_module_id(project_id, record.relative_path),
            kind=IRNodeKind.PY_MODULE,
            project_id=project_id,
            relative_path=record.relative_path,
            name=record.relative_path.rsplit("/", 1)[-1].removesuffix(".py"),
            attributes=(result.payload if result and result.success else {}),
        )
        graph.nodes[node.id] = node
        symbols.index_node(node)

    def _create_tcl_node(self, graph, symbols, project_id, record, result) -> None:
        node = IRNode(
            id=tcl_script_id(project_id, record.relative_path),
            kind=IRNodeKind.TCL_SCRIPT,
            project_id=project_id,
            relative_path=record.relative_path,
            name=record.relative_path.rsplit("/", 1)[-1],
            attributes=(result.payload if result and result.success else {}),
        )
        graph.nodes[node.id] = node
        symbols.index_node(node)

    # ------------------------------------------------------------------
    # Pass 2
    # ------------------------------------------------------------------

    def _create_edges_for_file(
        self,
        graph: IRGraph,
        symbols: SymbolTables,
        project_id: str,
        record: FileRecord,
        result: ParseResult | None,
    ) -> None:
        if record.category in (
            FileCategory.RTL_VERILOG,
            FileCategory.TESTBENCH_UVM,
            FileCategory.ASSERTION,
        ):
            self._create_sv_edges(graph, symbols, project_id, record, result)
        elif record.category == FileCategory.PYTHON:
            self._create_py_edges(graph, symbols, project_id, record, result)
        elif record.category == FileCategory.TCL:
            self._create_tcl_edges(graph, symbols, project_id, record, result)

    def _create_sv_edges(self, graph, symbols, project_id, record, result) -> None:
        file_node_id = sv_file_id(project_id, record.relative_path)

        # `include vs `import disambiguation: Part 2's dependency_tracker
        # captures both into the same raw_dependency_refs list. An include
        # target is a filename (has a known SV file extension); a package
        # import target is a bare identifier. This split is possible
        # precisely because of how those two regexes are written (see
        # scanner/dependency_tracker.py) — an include ref always includes
        # the extension, an import ref never does.
        for ref in record.raw_dependency_refs:
            if any(ref.endswith(ext) for ext in _SV_FILE_EXTENSIONS):
                self._add_edge(
                    graph, symbols, symbols.sv_file_by_basename, IREdgeKind.INCLUDES,
                    file_node_id, ref.rsplit("/", 1)[-1],
                )
            else:
                self._add_edge(
                    graph, symbols, symbols.sv_package_by_name, IREdgeKind.IMPORTS_PACKAGE,
                    file_node_id, ref,
                )

        if result is None or not result.success:
            return

        for module_payload in result.payload.get("modules", []):
            module_id = sv_module_id(project_id, record.relative_path, module_payload["name"])
            for inst in module_payload.get("instantiations", []):
                self._add_edge(
                    graph, symbols, symbols.sv_module_by_name, IREdgeKind.INSTANTIATES,
                    module_id, inst["module_type"],
                )

        for class_payload in result.payload.get("classes", []):
            class_id = sv_class_id(project_id, record.relative_path, class_payload["name"])
            base_class = class_payload.get("extends")
            if base_class:
                self._add_edge(
                    graph, symbols, symbols.sv_class_by_name, IREdgeKind.EXTENDS,
                    class_id, base_class,
                )
            for referenced in class_payload.get("type_refs", []):
                self._add_edge(
                    graph, symbols, symbols.sv_class_by_name, IREdgeKind.REFERENCES,
                    class_id, referenced,
                )

            # R2E-D: virtual-interface members (``virtual i2c_master_if vif;``)
            # reference the interface node by name.
            for interface_name in class_payload.get("interface_refs", []):
                self._add_edge(
                    graph, symbols, symbols.sv_interface_by_name, IREdgeKind.REFERENCES,
                    class_id, interface_name,
                )

            # R2E-A/B: per-class member map — used to resolve the receiver and
            # target of each ``a.connect(b)`` to their declared classes. Both
            # sides must resolve uniquely to a project-local sv_class, and they
            # must be different classes; otherwise no edge is emitted (external
            # UVM types never fabricate a node or edge).
            member_map: dict[str, str] = {}
            for member in class_payload.get("members", []):
                member_map.setdefault(member["name"], member["type"])
            connect_sources = [
                (receiver, target)
                for method_payload in class_payload.get("methods", [])
                for connect in method_payload.get("connects", [])
                for receiver, target in [(connect["receiver"], connect["target"])]
            ]
            for receiver, target in connect_sources:
                receiver_type = member_map.get(receiver)
                target_type = member_map.get(target)
                if receiver_type is None or target_type is None:
                    continue
                receiver_ids = symbols.sv_class_by_name.get(receiver_type, [])
                if len(receiver_ids) != 1:
                    # Ambiguous or absent receiver class — cannot resolve safely.
                    continue
                if receiver_type == target_type:
                    continue
                # Canonical direction: the connecting class references the
                # connected-to class (driver -> sequencer). Reverse reachability
                # is served by the existing dependents traversal.
                self._add_edge(
                    graph, symbols, symbols.sv_class_by_name, IREdgeKind.REFERENCES,
                    receiver_ids[0], target_type,
                )

            # Method call sites from R2B-3b: project-local method calls
            # (sv_method -> sv_method) and UVM factory creates (Class::type_id::create)
            # which resolve to the created class and are recorded as a
            # REFERENCES edge from this class to the created class.
            for method_payload in class_payload.get("methods", []):
                method_id = sv_method_id(
                    project_id, record.relative_path, class_payload["name"], method_payload["name"]
                )
                for call in method_payload.get("calls", []):
                    if call["kind"] == "type_id_create":
                        self._add_edge(
                            graph, symbols, symbols.sv_class_by_name, IREdgeKind.REFERENCES,
                            class_id, call["receiver"],
                        )
                    else:
                        self._add_edge(
                            graph, symbols, symbols.sv_method_by_name, IREdgeKind.CALLS,
                            method_id, call["method"],
                        )

    def _create_py_edges(self, graph, symbols, project_id, record, result) -> None:
        if result is None or not result.success:
            return
        source_id = py_module_id(project_id, record.relative_path)
        for imp in result.payload.get("imports", []):
            module_ref = imp["module"].lstrip(".")
            if not module_ref:
                continue
            self._add_edge(
                graph, symbols, symbols.py_module_by_dotted_suffix, IREdgeKind.PY_IMPORTS,
                source_id, module_ref,
            )

    def _create_tcl_edges(self, graph, symbols, project_id, record, result) -> None:
        if result is None or not result.success:
            return
        source_id = tcl_script_id(project_id, record.relative_path)
        for ref in result.payload.get("source_refs", []):
            basename = ref["path"].rsplit("/", 1)[-1]
            self._add_edge(
                graph, symbols, symbols.tcl_by_basename, IREdgeKind.TCL_SOURCES,
                source_id, basename,
            )

    def _add_edge(
        self,
        graph: IRGraph,
        symbols: SymbolTables,
        index: dict[str, list[str]],
        kind: IREdgeKind,
        source_id: str,
        target_name: str,
    ) -> None:
        resolved_id, ambiguous, candidates = symbols.resolve(index, target_name)
        edge_id_value = edge_id(kind.value, source_id, target_name)
        # Content-derived edge ids make true duplicates possible (e.g. a
        # property-declared REFERENCES and a type_id::create REFERENCES to the
        # same class); skip them so the in-memory graph stays clean.
        if any(e.id == edge_id_value for e in graph.edges):
            return
        graph.edges.append(
            IREdge(
                id=edge_id_value,
                kind=kind,
                source_id=source_id,
                target_name=target_name,
                target_id=resolved_id,
                resolved=resolved_id is not None,
                ambiguous=ambiguous,
                candidate_ids=candidates if ambiguous else [],
            )
        )
