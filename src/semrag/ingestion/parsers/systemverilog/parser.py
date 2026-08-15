"""
SystemVerilog / UVM parser.

Design decision — syntax-tree parsing, not full elaboration/compilation:
pyslang supports two very different levels of analysis. `Compilation`
(elaboration) resolves types, binds identifiers across files, and requires
the *entire* project's include paths, package dependencies, and
`` `define `` macros to be correctly assembled up front — effectively a
full "does this project compile" pass. `SyntaxTree.fromText`/`fromFile`
(what this parser uses) parses one file's grammar into a concrete syntax
tree with zero cross-file knowledge required.

We use syntax-tree parsing here because Part 3's parsers run per-file,
independently, incrementally (only re-parsing what the scanner's
ChangeSet says changed) — which is fundamentally incompatible with
elaboration's "the whole project must be assembled correctly" requirement.
A single missing `include or one file mid-edit would fail elaboration for
the entire compilation unit, which is unacceptable for an indexer that
must keep working on a large, often-mid-edit, real engineering repo.
Syntax-tree parsing degrades gracefully per-file instead: a malformed file
produces diagnostics and partial results, not a failure that blocks
indexing everything else.

The trade-off, stated plainly: this parser does NOT resolve types, does
NOT know if `i2c_master` the instantiated module actually exists anywhere,
and does NOT know the bit-width of an identifier whose type comes from a
typedef in another file. Those are elaboration-level facts. If a future
part of this system needs them (e.g. a "does this design actually
elaborate cleanly" checker), that is a distinct, project-wide compilation
pass layered on top of this per-file syntax parser — not a replacement
for it, since the incremental, always-available indexing behavior this
parser provides is required for RAG regardless of whether full elaboration
is ever added.

Fallback: a tree-sitter-based fallback parser for files pyslang cannot
parse at all (severe syntax errors, non-standard preprocessor usage) is
part of the registry design (see interfaces.py's ordered-fallback-list
mechanism) but is NOT implemented in this delivery — flagged explicitly
rather than stubbed silently. See this package's README.md.
"""
from __future__ import annotations

from pathlib import Path

from pyslang import syntax as sx

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.parsers.systemverilog.models import (
    ParsedAssertion,
    ParsedCall,
    ParsedClass,
    ParsedConnect,
    ParsedInstantiation,
    ParsedMember,
    ParsedMethod,
    ParsedModule,
    ParsedPackage,
    ParsedParameter,
    ParsedPort,
    ParsedProceduralBlock,
    SourceSpan,
    SystemVerilogFileAST,
)
from semrag.ingestion.scanner.models import FileRecord

_PROCEDURAL_BLOCK_KINDS = {
    sx.SyntaxKind.AlwaysBlock: "always",
    sx.SyntaxKind.AlwaysCombBlock: "always_comb",
    sx.SyntaxKind.AlwaysFFBlock: "always_ff",
    sx.SyntaxKind.AlwaysLatchBlock: "always_latch",
    sx.SyntaxKind.InitialBlock: "initial",
    sx.SyntaxKind.FinalBlock: "final",
}

_MAX_ASSERTION_TEXT_CHARS = 500


class SystemVerilogParser:
    """Registered for FileCategory.RTL_VERILOG, TESTBENCH_UVM, and
    ASSERTION — see the parser registry wiring in `pipeline.py`. The
    grammar being parsed is the same in all three cases; the scanner's
    category only reflects *where in the repo* the file lives."""

    name = "pyslang_syntax"

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
                    ParseDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Could not read file: {exc}",
                    )
                ],
            )

        tree = sx.SyntaxTree.fromText(text, str(absolute_path))
        diagnostics = self._collect_tree_diagnostics(tree)

        # A syntax tree with the parser's own diagnostics marked fatal
        # (unrecoverable) is where a tree-sitter fallback would be
        # invoked by the registry — see module docstring. Pyslang
        # recovers from most single-construct errors and continues
        # parsing the rest of the file, so we still attempt extraction
        # even when diagnostics are non-empty, and only mark `success`
        # False if extraction produced zero modules AND there were error
        # diagnostics (i.e. we have nothing usable, not just noise).
        try:
            ast = self._extract(tree)
        except Exception as exc:  # noqa: BLE001 - genuinely must not crash the scan
            diagnostics.append(
                ParseDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Extraction failed after successful parse: {exc}",
                )
            )
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=diagnostics,
            )

        has_errors = any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)
        success = bool(ast.modules or ast.packages or ast.classes) or not has_errors

        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=success,
            diagnostics=diagnostics,
            payload=ast.model_dump(),
        )

    def _collect_tree_diagnostics(self, tree: sx.SyntaxTree) -> list[ParseDiagnostic]:
        results = []
        sm = tree.sourceManager
        for diag in tree.diagnostics:
            try:
                line = sm.getLineNumber(diag.location)
                col = sm.getColumnNumber(diag.location)
            except Exception:  # noqa: BLE001 - location resolution is best-effort
                line, col = None, None
            results.append(
                ParseDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=str(diag.code),
                    line=line,
                    column=col,
                )
            )
        return results

    def _extract(self, tree: sx.SyntaxTree) -> SystemVerilogFileAST:
        sm = tree.sourceManager
        root = tree.root
        top_members = root.members if root.kind == sx.SyntaxKind.CompilationUnit else [root]

        modules: list[ParsedModule] = []
        packages: list[ParsedPackage] = []
        classes: list[ParsedClass] = []

        # Top-level `Class::method` definitions carry the bodies (and hence
        # the call sites) for methods that are only `extern` prototypes
        # inside the class declaration — UVM's standard
        # declare-in-class/define-after pattern. Collected first, then merged
        # into the matching ParsedMethod once every class is built.
        extern_method_bodies: list[tuple[str, str, list[ParsedCall]]] = []

        for member in top_members:
            self._extract_member(member, sm, modules, packages, classes, extern_method_bodies)

        self._merge_extern_method_bodies(classes, extern_method_bodies)

        return SystemVerilogFileAST(modules=modules, packages=packages, classes=classes)

    def _extract_member(
        self,
        member,
        sm,
        modules: list[ParsedModule],
        packages: list[ParsedPackage],
        classes: list[ParsedClass],
        extern_method_bodies: list[tuple[str, str, list[ParsedCall]]],
    ) -> None:
        """Dispatch one compilation-unit member. Package declarations are
        walked recursively so that classes (and anything else) declared
        inside a package are extracted too — UVM testbench code typically
        nests its classes directly in ``package ... endpackage`` bodies."""
        if member.kind == sx.SyntaxKind.ModuleDeclaration:
            modules.append(self._extract_module(member, sm))
        elif member.kind == sx.SyntaxKind.InterfaceDeclaration:
            parsed = self._extract_module(member, sm)
            parsed.is_interface = True
            modules.append(parsed)
        elif member.kind == sx.SyntaxKind.PackageDeclaration:
            packages.append(self._extract_package(member, sm))
            for nested in member.members:
                self._extract_member(nested, sm, modules, packages, classes, extern_method_bodies)
        elif member.kind == sx.SyntaxKind.ClassDeclaration:
            classes.append(self._extract_class(member, sm))
        elif member.kind in (
            sx.SyntaxKind.FunctionDeclaration,
            sx.SyntaxKind.TaskDeclaration,
        ):
            entry = self._extern_method_body(member)
            if entry is not None:
                extern_method_bodies.append(entry)

    def _extern_method_body(
        self, node
    ) -> tuple[str, str, list[ParsedCall], list[ParsedConnect]] | None:
        """Extract ``(class_name, method_name, calls, connects)`` from a
        top-level ``Class::method`` definition (the extern body), or None for
        a free function/task that is not a class method.

        The prototype name is a ``ScopedName`` whose ``left`` is the class
        (possibly itself ``pkg::class``) and whose ``right`` is the method —
        e.g. ``i2c_master_driver::run_phase``. The body statements live in
        ``node.items`` for a TaskDeclaration/FunctionDeclaration.
        """
        proto = getattr(node, "prototype", None)
        if proto is None or proto.kind != sx.SyntaxKind.FunctionPrototype:
            return None
        name = getattr(proto, "name", None)
        if name is None or name.kind != sx.SyntaxKind.ScopedName:
            return None
        class_part, method_part = self._scoped_name_parts(name)
        if class_part is None or method_part is None:
            return None
        body = getattr(node, "items", None)
        if body is None:
            return None
        return (
            class_part,
            method_part,
            self._method_body_calls(node),
            self._method_body_connects(node),
        )

    def _scoped_name_parts(self, scoped_name) -> tuple[str | None, str | None]:
        """Split a ``Class::method`` (or ``pkg::Class::method``) ScopedName
        into its class part and rightmost method part. Returns (None, None)
        for anything that isn't a plain identifier chain (e.g. a parameterized
        or otherwise non-identifier receiver)."""
        parts = []
        cur = scoped_name
        while cur.kind == sx.SyntaxKind.ScopedName:
            right = getattr(cur, "right", None)
            identifier = getattr(right, "identifier", None)
            if identifier is None:
                return None, None
            parts.append(identifier.valueText)
            cur = cur.left
        identifier = getattr(cur, "identifier", None)
        if identifier is None:
            return None, None
        parts.append(identifier.valueText)

        parts = list(reversed(parts))
        if len(parts) < 2:
            return None, None
        return parts[0], parts[-1]

    def _span(self, node, sm) -> SourceSpan:
        r = node.sourceRange
        return SourceSpan(
            start_line=sm.getLineNumber(r.start),
            end_line=sm.getLineNumber(r.end),
        )

    def _extract_package(self, node, sm) -> ParsedPackage:
        return ParsedPackage(name=node.header.name.valueText, span=self._span(node, sm))

    def _extract_module(self, node, sm) -> ParsedModule:
        name = node.header.name.valueText
        ports = self._extract_ports(node)
        parameters = self._extract_parameters(node)

        collected: dict[sx.SyntaxKind, list] = {}

        def collect(n):
            collected.setdefault(n.kind, []).append(n)

        lookup_table = {kind: collect for kind in _PROCEDURAL_BLOCK_KINDS}
        lookup_table[sx.SyntaxKind.HierarchyInstantiation] = collect
        lookup_table[sx.SyntaxKind.ImmediateAssertStatement] = collect
        lookup_table[sx.SyntaxKind.ConcurrentAssertionMember] = collect

        node.visit(lookup_table=lookup_table)

        procedural_blocks = [
            ParsedProceduralBlock(kind=_PROCEDURAL_BLOCK_KINDS[kind], span=self._span(n, sm))
            for kind, nodes in collected.items()
            if kind in _PROCEDURAL_BLOCK_KINDS
            for n in nodes
        ]

        assertions: list[ParsedAssertion] = []
        for n in collected.get(sx.SyntaxKind.ImmediateAssertStatement, []):
            assertions.append(
                ParsedAssertion(
                    kind="immediate",
                    span=self._span(n, sm),
                    raw_text=str(n).strip()[:_MAX_ASSERTION_TEXT_CHARS],
                )
            )
        for n in collected.get(sx.SyntaxKind.ConcurrentAssertionMember, []):
            assertions.append(
                ParsedAssertion(
                    kind="concurrent",
                    span=self._span(n, sm),
                    raw_text=str(n).strip()[:_MAX_ASSERTION_TEXT_CHARS],
                )
            )

        instantiations: list[ParsedInstantiation] = []
        for n in collected.get(sx.SyntaxKind.HierarchyInstantiation, []):
            module_type = n.type.valueText
            for inst in n.instances:
                if inst.kind == sx.SyntaxKind.HierarchicalInstance and hasattr(inst, "decl"):
                    instantiations.append(
                        ParsedInstantiation(
                            instance_name=inst.decl.name.valueText,
                            module_type=module_type,
                        )
                    )

        return ParsedModule(
            name=name,
            span=self._span(node, sm),
            ports=ports,
            parameters=parameters,
            assertions=assertions,
            procedural_blocks=procedural_blocks,
            instantiations=instantiations,
        )

    def _extract_class(self, node, sm) -> ParsedClass:
        name = node.name.valueText
        methods: dict[str, ParsedMethod] = {}
        for item in node.items:
            method = self._class_method(item, sm)
            if method is None:
                continue
            # A method can be declared as an `extern` prototype and then
            # defined inline in the same class body; deduplicate by name so
            # the pair is captured as a single method (first occurrence).
            methods.setdefault(method.name, method)
        return ParsedClass(
            name=name,
            span=self._span(node, sm),
            methods=list(methods.values()),
            extends=self._class_extends(node),
            type_refs=self._class_type_refs(node),
            members=self._class_members(node),
            interface_refs=self._class_interface_refs(node),
        )

    def _class_members(self, node) -> list[ParsedMember]:
        """(name, bare-declared-type) pairs for every typed property.

        ``type`` is the bare identifier of the declared type: the class name
        for a ``NamedType`` (parameterized or not), the interface name for a
        ``virtual`` interface property. Builtins (``int``, ``logic``, ...)
        are not ``NamedType``/``VirtualInterfaceType`` and are naturally
        skipped. This is the per-class member map the IR builder uses to
        resolve ``connect`` receiver/target identifiers to their classes.
        """
        members: list[ParsedMember] = []
        for item in node.items:
            if item.kind != sx.SyntaxKind.ClassPropertyDeclaration:
                continue
            declaration = getattr(item, "declaration", None)
            data_type = getattr(declaration, "type", None)
            if data_type is None:
                continue
            declared_type: str | None = None
            if data_type.kind == sx.SyntaxKind.NamedType:
                type_name = getattr(data_type, "name", None)
                identifier = getattr(type_name, "identifier", None)
                if identifier is not None:
                    declared_type = identifier.valueText
            elif data_type.kind == sx.SyntaxKind.VirtualInterfaceType:
                name = getattr(data_type, "name", None)
                if name is not None:
                    declared_type = name.valueText
            if declared_type is None:
                continue
            for declarator in getattr(declaration, "declarators", []):
                member_name = getattr(getattr(declarator, "name", None), "valueText", None)
                if member_name:
                    members.append(ParsedMember(name=member_name, type=declared_type))
        return members

    def _class_interface_refs(self, node) -> list[str]:
        """Interface names referenced via ``virtual <interface> <member>;``.

        Each ``virtual i2c_master_if vif;`` property contributes the interface
        name (``i2c_master_if``). Deduplicated, source order.
        """
        refs: list[str] = []
        for item in node.items:
            if item.kind != sx.SyntaxKind.ClassPropertyDeclaration:
                continue
            declaration = getattr(item, "declaration", None)
            data_type = getattr(declaration, "type", None)
            if data_type is None or data_type.kind != sx.SyntaxKind.VirtualInterfaceType:
                continue
            name = getattr(data_type, "name", None)
            if name is None:
                continue
            raw = name.valueText.strip()
            if raw and raw not in refs:
                refs.append(raw)
        return refs

    def _class_type_refs(self, node) -> list[str]:
        """Bare type names this class structurally references.

        Captures, from the pyslang syntax tree only (no semantic analysis):
          * the named type of each ``ClassPropertyDeclaration`` data declaration,
            e.g. ``i2c_master_sequence_item item;`` -> ``i2c_master_sequence_item``;
          * the type arguments of any parameterized named type in a property,
            e.g. ``i2c_master_sequencer #(i2c_master_sequence_item) sqr;`` ->
            ``i2c_master_sequencer`` and ``i2c_master_sequence_item``;
          * the type arguments of a parameterized extends clause, e.g.
            ``extends uvm_sequencer #(i2c_master_sequence_item)`` ->
            ``i2c_master_sequence_item``.

        Builtin types (``int``, ``logic``, ``virtual``...) are not ``NamedType``
        so they are naturally skipped. References are deduplicated in source
        order. This is intentionally structural — it does not infer references
        from method bodies or arbitrary text.
        """
        refs: list[str] = []

        def add_identifier(identifier) -> None:
            if identifier is None:
                return
            raw = getattr(identifier, "valueText", "")
            raw = raw.strip()
            if raw and raw not in refs:
                refs.append(raw)

        def add_type_args(base_name) -> None:
            """Collect the bare type names from a ClassName's parameter list."""
            if getattr(base_name, "kind", None) != sx.SyntaxKind.ClassName:
                return
            for param in getattr(base_name, "parameters", []):
                if param.kind != sx.SyntaxKind.OrderedParamAssignment:
                    continue
                expr = getattr(param, "expr", None)
                if expr is None:
                    continue
                add_identifier(getattr(expr, "identifier", None))

        for item in node.items:
            if item.kind != sx.SyntaxKind.ClassPropertyDeclaration:
                continue
            declaration = getattr(item, "declaration", None)
            data_type = getattr(declaration, "type", None)
            if data_type is None or data_type.kind != sx.SyntaxKind.NamedType:
                continue
            type_name = getattr(data_type, "name", None)
            if type_name is None:
                continue
            add_identifier(getattr(type_name, "identifier", None))
            add_type_args(type_name)

        clause = node.extendsClause
        if clause is not None:
            base_name = getattr(clause, "baseName", None)
            if base_name is not None:
                add_type_args(base_name)

        return refs

    def _class_extends(self, node) -> str | None:
        """Bare base-class name from ``class child extends parent``.

        ``extendsClause.baseName`` is either an ``IdentifierNameSyntax``
        (``extends parent``) or a ``ClassNameSyntax`` (``extends
        uvm_driver#(T)``); both carry the bare identifier, so parameterized
        bases are reduced to their class name (``uvm_driver``), which is the
        name an ``sv_class`` node would be indexed under.
        """
        clause = node.extendsClause
        if clause is None:
            return None
        base_name = getattr(clause, "baseName", None)
        if base_name is None:
            return None
        identifier = getattr(base_name, "identifier", None)
        if identifier is None:
            return None
        raw = getattr(identifier, "valueText", "")
        return raw.strip() or None

    def _class_method(self, item, sm) -> ParsedMethod | None:
        """Extract one method (function/task) from a class item.

        Accepts both an `extern` prototype (``ClassMethodPrototype``) and an
        inline implementation (``ClassMethodDeclaration``) — both carry the
        same ``FunctionPrototype`` shape (``name`` + ``keyword``). Returns
        None for any other class item.
        """
        if item.kind == sx.SyntaxKind.ClassMethodPrototype:
            proto = item.prototype
        elif item.kind == sx.SyntaxKind.ClassMethodDeclaration:
            declaration = getattr(item, "declaration", None)
            if declaration is None:
                return None
            proto = getattr(declaration, "prototype", None)
        else:
            return None

        if proto is None or proto.kind != sx.SyntaxKind.FunctionPrototype:
            return None
        method_name = str(proto.name).strip()
        if not method_name:
            return None
        keyword = str(proto.keyword).strip() if getattr(proto, "keyword", None) is not None else ""
        return ParsedMethod(
            name=method_name,
            kind="task" if "task" in keyword else "function",
            span=self._span(item, sm),
            calls=self._method_body_calls(item),
            connects=self._method_body_connects(item),
        )

    def _method_body_calls(self, node) -> list[ParsedCall]:
        """Collect method call sites from a method body node.

        ``node`` is either a ``ClassMethodDeclaration`` (inline body) or a
        top-level ``TaskDeclaration``/``FunctionDeclaration`` (extern body);
        in both cases the body statements live in ``node.items`` and are
        reachable via ``visit``. Returns deduplicated ParsedCall entries in
        source order.
        """
        invocations: list = []
        node.visit(lookup_table={sx.SyntaxKind.InvocationExpression: invocations.append})

        calls: list[ParsedCall] = []
        seen: set[tuple[str, str | None, str]] = set()
        for inv in invocations:
            parsed = self._call_from_invocation(inv)
            if parsed is None:
                continue
            key = (parsed.method, parsed.receiver, parsed.kind)
            if key in seen:
                continue
            seen.add(key)
            calls.append(parsed)
        return calls

    def _method_body_connects(self, node) -> list[ParsedConnect]:
        """Collect ``a.connect(b)`` sites from a method body node.

        ``node`` is either a ``ClassMethodDeclaration`` (inline body) or a
        top-level ``TaskDeclaration``/``FunctionDeclaration`` (extern body),
        mirroring ``_method_body_calls``. Returns deduplicated ParsedConnect
        entries (by receiver, target) in source order. Only plain member
        identifier chains whose leftmost identifier is a name (not
        ``this``/``super``) are kept — see ``_connect_from_invocation``.
        """
        invocations: list = []
        node.visit(lookup_table={sx.SyntaxKind.InvocationExpression: invocations.append})

        connects: list[ParsedConnect] = []
        seen: set[tuple[str, str]] = set()
        for inv in invocations:
            parsed = self._connect_from_invocation(inv)
            if parsed is None:
                continue
            key = (parsed.receiver, parsed.target)
            if key in seen:
                continue
            seen.add(key)
            connects.append(parsed)
        return connects

    def _connect_from_invocation(self, inv) -> ParsedConnect | None:
        """Reduce one InvocationExpression to a ParsedConnect, or None.

        Matches the UVM connect pattern ``<receiver>.<...>.connect(<target>
        .<...>)`` where ``<receiver>`` and ``<target>`` are the leftmost
        identifiers of the callee path and the first ordered argument path
        respectively. Returns None unless the invoked method is exactly
        ``connect`` and both leftmost identifiers are plain member names
        (``this``/``super`` handles are dropped so we never emit a fabricated
        ``this``/``super`` pair)."""
        left = getattr(inv, "left", None)
        if left is None or left.kind != sx.SyntaxKind.ScopedName:
            return None

        parts: list[str] = []
        cur = left
        while cur.kind == sx.SyntaxKind.ScopedName:
            right = getattr(cur, "right", None)
            identifier = getattr(right, "identifier", None)
            if identifier is None:
                return None
            parts.append(identifier.valueText)
            cur = cur.left
        identifier = getattr(cur, "identifier", None)
        if identifier is not None:
            parts.append(identifier.valueText)
        else:
            # leftmost of the callee path is a handle (this/super) — not a member.
            return None
        parts = list(reversed(parts))
        if len(parts) < 2 or parts[-1] != "connect":
            return None

        receiver = parts[0]
        target = None
        for arg in getattr(inv, "arguments", []):
            if arg.kind != sx.SyntaxKind.OrderedArgument:
                continue
            target = self._leftmost_identifier(getattr(arg, "expr", None))
            break
        if target is None:
            return None
        return ParsedConnect(receiver=receiver, target=target)

    def _leftmost_identifier(self, expr) -> str | None:
        """Leftmost identifier of an expression path, or None.

        Unwraps ``SimplePropertyExpr``/``SimpleSequenceExpr`` wrappers and
        descends the left spine of a ``ScopedName`` chain to the first
        ``IdentifierName`` (``sqr.seq_item_export`` -> ``sqr``). Returns None
        for ``this``/``super`` handles and any other non-identifier base so
        those never reach the connect receiver/target pair."""
        for _ in range(20):  # safety bound on chain depth
            if expr is None:
                return None
            kind = expr.kind
            if kind == sx.SyntaxKind.IdentifierName:
                identifier = getattr(expr, "identifier", None)
                return identifier.valueText if identifier is not None else None
            if kind in (sx.SyntaxKind.SimplePropertyExpr, sx.SyntaxKind.SimpleSequenceExpr):
                expr = getattr(expr, "expr", None)
                continue
            if kind == sx.SyntaxKind.ScopedName:
                expr = getattr(expr, "left", None)
                continue
            return None
        return None

    def _call_from_invocation(self, inv) -> ParsedCall | None:
        """Reduce one InvocationExpression to a ParsedCall, or None for calls
        we cannot (or should not) represent structurally — system tasks
        (``$display``), and anything whose callee isn't a plain identifier or
        identifier chain."""
        left = getattr(inv, "left", None)
        if left is None:
            return None

        if left.kind == sx.SyntaxKind.IdentifierName:
            identifier = getattr(left, "identifier", None)
            if identifier is None:
                return None
            return ParsedCall(method=identifier.valueText, receiver=None, kind="method")

        if left.kind == sx.SyntaxKind.ScopedName:
            parts: list[str] = []
            cur = left
            while cur.kind == sx.SyntaxKind.ScopedName:
                right = getattr(cur, "right", None)
                identifier = getattr(right, "identifier", None)
                if identifier is None:
                    return None
                parts.append(identifier.valueText)
                cur = cur.left
            identifier = getattr(cur, "identifier", None)
            if identifier is not None:
                parts.append(identifier.valueText)
            elif cur.kind == sx.SyntaxKind.SuperHandle:
                # super.method() — the receiver is the special `super` handle.
                parts.append("super")
            else:
                return None
            parts = list(reversed(parts))

            if len(parts) >= 2:
                receiver = parts[0]
                method = parts[-1]
            else:
                receiver, method = None, parts[0]

            # UVM factory pattern: Class::type_id::create(...) — the receiver
            # is the class whose instance is being created. Recorded with a
            # distinct kind so the IR can map it to a class reference instead
            # of a method call.
            if len(parts) >= 3 and parts[-2] == "type_id" and parts[-1] == "create":
                return ParsedCall(method="create", receiver=parts[0], kind="type_id_create")

            return ParsedCall(method=method, receiver=receiver, kind="method")

        return None

    def _merge_extern_method_bodies(
        self,
        classes: list[ParsedClass],
        extern_method_bodies: list[tuple[str, str, list[ParsedCall], list[ParsedConnect]]],
    ) -> None:
        """Attach call sites and connects from top-level ``Class::method``
        definitions to the matching ``ParsedMethod`` in the same file.

        The class declaration carries the ``extern`` prototype (no body); the
        separate top-level definition carries the body. Both are captured in
        the same file (UVM's declare-in-class/define-after pattern), so we
        look up by ``(class_name, method_name)`` and merge the calls and
        connects into the existing method. Cross-file extern definitions
        cannot be attached here — the class isn't in this file's payload —
        and are dropped (the prototype still yields the method, just with no
        call sites)."""
        for class_name, method_name, calls, connects in extern_method_bodies:
            for cls in classes:
                if cls.name != class_name:
                    continue
                for method in cls.methods:
                    if method.name != method_name:
                        continue
                    merged_calls = list(method.calls)
                    seen_calls = {(c.method, c.receiver, c.kind) for c in merged_calls}
                    for call in calls:
                        key = (call.method, call.receiver, call.kind)
                        if key in seen_calls:
                            continue
                        seen_calls.add(key)
                        merged_calls.append(call)
                    method.calls = merged_calls

                    merged_connects = list(method.connects)
                    seen_connects = {(c.receiver, c.target) for c in merged_connects}
                    for connect in connects:
                        key = (connect.receiver, connect.target)
                        if key in seen_connects:
                            continue
                        seen_connects.add(key)
                        merged_connects.append(connect)
                    method.connects = merged_connects
                    break
                break

    def _extract_ports(self, module_node) -> list[ParsedPort]:
        ports: list[ParsedPort] = []
        port_list = module_node.header.ports
        if port_list is None or port_list.kind != sx.SyntaxKind.AnsiPortList:
            # Non-ANSI port lists (declared separately in the module body)
            # are a known gap — see this package's README.md future work.
            return ports

        for p in port_list.ports:
            if p.kind != sx.SyntaxKind.ImplicitAnsiPort:
                continue
            declarator = p.declarator
            header = p.header
            direction = ""
            data_type = ""
            if header is not None and header.kind == sx.SyntaxKind.VariablePortHeader:
                direction = str(header.direction).strip() if header.direction else ""
                data_type = str(header.dataType).strip()
            ports.append(
                ParsedPort(
                    name=declarator.name.valueText,
                    direction=direction,
                    data_type=data_type,
                )
            )
        return ports

    def _extract_parameters(self, module_node) -> list[ParsedParameter]:
        parameters: list[ParsedParameter] = []
        param_list = module_node.header.parameters
        if param_list is None:
            return parameters

        for decl in param_list.declarations:
            if decl.kind != sx.SyntaxKind.ParameterDeclaration:
                continue
            data_type = str(decl.type).strip() if getattr(decl, "type", None) else None
            for declarator in decl.declarators:
                default_value = None
                if declarator.initializer is not None:
                    default_value = str(declarator.initializer.expr).strip()
                parameters.append(
                    ParsedParameter(
                        name=declarator.name.valueText,
                        data_type=data_type,
                        default_value=default_value,
                    )
                )
        return parameters
