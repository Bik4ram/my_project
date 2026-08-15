"""
Python parser.

Uses the standard library `ast` module rather than a third-party parser —
Python's own grammar is stable, versioned with the interpreter, and `ast`
already gives exact line/column info and full fidelity for every construct
we need (imports, function/class signatures, decorators, docstrings)
without adding a dependency. This is the one parser in Part 3 where "use
the language's own compiler frontend" is both the simplest and the most
correct option, unlike SystemVerilog where no such built-in exists in
Python.

Scope: this extracts *structure* (what's imported, what functions/classes
exist and their signatures) for retrieval and dependency-graph purposes.
It does not attempt call-graph analysis, type inference, or control-flow
extraction — those would require a much heavier analysis (e.g. building on
top of a tool like Jedi or a full CPG) that isn't justified for the RAG
use case this system targets. If the future agent/editing layer (Part 15)
needs call-graph-level precision, that is a distinct, additive analysis
pass over the same source, not a rebuild of this parser.
"""
from __future__ import annotations

import ast
from pathlib import Path

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.parsers.python_tcl.models import (
    ParsedClass,
    ParsedFunction,
    ParsedImport,
    PythonModuleAST,
)
from semrag.ingestion.scanner.models import FileRecord


class PythonParser:
    name = "stdlib_ast"

    def parse(self, absolute_path: Path, file_record: FileRecord) -> ParseResult:
        try:
            source = absolute_path.read_text(errors="replace")
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

        try:
            tree = ast.parse(source, filename=str(absolute_path))
        except SyntaxError as exc:
            return ParseResult(
                project_id=file_record.project_id,
                relative_path=file_record.relative_path,
                parser_name=self.name,
                success=False,
                diagnostics=[
                    ParseDiagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        message=f"SyntaxError: {exc.msg}",
                        line=exc.lineno,
                        column=exc.offset,
                    )
                ],
            )

        module_ast = self._extract(tree)

        return ParseResult(
            project_id=file_record.project_id,
            relative_path=file_record.relative_path,
            parser_name=self.name,
            success=True,
            payload=module_ast.model_dump(),
        )

    def _extract(self, tree: ast.Module) -> PythonModuleAST:
        imports: list[ParsedImport] = []
        functions: list[ParsedFunction] = []
        classes: list[ParsedClass] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ParsedImport(module=alias.name, line=node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                imports.append(
                    ParsedImport(
                        module=module,
                        names=[a.name for a in node.names],
                        line=node.lineno,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))

        return PythonModuleAST(
            module_docstring=ast.get_docstring(tree),
            imports=imports,
            functions=functions,
            classes=classes,
        )

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ParsedFunction:
        args = [a.arg for a in node.args.args]
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        return ParsedFunction(
            name=node.name,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            args=args,
            decorators=decorators,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

    def _extract_class(self, node: ast.ClassDef) -> ParsedClass:
        base_classes = [self._expr_to_str(b) for b in node.bases]
        decorators = [self._decorator_name(d) for d in node.decorator_list]
        methods = [
            self._extract_function(item)
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        return ParsedClass(
            name=node.name,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            base_classes=base_classes,
            methods=methods,
            decorators=decorators,
            docstring=ast.get_docstring(node),
        )

    def _decorator_name(self, node: ast.expr) -> str:
        return self._expr_to_str(node)

    def _expr_to_str(self, node: ast.expr) -> str:
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 - best-effort textual fallback
            return "<expr>"
