"""Structured payload for the Python parser."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedImport(BaseModel):
    module: str
    names: list[str] = Field(
        default_factory=list,
        description="Empty for a plain `import module` statement; populated for `from module import a, b`.",
    )
    line: int


class ParsedFunction(BaseModel):
    name: str
    line: int
    end_line: int
    args: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None
    is_async: bool = False


class ParsedClass(BaseModel):
    name: str
    line: int
    end_line: int
    base_classes: list[str] = Field(default_factory=list)
    methods: list[ParsedFunction] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    docstring: str | None = None


class PythonModuleAST(BaseModel):
    module_docstring: str | None = None
    imports: list[ParsedImport] = Field(default_factory=list)
    functions: list[ParsedFunction] = Field(
        default_factory=list, description="Top-level functions only; methods live under their class."
    )
    classes: list[ParsedClass] = Field(default_factory=list)
