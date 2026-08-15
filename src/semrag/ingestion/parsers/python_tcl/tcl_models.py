"""Structured payload for the TCL parser."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedProc(BaseModel):
    name: str
    line: int
    args: list[str] = Field(default_factory=list)


class ParsedSourceRef(BaseModel):
    path: str
    line: int


class TclFileStructure(BaseModel):
    procs: list[ParsedProc] = Field(default_factory=list)
    source_refs: list[ParsedSourceRef] = Field(default_factory=list)
