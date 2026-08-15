"""Structured payload for the VHDL parser."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedVhdlPort(BaseModel):
    name: str
    direction: str  # "in" | "out" | "inout" | "buffer"
    data_type: str


class ParsedVhdlGeneric(BaseModel):
    name: str
    data_type: str | None = None
    default_value: str | None = None


class ParsedVhdlEntity(BaseModel):
    name: str
    line: int
    ports: list[ParsedVhdlPort] = Field(default_factory=list)
    generics: list[ParsedVhdlGeneric] = Field(default_factory=list)


class ParsedVhdlArchitecture(BaseModel):
    name: str
    entity_name: str
    line: int


class VhdlFileStructure(BaseModel):
    entities: list[ParsedVhdlEntity] = Field(default_factory=list)
    architectures: list[ParsedVhdlArchitecture] = Field(default_factory=list)
