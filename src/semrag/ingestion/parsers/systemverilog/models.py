"""
Structured payload produced by the SystemVerilog parser.

This is the language-specific shape referenced generically by
`ParseResult.payload` in Part 3's shared models. It becomes the input to
Part 4's IR builder for module/port/parameter/assertion nodes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    start_line: int
    end_line: int


class ParsedPort(BaseModel):
    name: str
    direction: str  # "input" | "output" | "inout" | "ref" | "" (unspecified)
    data_type: str  # raw text, e.g. "logic", "logic [7:0]", "i2c_state_t"


class ParsedParameter(BaseModel):
    name: str
    data_type: str | None = None
    default_value: str | None = None


class ParsedAssertion(BaseModel):
    kind: str  # "concurrent" | "immediate"
    span: SourceSpan
    raw_text: str = Field(
        ..., description="Full assertion text, truncated by the caller if huge."
    )


class ParsedProceduralBlock(BaseModel):
    kind: str  # "always_ff" | "always_comb" | "always" | "initial" | "final"
    span: SourceSpan


class ParsedInstantiation(BaseModel):
    """A `module_type inst_name(...)` instantiation found inside a module
    body — the raw signal for Part 4's module-dependency graph edges."""
    instance_name: str
    module_type: str


class ParsedModule(BaseModel):
    name: str
    span: SourceSpan
    ports: list[ParsedPort] = Field(default_factory=list)
    parameters: list[ParsedParameter] = Field(default_factory=list)
    assertions: list[ParsedAssertion] = Field(default_factory=list)
    procedural_blocks: list[ParsedProceduralBlock] = Field(default_factory=list)
    instantiations: list[ParsedInstantiation] = Field(default_factory=list)
    is_interface: bool = False


class ParsedPackage(BaseModel):
    name: str
    span: SourceSpan


class ParsedCall(BaseModel):
    """A single method call site found structurally inside a method body.

    ``method`` is the invoked method's bare name (rightmost identifier of
    the callee path). ``receiver`` is the leftmost identifier of the callee
    path — the object/class the call is made on (``seq_item_port`` in
    ``seq_item_port.get_next_item(...)``, ``i2c_master_sequence_item`` in
    ``i2c_master_sequence_item::type_id::create(...)``), or None for an
    unqualified call (``start_item(item)``). ``kind`` distinguishes an
    ordinary call from the UVM factory pattern ``Class::type_id::create``.
    """
    method: str
    receiver: str | None = None
    kind: str = "method"  # "method" | "type_id_create"


class ParsedMember(BaseModel):
    """One typed member of a class: the bare declared type and the member
    name. ``type`` is the class name for a NamedType property, the interface
    name for a ``virtual`` interface property, and the bare name (stripped of
    parameters) for a parameterized type."""
    name: str
    type: str


class ParsedConnect(BaseModel):
    """One ``a.connect(b)`` relationship found structurally in a method body.

    ``receiver`` is the leftmost identifier of the callee path (the member
    whose port is being connected); ``target`` is the leftmost identifier of
    the first argument path (the member whose port is the connect target).
    Only project-local member identifiers that resolve to classes are kept
    (``this``/``super`` targets are dropped by the parser).
    """
    receiver: str
    target: str


class ParsedMethod(BaseModel):
    """A function/task declared inside a class (captured from its
    ``ClassMethodPrototype``), or as a top-level ``Class::method`` extern
    definition. ``kind`` is ``"function"`` or ``"task"``."""
    name: str
    kind: str  # "function" | "task"
    span: SourceSpan
    calls: list[ParsedCall] = Field(
        default_factory=list,
        description=(
            "Method call sites found in this method's body, in source order, "
            "deduplicated by (method, receiver, kind). Empty for a bare "
            "``extern`` prototype whose body lives in a separate "
            "``Class::method`` definition (merged in after extraction) and for "
            "methods with no calls."
        ),
    )
    connects: list[ParsedConnect] = Field(
        default_factory=list,
        description=(
            "``a.connect(b)`` sites found in this method's body, in source "
            "order, deduplicated by (receiver, target). Empty for a bare "
            "``extern`` prototype whose body lives in a separate "
            "``Class::method`` definition (merged in after extraction) and for "
            "methods with no connects."
        ),
    )


class ParsedClass(BaseModel):
    name: str
    span: SourceSpan
    methods: list[ParsedMethod] = Field(default_factory=list)
    extends: str | None = Field(
        default=None,
        description=(
            "Bare base-class name from the ``extends`` clause, e.g. ``i2c_master_base_seq`` "
            "for ``class child extends i2c_master_base_seq``. Parameterized bases are "
            "stripped to the bare name (``uvm_driver#(T)`` -> ``uvm_driver``). None when "
            "the class has no extends clause."
        ),
    )
    type_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Bare project-local type/class names this class structurally references: the "
            "named types of its data-declaration properties (``i2c_master_sequence_item "
            "item;`` -> ``i2c_master_sequence_item``) and the type arguments of any "
            "parameterized reference (``uvm_sequencer #(i2c_master_sequence_item)`` -> "
            "``i2c_master_sequence_item``). Deduplicated, in source order. Builtin types "
            "(``int``, ``logic``, ``virtual``...) are not included."
        ),
    )
    members: list[ParsedMember] = Field(
        default_factory=list,
        description=(
            "The class's typed properties as (name, type) pairs, in source order. "
            "Serves as the per-class member map the IR builder uses to resolve "
            "``connect`` receiver/target identifiers to their declared classes."
        ),
    )
    interface_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Interface names referenced via ``virtual <interface> <member>;`` "
            "properties, in source order, deduplicated. These resolve to "
            "``sv_interface`` nodes in the IR (``virtual i2c_master_if vif;`` -> "
            "``i2c_master_if``)."
        ),
    )


class SystemVerilogFileAST(BaseModel):
    """Top-level payload for one .sv/.svh file. A single file can contain
    multiple modules (and/or packages, and/or classes), so these are lists,
    not a single item."""
    modules: list[ParsedModule] = Field(default_factory=list)
    packages: list[ParsedPackage] = Field(default_factory=list)
    classes: list[ParsedClass] = Field(default_factory=list)
