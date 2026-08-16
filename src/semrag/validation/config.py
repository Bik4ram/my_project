"""Project validation configuration (model + loader).

The validation layer deliberately does NOT accept free-form shell commands.
Each registered project has a JSON config that declares a fixed set of named
validation targets (e.g. ``rtl_lint``, ``tb_lint``). Each target is a fixed
argv list for a trusted tool (currently Verilator) plus the project-relative
include dirs / source files / defines it needs.

The API endpoint only accepts a target *name*; the command is assembled
server-side from the config. The LLM can never inject arguments, chain
shell operators, or pick a working directory.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Tools that a validation config may invoke. Anything else is rejected at
# load time so a compromised/buggy config cannot point the runner at an
# arbitrary binary.
_ALLOWED_TOOLS = frozenset({"verilator"})


class ValidationTarget(BaseModel):
    """A single named validation command for a project."""

    description: str = ""
    tool_args: list[str] = Field(default_factory=list)
    incdirs: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    defines: list[str] = Field(default_factory=list)


class TrustedExternalDep(BaseModel):
    """A single trusted external dependency for a validation project.

    The path is declared by the operator in the project config (not by the
    LLM or request body) and is the ONLY way a validation target can
    reference files outside the project root. Targets reference these via
    the ``ext:<name>/<subpath>`` form for incdirs and files.

    The path may be an absolute directory, an absolute directory using
    ``$VAR``/``${VAR}`` environment expansion (portable UVM checkouts via
    ``${SEMRAG_UVM_ROOT}``), or a relative directory resolved against the
    validation config directory (in-repo helper stubs). See
    ``runner._resolve_path``.
    """

    name: str
    path: str

    def model_post_init(self, _ctx) -> None:  # noqa: D105
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(
                f"trusted_external_dep name {self.name!r} is invalid "
                "(must be a non-empty bare identifier)"
            )
        if not self.path:
            raise ValueError(
                f"trusted_external_dep {self.name!r} has empty path"
            )


class ValidationConfig(BaseModel):
    """Top-level validation config for one project."""

    project_id: str
    tool: str = "verilator"
    timeout_seconds: int = 120
    trusted_external_deps: list[TrustedExternalDep] = Field(default_factory=list)
    targets: dict[str, ValidationTarget]

    def model_post_init(self, _ctx) -> None:  # noqa: D105
        if self.tool not in _ALLOWED_TOOLS:
            raise ValueError(
                f"validation tool {self.tool!r} is not in the allowlist "
                f"{sorted(_ALLOWED_TOOLS)}"
            )
        names = [d.name for d in self.trusted_external_deps]
        if len(names) != len(set(names)):
            raise ValueError(
                f"duplicate trusted_external_dep names: {names}"
            )


def validation_config_dir() -> Path:
    """Resolve the directory that holds per-project validation configs."""
    return _REPO_ROOT / "configs" / "validation"


def load_validation_config(project_id: str) -> ValidationConfig | None:
    """Load the validation config for a project, or None if not registered."""
    cfg_path = validation_config_dir() / f"{project_id}.json"
    if not cfg_path.is_file():
        return None
    raw = cfg_path.read_text(encoding="utf-8")
    return ValidationConfig.model_validate_json(raw)
