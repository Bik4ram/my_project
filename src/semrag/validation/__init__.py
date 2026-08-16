"""Controlled project validation.

Provides a safe, config-driven execution layer for project validation
commands (currently Verilator lint). The LLM selects a named target from a
per-project JSON config; it never supplies shell text.
"""

from semrag.validation.config import ValidationConfig, load_validation_config
from semrag.validation.runner import ValidationResult, run_validation

__all__ = [
    "ValidationConfig",
    "ValidationResult",
    "load_validation_config",
    "run_validation",
]