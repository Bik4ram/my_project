"""
Tests for the project validation config + runner.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from semrag.validation.config import ValidationConfig, load_validation_config
from semrag.validation.runner import _classify, build_argv, run_validation


def test_config_rejects_unallowed_tool():
    with pytest.raises(ValueError, match="not in the allowlist"):
        ValidationConfig(
            project_id="x",
            tool="bash",
            targets={"t": {"files": ["a.sv"]}},
        )


def test_load_validation_config_missing_project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "semrag.validation.config._REPO_ROOT",
        tmp_path,
    )
    (tmp_path / "configs" / "validation").mkdir(parents=True)
    assert load_validation_config("nope") is None


def test_build_argv_resolves_paths_under_root(tmp_path):
    root = tmp_path
    (root / "rtl").mkdir(parents=True)
    (root / "rtl/a.sv").touch()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={
            "lint": {
                "tool_args": ["--lint-only"],
                "incdirs": ["rtl"],
                "files": ["rtl/a.sv"],
                "defines": ["FOO"],
            }
        },
    )
    argv = build_argv(root, cfg, "lint")
    assert argv[0] == "verilator"
    assert "--lint-only" in argv
    assert f"+incdir+{root / 'rtl'}" in argv
    assert str(root / "rtl/a.sv") in argv
    assert "-DFOO" in argv


def test_build_argv_rejects_traversal(tmp_path):
    root = tmp_path
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={"lint": {"files": ["../escape.sv"]}},
    )
    with pytest.raises(ValueError, match="escapes project root"):
        build_argv(root, cfg, "lint")


def test_build_argv_rejects_absolute_file(tmp_path):
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={"lint": {"files": ["/etc/passwd"]}},
    )
    with pytest.raises(ValueError, match="escapes project root"):
        build_argv(tmp_path, cfg, "lint")


def test_classify_passed():
    result = subprocess.CompletedProcess([], 0)
    assert _classify(result) == ("passed", None)


def test_classify_failed():
    result = subprocess.CompletedProcess([], 1, stdout="", stderr="%Error: syntax")
    assert _classify(result) == ("failed", None)


def test_classify_blocked_on_uvm():
    result = subprocess.CompletedProcess(
        [], 1, stdout="", stderr="Cannot find include file: uvm_macros.svh"
    )
    status, reason = _classify(result)
    assert status == "blocked"
    assert "UVM" in (reason or "")


def test_run_validation_missing_tool(tmp_path, monkeypatch):
    (tmp_path / "a.sv").touch()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={"lint": {"files": ["a.sv"]}},
    )
    monkeypatch.setattr("semrag.validation.runner.shutil.which", lambda _: None)
    result = run_validation(tmp_path, cfg, "lint", log_output=False)
    assert result.status == "blocked"
    assert "verilator" in (result.blocked_reason or "")


def test_run_validation_unknown_target(tmp_path):
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={"lint": {"files": ["a.sv"]}},
    )
    with pytest.raises(KeyError):
        run_validation(tmp_path, cfg, "nope", log_output=False)


# --------------------------------------------------------------------------
# Trusted external dependency tests
# --------------------------------------------------------------------------


def test_build_argv_accepts_trusted_external_incdir(tmp_path):
    dep_root = tmp_path / "deps" / "uvm"
    dep_root.mkdir(parents=True)
    (dep_root / "uvm_macros.svh").touch()
    (dep_root / "uvm_pkg.sv").touch()
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "uvm", "path": str(dep_root)}],
        targets={
            "lint": {
                "tool_args": ["--lint-only"],
                "incdirs": ["ext:uvm"],
                "files": ["ext:uvm/uvm_pkg.sv"],
            }
        },
    )
    argv = build_argv(project_root, cfg, "lint")
    assert f"+incdir+{dep_root.resolve()}" in argv
    assert str((dep_root / "uvm_pkg.sv").resolve()) in argv


def test_build_argv_rejects_unconfigured_external_dep(tmp_path):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        targets={
            "lint": {
                "incdirs": ["ext:not_configured/whatever"],
            }
        },
    )
    with pytest.raises(ValueError, match="unconfigured external dependency"):
        build_argv(project_root, cfg, "lint")


def test_build_argv_rejects_external_dep_traversal(tmp_path):
    dep_root = tmp_path / "deps" / "uvm"
    dep_root.mkdir(parents=True)
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "uvm", "path": str(dep_root)}],
        targets={
            "lint": {
                "incdirs": ["ext:uvm/../escape"],
            }
        },
    )
    with pytest.raises(ValueError, match="escapes dep root"):
        build_argv(project_root, cfg, "lint")


def test_build_argv_rejects_external_dep_absolute_subpath(tmp_path):
    dep_root = tmp_path / "deps" / "uvm"
    dep_root.mkdir(parents=True)
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "uvm", "path": str(dep_root)}],
        targets={
            "lint": {
                "incdirs": ["ext:uvm//etc/passwd"],
            }
        },
    )
    with pytest.raises(ValueError, match="escapes dep root"):
        build_argv(project_root, cfg, "lint")


def test_build_argv_rejects_external_dep_missing_file(tmp_path):
    dep_root = tmp_path / "deps" / "uvm"
    dep_root.mkdir(parents=True)
    project_root = tmp_path / "proj"
    project_root.mkdir()
    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "uvm", "path": str(dep_root)}],
        targets={
            "lint": {
                "files": ["ext:uvm/uvm_pkg.sv"],
            }
        },
    )
    with pytest.raises(FileNotFoundError, match="validation file does not exist"):
        build_argv(project_root, cfg, "lint")


def test_config_rejects_duplicate_external_dep_names():
    with pytest.raises(ValueError, match="duplicate trusted_external_dep"):
        ValidationConfig(
            project_id="p",
            tool="verilator",
            trusted_external_deps=[
                {"name": "uvm", "path": "/tmp/uvm-a"},
                {"name": "uvm", "path": "/tmp/uvm-b"},
            ],
            targets={"lint": {"files": ["a.sv"]}},
        )


def test_config_rejects_external_dep_name_with_slash():
    with pytest.raises(ValueError, match="invalid"):
        ValidationConfig(
            project_id="p",
            tool="verilator",
            trusted_external_deps=[{"name": "bad/name", "path": "/tmp/uvm"}],
            targets={"lint": {"files": ["a.sv"]}},
        )


# --------------------------------------------------------------------------
# sv_generated_lint target tests
# --------------------------------------------------------------------------


def test_rtl_uvm_config_registers_sv_generated_lint():
    """The rtl_uvm config must register the lightweight generated-artifact
    lint target and it must not depend on the full UVM library."""
    real_cfg = load_validation_config("rtl_uvm")
    assert real_cfg is not None
    assert "sv_generated_lint" in real_cfg.targets
    target = real_cfg.targets["sv_generated_lint"]
    # Lightweight: must NOT compile uvm_pkg.sv
    assert "uvm_pkg.sv" not in " ".join(target.files)
    # Must use verilator only
    assert real_cfg.tool == "verilator"


def _make_generated_lint_fixture(tmp_path, seq_text, test_text):
    """Build a fake RTL_UVM layout (tb seq/test dirs) plus the real stub
    harness files, returning the built argv (unused files checked)."""
    import semrag.validation.config as cfg_mod

    real_cfg = load_validation_config("rtl_uvm")
    assert real_cfg is not None
    (tmp_path / "tb" / "i2c_master_seq_lib").mkdir(parents=True)
    (tmp_path / "tb" / "i2c_master_test_lib").mkdir(parents=True)
    (tmp_path / "tb" / "i2c_master_seq_lib" / "i2c_master_read_seq.svh").write_text(
        seq_text
    )
    (tmp_path / "tb" / "i2c_master_test_lib" / "i2c_master_read_test.svh").write_text(
        test_text
    )
    (tmp_path / "tb" / "i2c_master_test_lib" / "i2c_master_read_test4.svh").write_text(
        "class i2c_master_read_test4; endclass\n"
    )
    return real_cfg


def test_sv_generated_lint_argv_build(tmp_path):
    """build_argv must produce a full verilator argv for the lightweight
    generated-artifact lint target using the stub harness."""
    cfg = _make_generated_lint_fixture(
        tmp_path,
        "class i2c_master_read_seq; endclass\n",
        "class i2c_master_read_test; endclass\n",
    )
    argv = build_argv(tmp_path, cfg, "sv_generated_lint")
    joined = " ".join(argv)
    assert argv[0] == "verilator"
    assert "--lint-only" in joined
    assert "--bbox-unsup" in joined
    assert "-Wno-fatal" in joined
    assert "--timing" in joined
    assert "+incdir+" + str(tmp_path / "tb" / "i2c_master_seq_lib") in joined
    assert "+incdir+" + str(tmp_path / "tb" / "i2c_master_test_lib") in joined
    # Harness file must be included and must NOT pull in uvm_pkg.sv
    assert "sv_generated_lint_harness.sv" in joined
    assert "uvm_pkg.sv" not in joined


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_sv_generated_lint_rejects_syntax_error(tmp_path):
    """A syntax error in a generated file must produce a non-zero result —
    the lightweight target must not silently pass broken generated code."""
    cfg = _make_generated_lint_fixture(
        tmp_path,
        "class i2c_master_read_seq; this is a broken junk line endclass\n",
        "class i2c_master_read_test; endclass\n",
    )
    result = run_validation(tmp_path, cfg, "sv_generated_lint", log_output=False)
    assert result.exit_code != 0
    assert result.status == "failed"

def test_build_argv_expands_env_var_in_external_dep_path(tmp_path, monkeypatch):
    dep_root = tmp_path / "uvm_core"
    (dep_root / "inc").mkdir(parents=True)
    (dep_root / "uvm_pkg.sv").write_text("package uvm_pkg; endpackage\n")
    monkeypatch.setenv("SEMRAG_UVM_ROOT", str(dep_root))

    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "uvm", "path": "${SEMRAG_UVM_ROOT}"}],
        targets={
            "t": {
                "incdirs": ["ext:uvm/inc"],
                "files": ["ext:uvm/uvm_pkg.sv"],
            }
        },
    )
    argv = build_argv(tmp_path, cfg, "t")
    joined = " ".join(argv)
    assert f"+incdir+{dep_root}/inc" in joined
    assert f"{dep_root}/uvm_pkg.sv" in joined


def test_build_argv_resolves_relative_external_dep_to_config_dir(tmp_path, monkeypatch):
    import semrag.validation.config as cfg_mod
    from semrag.validation.runner import _resolve_path

    monkeypatch.setattr(cfg_mod, "_REPO_ROOT", tmp_path)
    stubs = tmp_path / "configs" / "validation" / "stubs" / "rtl_uvm"
    stubs.mkdir(parents=True)
    (stubs / "harness.sv").write_text("// harness\n")

    cfg = ValidationConfig(
        project_id="p",
        tool="verilator",
        trusted_external_deps=[{"name": "stubs", "path": "stubs/rtl_uvm"}],
        targets={
            "t": {"incdirs": ["ext:stubs"], "files": ["ext:stubs/harness.sv"]}
        },
    )
    argv = build_argv(tmp_path, cfg, "t")
    joined = " ".join(argv)
    assert f"+incdir+{stubs}" in joined
    assert f"{stubs}/harness.sv" in joined
