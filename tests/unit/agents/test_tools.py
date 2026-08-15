from pathlib import Path

from semrag.reasoning.agents.tools import ListFilesTool, ReadFileTool, ToolRegistry
from semrag.storage.postgres.manifest_store import PostgresManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner


def test_read_file_returns_content(tmp_path: Path):
    (tmp_path / "RTL").mkdir()
    (tmp_path / "RTL" / "top.sv").write_text("module top;\nendmodule\n")

    tool = ReadFileTool(str(tmp_path))
    result = tool.execute("RTL/top.sv")
    assert "module top" in result


def test_read_file_with_line_range(tmp_path: Path):
    (tmp_path / "f.txt").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadFileTool(str(tmp_path))
    result = tool.execute("f.txt", line_start=2, line_end=3)
    assert result == "line2\nline3"


def test_read_file_missing_file_returns_error_not_exception(tmp_path: Path):
    tool = ReadFileTool(str(tmp_path))
    result = tool.execute("does_not_exist.sv")
    assert "does not exist" in result


def test_read_file_refuses_path_traversal(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("should not be readable via the tool")

    tool = ReadFileTool(str(project_root))
    result = tool.execute("../secret.txt")

    assert "refusing" in result
    assert "should not be readable" not in result


def test_read_file_truncates_very_large_files(tmp_path: Path):
    big_file = tmp_path / "big.txt"
    big_file.write_text("x" * 50_000)
    tool = ReadFileTool(str(tmp_path))
    result = tool.execute("big.txt")
    assert len(result) < 50_000
    assert "truncated" in result


def test_tool_registry_unknown_tool_returns_error():
    registry = ToolRegistry([ReadFileTool(".")])
    result = registry.execute("nonexistent", {})
    assert "unknown tool" in result


def test_tool_registry_invalid_arguments_returns_error(tmp_path: Path):
    registry = ToolRegistry([ReadFileTool(str(tmp_path))])
    result = registry.execute("read_file", {"wrong_param": "x"})
    assert "invalid arguments" in result


def test_list_files_tool_against_live_db(db_session, tmp_path: Path):
    (tmp_path / "RTL").mkdir()
    (tmp_path / "RTL" / "top.sv").write_text("module top; endmodule\n")

    manifest_store = PostgresManifestStore(db_session)
    scanner = Scanner(manifest_store=manifest_store)
    scanner.scan("proj1", str(tmp_path))
    db_session.flush()

    tool = ListFilesTool(db_session, "proj1")
    result = tool.execute()
    assert "RTL/top.sv" in result


def test_list_files_tool_empty_project_returns_message(db_session):
    tool = ListFilesTool(db_session, "nonexistent_project")
    result = tool.execute()
    assert result == "No files found."
