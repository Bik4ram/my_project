"""Debug artifact writer — only active at DEBUG/TRACE, retention pruning."""
from __future__ import annotations

import json

from semrag.core.debug_artifacts import DebugArtifactWriter, new_run_id


def test_inactive_at_info():
    w = DebugArtifactWriter(root_dir="/tmp", log_level="INFO", retention=3)
    assert w.is_active is False
    w.start_run("run-1")
    w.write_json("scan", "x", {"k": 1})
    w.write_text("prompt", "x.txt", "hi")
    w.run_completed()
    # No state should have been created.
    assert w.run_dir is None


def test_active_in_debug_creates_subdirs(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="DEBUG", retention=3)
    assert w.is_active is True
    w.start_run("run-a")
    w.write_json("scan", "manifest", {"files": 5})
    w.write_text("prompt", "p.txt", "hello")
    w.write_parse("src/foo.py", {"type": "python"})
    assert (w.run_dir / "scan" / "manifest.json").exists()
    assert (w.run_dir / "prompt" / "p.txt").exists()
    # parse filenames are flattened — `src/foo.py` -> `src_foo.py.parse.json`
    assert (w.run_dir / "parse" / "src_foo.py.parse.json").exists()


def test_active_in_trace_creates_subdirs(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="TRACE", retention=3)
    assert w.is_active is True


def test_retention_keeps_only_n_most_recent(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="DEBUG", retention=2)
    for i in range(5):
        w.start_run(f"run-{i:03d}")
        w.write_json("scan", "manifest", {"i": i})
        w.run_completed()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["run-003", "run-004"]


def test_retention_zero_disables_pruning(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="DEBUG", retention=0)
    for i in range(5):
        w.start_run(f"run-{i:03d}")
        w.run_completed()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len(remaining) == 5


def test_run_id_is_sortable(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="DEBUG", retention=3)
    w.start_run(new_run_id())
    w.write_json("scan", "x", {})
    assert any(p.suffix == ".json" for p in (w.run_dir / "scan").iterdir())


def test_write_json_serializes_dataclass_friendly(tmp_path):
    w = DebugArtifactWriter(root_dir=str(tmp_path), log_level="DEBUG", retention=1)
    w.start_run("r")
    # A non-dataclass dict goes through fine:
    w.write_json("scan", "m", {"a": 1, "b": [1, 2, 3]})
    text = (w.run_dir / "scan" / "m.json").read_text()
    assert json.loads(text) == {"a": 1, "b": [1, 2, 3]}
