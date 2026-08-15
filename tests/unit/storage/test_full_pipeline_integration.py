"""
Full pipeline test: real files on disk -> real Scanner -> real parser
registry -> real IR builder -> real Postgres persistence -> reload and
verify. This is the test that proves Part 5 actually plugs into Parts 2-4
correctly, not just that its own repositories work in isolation against
hand-built IRGraph/ScanManifest objects.
"""
from pathlib import Path

from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.ir.models import IREdgeKind, IRNodeKind
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.postgres.manifest_store import PostgresManifestStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_full_pipeline_persists_to_postgres(tmp_path: Path, db_session):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl); endmodule\n",
    )
    _write(
        tmp_path, "RTL/i2c_top.sv",
        "module i2c_top;\n  i2c_master u_master(.clk(1'b0));\nendmodule\n",
    )

    manifest_store = PostgresManifestStore(db_session)
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _change_set = scanner.scan("pg_e2e_project", str(tmp_path))
    db_session.flush()

    registry = build_default_registry()
    parse_results = {}
    for relative_path, record in manifest.files.items():
        result = registry.parse_file(tmp_path / relative_path, record)
        if result is not None:
            parse_results[relative_path] = result

    graph = IRBuilder().build("pg_e2e_project", manifest.files, parse_results)

    ir_repo = IRGraphRepository(db_session)
    ir_repo.save(graph)
    db_session.flush()

    # Reload everything from Postgres — not the in-memory objects — to
    # prove the round trip through the database is correct end to end.
    reloaded_manifest = manifest_store.get_latest_manifest("pg_e2e_project")
    reloaded_graph = ir_repo.load("pg_e2e_project")

    assert reloaded_manifest is not None
    assert set(reloaded_manifest.files.keys()) == {"RTL/i2c_master.sv", "RTL/i2c_top.sv"}

    top_module = next(n for n in reloaded_graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "i2c_top")
    master_module = next(n for n in reloaded_graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "i2c_master")

    inst_edges = [e for e in reloaded_graph.edges_from(top_module.id) if e.kind == IREdgeKind.INSTANTIATES]
    assert len(inst_edges) == 1
    assert inst_edges[0].resolved is True
    assert inst_edges[0].target_id == master_module.id


def test_rescanning_after_file_deletion_prunes_postgres_state(tmp_path: Path, db_session):
    _write(tmp_path, "RTL/a.sv", "module a; endmodule\n")
    _write(tmp_path, "RTL/b.sv", "module b; endmodule\n")

    manifest_store = PostgresManifestStore(db_session)
    scanner = Scanner(manifest_store=manifest_store)
    registry = build_default_registry()
    ir_repo = IRGraphRepository(db_session)

    def run_pipeline():
        manifest, _ = scanner.scan("pg_prune_project", str(tmp_path))
        db_session.flush()
        parse_results = {
            rel: r for rel, record in manifest.files.items()
            if (r := registry.parse_file(tmp_path / rel, record)) is not None
        }
        graph = IRBuilder().build("pg_prune_project", manifest.files, parse_results)
        ir_repo.save(graph)
        db_session.flush()

    run_pipeline()
    (tmp_path / "RTL" / "b.sv").unlink()
    run_pipeline()

    reloaded_manifest = manifest_store.get_latest_manifest("pg_prune_project")
    reloaded_graph = ir_repo.load("pg_prune_project")

    assert set(reloaded_manifest.files.keys()) == {"RTL/a.sv"}
    module_names = {n.name for n in reloaded_graph.nodes_of_kind(IRNodeKind.SV_MODULE)}
    assert module_names == {"a"}
