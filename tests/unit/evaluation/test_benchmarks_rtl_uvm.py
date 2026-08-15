"""
Validation tests for the curated rtl_uvm retrieval benchmark set.

The rtl_uvm golden queries are grounded in a deterministic offline
scan -> parse -> IR -> chunk run of the I2C UVM corpus (467 chunks).
These tests enforce well-formedness, project consistency, chunk-ID
existence against the authoritative chunk set, and serialization.

The authoritative chunk-ID universe is verified independently by
running the offline chunk pipeline; this file only requires that every
referenced ID match the deterministic chunk-ID grammar and belong to
the rtl_uvm project. The full cross-check against the 467-chunk dump
lives in the RB-04 curation script (see CHANGELOG).
"""

from __future__ import annotations

import re

from semrag.evaluation.benchmarks_rtl_uvm import BENCHMARKS_RTL_UVM, PROJECT_ID, get_rtl_uvm_benchmarks
from semrag.evaluation.models import ExpectedIntent, GoldenQuery, RepoArea

CHUNK_ID_RE = re.compile(
    r"^rtl_uvm::[^:]+::[a-z_]+::[A-Za-z0-9_:]+$"
)


class TestRtlUvmBenchmarkData:
    """Curated rtl_uvm benchmarks must be well-formed."""

    def test_query_count_in_range(self):
        assert 30 <= len(get_rtl_uvm_benchmarks()) <= 50

    def test_all_benchmarks_have_required_fields(self):
        for bm in get_rtl_uvm_benchmarks():
            assert bm.query_id, f"Missing query_id in {bm}"
            assert bm.query_text, f"Missing query_text in {bm}"
            assert bm.description, f"Missing description in {bm.query_id}"
            assert bm.expected_intent in ExpectedIntent, f"Invalid intent in {bm.query_id}"
            assert bm.repo_area in RepoArea, f"Invalid repo area in {bm.query_id}"

    def test_all_benchmarks_target_rtl_uvm(self):
        for bm in get_rtl_uvm_benchmarks():
            assert bm.project_id == PROJECT_ID, f"{bm.query_id} targets {bm.project_id!r}"

    def test_all_benchmarks_have_unique_ids(self):
        ids = [bm.query_id for bm in get_rtl_uvm_benchmarks()]
        assert len(ids) == len(set(ids)), "rtl_uvm benchmark query IDs must be unique"

    def test_all_benchmarks_have_non_empty_relevant_chunks(self):
        for bm in get_rtl_uvm_benchmarks():
            assert bm.relevant_chunk_ids, f"No relevant_chunk_ids in {bm.query_id}"

    def test_relevant_chunk_ids_are_well_formed(self):
        for bm in get_rtl_uvm_benchmarks():
            for cid in bm.relevant_chunk_ids:
                assert CHUNK_ID_RE.match(cid), f"Malformed chunk id {cid!r} in {bm.query_id}"
                assert cid.startswith(f"{PROJECT_ID}::"), (
                    f"{cid!r} in {bm.query_id} does not belong to {PROJECT_ID}"
                )

    def test_no_duplicate_chunk_ids_within_a_query(self):
        for bm in get_rtl_uvm_benchmarks():
            assert len(bm.relevant_chunk_ids) == len(set(bm.relevant_chunk_ids))

    def test_no_semrag_repo_paths_leak_into_rtl_uvm_benchmarks(self):
        for bm in get_rtl_uvm_benchmarks():
            for path in bm.primary_files | bm.secondary_files:
                assert not path.startswith("src/semrag/"), (
                    f"{bm.query_id} references semrag repo path {path!r}"
                )

    def test_intent_coverage(self):
        """RTL and DEBUG intents must be represented (the reason this
        corpus-specific set exists); every intent used must be one of
        the curated values."""
        covered = {bm.expected_intent for bm in get_rtl_uvm_benchmarks()}
        assert ExpectedIntent.RTL in covered
        assert ExpectedIntent.DEBUG in covered
        assert ExpectedIntent.VERIFICATION in covered
        allowed = {ExpectedIntent.RTL, ExpectedIntent.DEBUG,
                   ExpectedIntent.VERIFICATION, ExpectedIntent.CONFIG,
                   ExpectedIntent.GENERAL}
        assert covered <= allowed, covered

    def test_category_distribution(self):
        """Curation targets: nav 5-8, des 5-8, dbg 5-8, asr 4-6, bus
        4-6, uvm 4-6, log 3-5, term 3-5."""
        counts: dict[str, int] = {}
        for bm in get_rtl_uvm_benchmarks():
            prefix = bm.query_id.split("_")[1]
            counts[prefix] = counts.get(prefix, 0) + 1
        bounds = {
            "nav": (5, 8), "des": (5, 8), "dbg": (5, 8),
            "asr": (4, 6), "bus": (4, 6), "uvm": (4, 6),
            "log": (3, 5), "term": (3, 5),
        }
        for prefix, (lo, hi) in bounds.items():
            assert lo <= counts.get(prefix, 0) <= hi, (
                f"category {prefix} has {counts.get(prefix, 0)} queries, want {lo}-{hi}"
            )

    def test_serialization_roundtrip(self):
        """GoldenQuery serialization/deserialization must preserve the
        curated ground truth exactly."""
        for bm in get_rtl_uvm_benchmarks():
            dumped = bm.model_dump_json()
            restored = GoldenQuery.model_validate_json(dumped)
            assert restored.model_dump() == bm.model_dump(), f"Roundtrip mismatch for {bm.query_id}"
            assert restored.relevant_chunk_ids == bm.relevant_chunk_ids


class TestRtlUvmGoldenQueryModel:
    """GoldenQuery schema interplay for the rtl_uvm set."""

    def test_returns_a_copy(self):
        first = get_rtl_uvm_benchmarks()
        second = get_rtl_uvm_benchmarks()
        assert first == second
        assert first is not second
        first.append(first[0])
        assert len(get_rtl_uvm_benchmarks()) < len(first)

    def test_module_constant_matches_accessor(self):
        assert BENCHMARKS_RTL_UVM == get_rtl_uvm_benchmarks()

    def test_every_query_has_an_id_referencing_a_real_corpus_path(self):
        corpus_dirs = {
            "README.md",
            "rtl/i2c_master.sv",
            "sim/log/write1_test_log.log",
            "sim/log/write2_test_log.log",
            "sim/coverage/assertions.txt",
            "sim/coverage/code_coverage.txt",
            "sim/coverage/fun_coverage.txt",
            "tb/i2c_master_top.sv",
            "tb/i2c_master_if.sv",
            "tb/i2c_master_param_pkg.sv",
            "tb/i2c_master_pkg.sv",
            "tb/i2c_master_sequence_item.svh",
            "tb/i2c_master_env_config.svh",
            "tb/i2c_master_environment.svh",
            "tb/i2c_master_scoreboard.svh",
            "tb/i2c_master_coverage_collector.svh",
            "tb/i2c_master_agent/i2c_master_agent.svh",
            "tb/i2c_master_agent/i2c_master_agent_config.svh",
            "tb/i2c_master_agent/i2c_master_agent_pkg.sv",
            "tb/i2c_master_agent/i2c_master_driver.svh",
            "tb/i2c_master_agent/i2c_master_monitor.svh",
            "tb/i2c_master_agent/i2c_master_sequencer.svh",
            "tb/i2c_master_agent/i2c_master_stat_monitor.svh",
            "tb/i2c_master_seq_lib/i2c_master_base_seq.svh",
            "tb/i2c_master_seq_lib/i2c_master_seq_pkg.sv",
            "tb/i2c_master_seq_lib/i2c_master_write1_seq.svh",
            "tb/i2c_master_seq_lib/i2c_master_write2_seq.svh",
            "tb/i2c_master_test_lib/i2c_master_base_test.svh",
            "tb/i2c_master_test_lib/i2c_master_test_pkg.sv",
            "tb/i2c_master_test_lib/i2c_master_write1_test.svh",
            "tb/i2c_master_test_lib/i2c_master_write2_test.svh",
        }
        for bm in get_rtl_uvm_benchmarks():
            for path in bm.primary_files:
                assert path in corpus_dirs, f"{bm.query_id} primary file {path!r} not in corpus"