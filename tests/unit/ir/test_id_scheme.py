from semrag.ingestion.ir import id_scheme


def test_sv_module_id_deterministic():
    a = id_scheme.sv_module_id("proj1", "RTL/i2c_master.sv", "i2c_master")
    b = id_scheme.sv_module_id("proj1", "RTL/i2c_master.sv", "i2c_master")
    assert a == b


def test_sv_module_id_differs_by_module_name():
    a = id_scheme.sv_module_id("proj1", "RTL/multi.sv", "i2c_master")
    b = id_scheme.sv_module_id("proj1", "RTL/multi.sv", "i2c_slave")
    assert a != b


def test_edge_id_deterministic_and_distinguishes_targets():
    e1 = id_scheme.edge_id("includes", "src::foo", "defines.svh")
    e2 = id_scheme.edge_id("includes", "src::foo", "defines.svh")
    e3 = id_scheme.edge_id("includes", "src::foo", "other.svh")
    assert e1 == e2
    assert e1 != e3
