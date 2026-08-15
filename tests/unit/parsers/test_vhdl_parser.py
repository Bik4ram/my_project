from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.vhdl.parser import VhdlParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=f"RTL/{path.name}",
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=FileCategory.RTL_VHDL,
        extension=path.suffix,
        is_binary=False,
    )


def test_extracts_entity_ports_and_generics(tmp_path: Path):
    f = tmp_path / "i2c_master.vhd"
    f.write_text(
        """
        entity i2c_master is
          generic (
            CLK_FREQ : integer := 100000000;
            I2C_FREQ : integer := 400000
          );
          port (
            clk, rst_n : in  std_logic;
            sda        : out std_logic;
            data_in    : in  std_logic_vector(7 downto 0)
          );
        end entity i2c_master;

        architecture rtl of i2c_master is
        begin
        end architecture rtl;
        """
    )
    result = VhdlParser().parse(f, _record(f))
    assert result.success is True

    entities = result.payload["entities"]
    assert len(entities) == 1
    entity = entities[0]
    assert entity["name"] == "i2c_master"

    port_names = {p["name"]: p for p in entity["ports"]}
    assert port_names["clk"]["direction"] == "in"
    assert port_names["sda"]["direction"] == "out"
    assert port_names["data_in"]["data_type"] == "std_logic_vector(7 downto 0)"

    generics = {g["name"]: g for g in entity["generics"]}
    assert generics["CLK_FREQ"]["default_value"] == "100000000"

    architectures = result.payload["architectures"]
    assert architectures[0]["entity_name"] == "i2c_master"
    assert architectures[0]["name"] == "rtl"


def test_entity_without_generics_still_parses(tmp_path: Path):
    f = tmp_path / "simple.vhd"
    f.write_text(
        """
        entity simple is
          port (
            clk : in std_logic
          );
        end entity simple;
        """
    )
    result = VhdlParser().parse(f, _record(f))
    entity = result.payload["entities"][0]
    assert entity["generics"] == []
    assert entity["ports"][0]["name"] == "clk"


def test_multiple_entities_in_one_file(tmp_path: Path):
    f = tmp_path / "multi.vhd"
    f.write_text(
        """
        entity ent_a is
          port (a : in std_logic);
        end entity ent_a;

        entity ent_b is
          port (b : out std_logic);
        end entity ent_b;
        """
    )
    result = VhdlParser().parse(f, _record(f))
    names = {e["name"] for e in result.payload["entities"]}
    assert names == {"ent_a", "ent_b"}
