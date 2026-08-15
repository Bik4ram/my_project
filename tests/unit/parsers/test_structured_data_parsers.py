from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.structured_data.parsers import (
    CsvParser,
    JsonParser,
    XmlParser,
    YamlParser,
)
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path, category: FileCategory) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=path.name,
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=category,
        extension=path.suffix,
        is_binary=False,
    )


def test_json_object_shape(tmp_path: Path):
    f = tmp_path / "config.json"
    f.write_text('{"clk_freq": 100000000, "i2c_freq": 400000, "enable_coverage": true}')
    result = JsonParser().parse(f, _record(f, FileCategory.JSON))
    assert result.success is True
    assert result.payload["root_type"] == "object"
    assert set(result.payload["top_level_keys"]) == {"clk_freq", "i2c_freq", "enable_coverage"}


def test_json_array_shape(tmp_path: Path):
    f = tmp_path / "tests.json"
    f.write_text('[{"name": "t1"}, {"name": "t2"}, {"name": "t3"}]')
    result = JsonParser().parse(f, _record(f, FileCategory.JSON))
    assert result.payload["root_type"] == "array"
    assert result.payload["array_length"] == 3


def test_json_invalid_fails_gracefully(tmp_path: Path):
    f = tmp_path / "broken.json"
    f.write_text("{not valid json")
    result = JsonParser().parse(f, _record(f, FileCategory.JSON))
    assert result.success is False
    assert result.has_errors is True


def test_yaml_object_shape(tmp_path: Path):
    f = tmp_path / "config.yaml"
    f.write_text("clk_freq: 100000000\ni2c_freq: 400000\n")
    result = YamlParser().parse(f, _record(f, FileCategory.YAML))
    assert result.success is True
    assert result.payload["root_type"] == "object"
    assert set(result.payload["top_level_keys"]) == {"clk_freq", "i2c_freq"}


def test_csv_extracts_columns_and_sample_rows(tmp_path: Path):
    f = tmp_path / "register_map.csv"
    f.write_text(
        "Address,Register Name,Width\n"
        "0x00,CTRL,8\n"
        "0x04,STATUS,8\n"
        "0x08,DATA,32\n"
    )
    result = CsvParser().parse(f, _record(f, FileCategory.CSV))
    assert result.success is True
    assert result.payload["columns"] == ["Address", "Register Name", "Width"]
    assert result.payload["row_count"] == 3
    assert result.payload["sample_rows"][0]["Register Name"] == "CTRL"


def test_xml_tag_inventory_and_depth(tmp_path: Path):
    f = tmp_path / "waveform_config.xml"
    f.write_text(
        "<config><signals><signal name='clk'/><signal name='rst_n'/></signals></config>"
    )
    result = XmlParser().parse(f, _record(f, FileCategory.XML))
    assert result.success is True
    assert result.payload["root_tag"] == "config"
    assert set(result.payload["distinct_tags"]) == {"config", "signals", "signal"}
    assert result.payload["max_depth"] == 3


def test_xml_invalid_fails_gracefully(tmp_path: Path):
    f = tmp_path / "broken.xml"
    f.write_text("<config><unclosed>")
    result = XmlParser().parse(f, _record(f, FileCategory.XML))
    assert result.success is False
