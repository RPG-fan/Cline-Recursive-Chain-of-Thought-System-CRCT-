import json
from unittest.mock import patch

from code_analysis.reporting.json_exporter import export_json

def test_export_json_success(tmp_path):
    output_path = tmp_path / "subdir" / "output.json"
    issues = [{"id": 1, "msg": "test issue"}]
    unused = [{"id": 2, "msg": "unused item"}]

    result = export_json(issues, unused, str(output_path))
    assert result is True

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["issues"] == issues
    assert data["unused"] == unused

def test_export_json_error_path(tmp_path, capsys):
    output_path = tmp_path / "output.json"
    issues = []
    unused = []

    export_json(issues, unused, str(output_path))

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["issues"] == []
    assert data["unused"] == []

def test_export_json_nested_dir(tmp_path):
    nested_dir = tmp_path / "deep" / "nested" / "dir"
    output_path = nested_dir / "output.json"

    issues = [{"issue": "test"}]
    unused = [{"unused": "test"}]

    export_json(issues, unused, str(output_path))

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["issues"] == issues
    assert data["unused"] == unused

def test_export_json_exception_handling(capsys, tmp_path):
    output_path = tmp_path / "dummy.json"

    # Patch 'open' to raise an exception
    with patch("code_analysis.reporting.json_exporter.open", side_effect=PermissionError("Permission denied")):
        export_json([], [], str(output_path))

    captured = capsys.readouterr()
    assert "[report] Failed writing JSON report:" in captured.out
    assert "Permission denied" in captured.out
