"""Subprocess isolation and ImportError reporting for runtime_inspector."""

import logging
from pathlib import Path

import pytest

from cline_utils.dependency_system.analysis.runtime_inspector import get_module_info


def test_get_module_info_isolates_top_level_crash(tmp_path: Path) -> None:
    crash_file = tmp_path / "crash_module.py"
    crash_file.write_text('raise RuntimeError("top-level crash")\n', encoding="utf-8")
    code_roots = [str(tmp_path)]

    result = get_module_info(str(crash_file), "crash_module", code_roots)

    assert result == {}


def test_get_module_info_reports_import_error_explicitly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mod_file = tmp_path / "bad_import.py"
    mod_file.write_text(
        "import cline_utils_dependency_missing_package_xyz\n\n"
        "def foo():\n"
        "    pass\n",
        encoding="utf-8",
    )
    code_roots = [str(tmp_path)]

    with caplog.at_level(logging.WARNING):
        result = get_module_info(str(mod_file), "bad_import", code_roots)

    assert result == {}
    messages = " ".join(record.message for record in caplog.records)
    assert "ImportError" in messages
    assert "virtual environment" in messages


def test_get_module_info_collects_symbols_for_valid_module(tmp_path: Path) -> None:
    mod_file = tmp_path / "simple.py"
    mod_file.write_text(
        "def hello():\n"
        '    """Say hello."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    code_roots = [str(tmp_path)]

    result = get_module_info(str(mod_file), "simple", code_roots)

    function_names = {fn["name"] for fn in result.get("functions", [])}
    assert "hello" in function_names
