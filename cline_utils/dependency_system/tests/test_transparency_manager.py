import os
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from cline_utils.dependency_system.io.transparency_manager import (
    TransparencyManager,
    get_transparency_manager,
    read_file_transparently,
    overlay_connection_maps,
    _coerce_line,
    _find_source_line,
    extract_connection_map_metadata,
    _parse_connection_map_text,
    _adjust_connection_record_lines,
    TransparencyLock
)

def test_coerce_line():
    assert _coerce_line("10") == 10
    assert _coerce_line(15) == 15
    assert _coerce_line(None) is None
    assert _coerce_line("notanumber") is None

def test_get_transparency_manager(tmp_path):
    registry_file = tmp_path / "temp_registry.json"
    from cline_utils.dependency_system.io.transparency_manager import TransparencyManager
    
    orig_defaults = TransparencyManager.__init__.__defaults__
    try:
        TransparencyManager.__init__.__defaults__ = (str(registry_file),)
        with patch("cline_utils.dependency_system.io.transparency_manager._manager_instance", None):
            manager1 = get_transparency_manager()
            manager2 = get_transparency_manager()
            assert manager1 is manager2
            assert manager1.registry_path == str(registry_file)
    finally:
        TransparencyManager.__init__.__defaults__ = orig_defaults

def test_transparency_manager_init(tmp_path):
    registry_file = tmp_path / "registry.json"
    manager = TransparencyManager(str(registry_file))
    assert manager.registry_path == str(registry_file)
    assert manager._registry["files"] == {}

def test_transparency_manager_update_and_get(tmp_path):
    registry_file = tmp_path / "registry.json"
    manager = TransparencyManager(str(registry_file))

    file_path = str(tmp_path / "test.py")
    content = "def test():\n    pass\n"

    manager.update_file_metadata(
        file_path=file_path,
        sections={"FUNC": [1, 2]},
        content=content
    )

    meta = manager.get_file_metadata(file_path)
    assert meta is not None
    assert "sections" in meta
    assert meta["sections"] == {"FUNC": [1, 2]}
    assert meta["total_lines"] == 2

def test_read_file_transparently_no_drift(tmp_path):
    registry_file = tmp_path / "transparency_registry.json"
    file_path = tmp_path / "test.py"
    content = "def hello():\n    print('hello')\n"
    file_path.write_text(content, encoding="utf-8")

    with patch('cline_utils.dependency_system.io.transparency_manager.get_transparency_manager') as mock_get:
        manager = TransparencyManager(str(registry_file))
        manager.update_file_metadata(str(file_path), {"hello": [1, 2]}, content)
        mock_get.return_value = manager

        read_content, metadata = read_file_transparently(str(file_path))

        assert read_content == content
        assert metadata is not None
        assert metadata["sections"] == {"hello": [1, 2]}

def test_overlay_connection_maps():
    content = "def hello():\n    print('hello')\n"
    metadata = {
        "connection_map_lines": [
            {
                "line": 1,
                "content": "# CONNECTION_MAP: module(target) {test} --- hello [AUTO]\n"
            }
        ]
    }

    overlaid = overlay_connection_maps(content, metadata)
    assert "CONNECTION_MAP: module(target) {test} --- hello [AUTO]" in overlaid
    assert "def hello()" in overlaid

def test_extract_connection_map_metadata():
    content = """# CONNECTION_MAP: module(target) {test} --- my_func [AUTO]
def my_func():
    pass
"""
    metadata = extract_connection_map_metadata(content)
    assert len(metadata) == 1
    assert metadata[0]["source_symbol"] == "my_func"
    assert metadata[0]["target_key"] == "module"
    assert metadata[0]["target_symbol"] == "target"
    assert metadata[0]["dep_char"] == "test"

def test_transparency_lock(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock = TransparencyLock(str(lock_path))

    lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            lock2 = TransparencyLock(str(lock_path), timeout=0.1)
            lock2.acquire()
    finally:
        lock.release()

def test_transparency_lock_context_manager(tmp_path):
    lock_path = tmp_path / "test2.lock"

    with TransparencyLock(str(lock_path)):
        with pytest.raises(TimeoutError):
            lock2 = TransparencyLock(str(lock_path), timeout=0.1)
            lock2.acquire()

def test_find_source_line():
    lines = [
        "# some comment",
        "# CONNECTION_MAP: module(target) {test} --- my_func [AUTO]",
        "def my_func():",
        "    pass"
    ]
    # It should find it on line 3 (index 2 + 1)
    assert _find_source_line(lines, 1, "my_func") == 3

def test_adjust_connection_record_lines():
    records = [
        {
            "source_symbol": "func_a",
            "source_line": 10,
            "target_symbol": "target_a",
            "target_line": 20
        }
    ]
    removed_line_numbers = [5]
    adjusted = _adjust_connection_record_lines(records, removed_line_numbers)

    assert adjusted[0]["source_line"] == 9
    assert adjusted[0]["target_line"] == 20

def test_transparency_manager_invalidate_entry(tmp_path):
    registry_file = tmp_path / "registry.json"
    manager = TransparencyManager(str(registry_file))

    file_path = str(tmp_path / "test.py")
    content = "def test():\n    pass\n"

    manager.update_file_metadata(
        file_path=file_path,
        sections={"FUNC": [1, 2]},
        content=content
    )

    assert manager.get_file_metadata(file_path) is not None
    manager.invalidate_entry(file_path)
    assert manager.get_file_metadata(file_path) is None
