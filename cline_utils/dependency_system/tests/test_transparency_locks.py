import os
import json
from pathlib import Path
import pytest
from cline_utils.dependency_system.io.transparency_manager import TransparencyManager
from cline_utils.dependency_system.utils.calculate_hash import calculate_content_hash
from cline_utils.dependency_system.io.transparency_manager import normalize_path

def test_transparency_locks_and_realignment(tmp_path):
    """
    Tests the entire flow of:
    1. Registering file metadata.
    2. Simulating unrecoverable drift (calling check_drift and recover_alignment), which should LOCK the entry.
    3. Verifying that get_file_metadata returns None for locked entries, while get_raw_file_metadata returns the entry.
    4. Realigning locked entries using a mock project symbol map.
    5. Verifying that the lock is cleared and ranges are updated correctly!
    """
    registry_file = tmp_path / "transparency_registry.json"
    manager = TransparencyManager(str(registry_file))

    # Create a simulated markdown file
    md_file = tmp_path / "test_doc.md"
    content = (
        "---TAGS_START---\n"
        "tags: test, lock\n"
        "---TAGS_END---\n"
        "## Context\n"
        "This is context content.\n"
        "## Overview\n"
        "This is overview content.\n"
        "## Details\n"
        "This is details content.\n"
        "## References\n"
        "This is references content.\n"
    )
    md_file.write_text(content, encoding="utf-8")

    # Initial metadata registration
    sections = {
        "TAGS": {
            "content": "tags: test, lock",
            "start_line": 1
        },
        "CONTEXT": {
            "range": [4, 5],
            "anchors": ["## Context", "This is context content."]
        },
        "OVERVIEW": {
            "range": [6, 7],
            "anchors": ["## Overview", "This is overview content."]
        },
        "DETAILS": {
            "range": [8, 9],
            "anchors": ["## Details", "This is details content."]
        },
        "REFERENCES": {
            "range": [10, 11],
            "anchors": ["## References", "This is references content."]
        }
    }
    manager.update_file_metadata(str(md_file), sections, content)

    # 1. Verify get_file_metadata is returned and not locked
    meta = manager.get_file_metadata(str(md_file))
    assert meta is not None
    assert "locked" not in meta
    assert meta["sections"]["CONTEXT"]["range"] == [4, 5]

    # 2. Simulate drift and trigger lock
    # Let's mismatch the anchors to cause unrecoverable drift and trigger the lock
    drifted_content = (
        "---TAGS_START---\n"
        "tags: test, lock\n"
        "---TAGS_END---\n"
        "## Context Drifted\n"
        "\n\n\n\n\n\n\n\n\n"
        "This is context content drifted.\n"
        "## Overview Drifted\n"
        "This is overview content drifted.\n"
        "## Details Drifted\n"
        "This is details content drifted.\n"
        "## References Drifted\n"
        "This is references content drifted.\n"
    )
    md_file.write_text(drifted_content, encoding="utf-8")

    # Call check_drift and simulate read transparently / recover alignment failing
    is_drifted = manager.check_drift(str(md_file), drifted_content)
    assert is_drifted is True

    # Call recover_alignment (it will fail to resolve because anchors drifted too far and we will trigger lock)
    # Let's verify that calling recover_alignment locks the entry instead of invalidating it!
    recovered_meta = manager.recover_alignment(str(md_file), drifted_content)
    assert recovered_meta is None

    # Verify locked entry exists in raw but is hidden from get_file_metadata
    meta_locked = manager.get_file_metadata(str(md_file))
    assert meta_locked is None  # Hidden!

    raw_meta = manager.get_raw_file_metadata(str(md_file))
    assert raw_meta is not None
    assert raw_meta.get("locked") is True

    # 3. Realign using a mock project symbol map
    # We will simulate the symbol analyzer regenerating headers and line numbers:
    # ---TAGS_START--- is at 1
    # ## Context is at 4
    # ## Overview is at 15
    # ## Details is at 17
    # ## References is at 19
    norm_path = normalize_path(str(md_file))
    mock_symbol_map = {
        norm_path: {
            "headers": [
                {"name": "## Context", "line": 4},
                {"name": "## Overview", "line": 15},
                {"name": "## Details", "line": 17},
                {"name": "## References", "line": 19}
            ]
        }
    }

    # Call realign_locked_entries
    realigned_count = manager.realign_locked_entries(mock_symbol_map)
    assert realigned_count == 1

    # Verify that the lock is released and the metadata is fully updated with new range lines!
    updated_meta = manager.get_file_metadata(str(md_file))
    assert updated_meta is not None
    assert "locked" not in updated_meta

    # Check updated ranges
    # CONTEXT: range starts at 4, ends at 14 (Overview is at 15)
    assert updated_meta["sections"]["CONTEXT"]["range"] == [4, 14]
    # OVERVIEW: range starts at 15, ends at 16 (Details is at 17)
    assert updated_meta["sections"]["OVERVIEW"]["range"] == [15, 16]
    # DETAILS: range starts at 17, ends at 18 (References is at 19)
    assert updated_meta["sections"]["DETAILS"]["range"] == [17, 18]
    # REFERENCES: range starts at 19, ends at 20 (total line count)
    assert updated_meta["sections"]["REFERENCES"]["range"] == [19, 20]

    # Verify end anchors have been updated to match the new file contents at their respective end lines
    lines = drifted_content.splitlines()
    assert updated_meta["sections"]["CONTEXT"]["anchors"][1] == lines[13].strip()  # "This is context content drifted."
    assert updated_meta["sections"]["OVERVIEW"]["anchors"][1] == lines[15].strip()  # "This is overview content drifted."


def test_nested_boundary_restoration(tmp_path):
    """
    Tests that deeply nested transparency section structures (e.g. child inside parent)
    can be virtualized (remove_markers) and restored (restore_markers)
    without shift offset corruption, and the output matches original content exactly.
    """
    registry_file = tmp_path / "transparency_registry.json"
    manager = TransparencyManager(str(registry_file))

    # Create a simulated document with nested sections
    doc_file = tmp_path / "test_nested_doc.md"
    original_content = (
        "---PARENT_START---\n"
        "This is parent context start.\n"
        "---CHILD_START---\n"
        "This is nested child context content.\n"
        "---CHILD_END---\n"
        "This is parent context end.\n"
        "---PARENT_END---\n"
    )
    doc_file.write_text(original_content, encoding="utf-8")

    # 1. Virtualize: remove markers
    success_remove = manager.remove_markers(str(doc_file))
    assert success_remove is True

    # Check clean content: markers should be completely stripped
    clean_content = doc_file.read_text(encoding="utf-8")
    expected_clean = (
        "This is parent context start.\n"
        "This is nested child context content.\n"
        "This is parent context end.\n"
    )
    assert clean_content == expected_clean

    # Check metadata: sections should be registered
    meta = manager.get_file_metadata(str(doc_file))
    assert meta is not None
    assert "PARENT" in meta["sections"]
    assert "CHILD" in meta["sections"]

    # 2. Restore: restore markers
    success_restore = manager.restore_markers(str(doc_file))
    assert success_restore is True

    # Restored content should match original content exactly!
    restored_content = doc_file.read_text(encoding="utf-8")
    assert restored_content == original_content


def test_timestamp_synchronization(tmp_path):
    """
    Tests that the recorded last_modified timestamp in the registry matches
    the actual physical disk modification time (mtime) exactly to microsecond precision.
    We test both remove_markers (which uses update_file_metadata) and
    virtualize_connection_maps (which uses _write_connection_map_metadata).
    """
    registry_file = tmp_path / "transparency_registry.json"
    manager = TransparencyManager(str(registry_file))

    # Test Case 1: remove_markers (update_file_metadata)
    doc_file = tmp_path / "test_sync_doc.md"
    original_content = (
        "---SECTION_START---\n"
        "This is context content.\n"
        "---SECTION_END---\n"
    )
    doc_file.write_text(original_content, encoding="utf-8")

    success_remove = manager.remove_markers(str(doc_file))
    assert success_remove is True

    # Read registry entry
    meta = manager.get_file_metadata(str(doc_file))
    assert meta is not None
    recorded_mtime = meta.get("last_modified")
    actual_mtime = os.path.getmtime(str(doc_file))

    assert recorded_mtime is not None
    # Verify exact microsecond precision match
    assert recorded_mtime == actual_mtime

    # Test Case 2: virtualize_connection_maps (_write_connection_map_metadata)
    py_file = tmp_path / "test_sync_py.py"
    py_content = (
        "def hello():\n"
        "    # CONNECTION_MAP: none --- hello [AUTO]\n"
        "    pass\n"
    )
    py_file.write_text(py_content, encoding="utf-8")

    success_virtualize = manager.virtualize_connection_maps(str(py_file))
    assert success_virtualize is True

    meta_py = manager.get_file_metadata(str(py_file))
    assert meta_py is not None
    recorded_mtime_py = meta_py.get("last_modified")
    actual_mtime_py = os.path.getmtime(str(py_file))

    assert recorded_mtime_py is not None
    assert recorded_mtime_py == actual_mtime_py


