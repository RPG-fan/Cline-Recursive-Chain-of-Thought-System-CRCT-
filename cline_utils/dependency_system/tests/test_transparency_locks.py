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
