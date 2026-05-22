
import os
import shutil
import pytest
from pathlib import Path
from cline_utils.dependency_system.core import _migrate_state_files, resolve_state_path

def test_state_file_migration(tmp_path: Path):
    """
    Test that _migrate_state_files correctly moves .json and .json.lock files
    from a directory to its 'state/' subdirectory.
    """
    core_dir = tmp_path / "mock_core"
    core_dir.mkdir()
    state_dir = core_dir / "state"
    
    # 1. Create mock state files in the "core" directory
    file1 = core_dir / "test1.json"
    file2 = core_dir / "test2.json.lock"
    file3 = core_dir / "not_state.py"
    
    file1.write_text('{"key": "value"}', encoding="utf-8")
    file2.write_text("lock content", encoding="utf-8")
    file3.write_text("print('hi')", encoding="utf-8")
    
    # 2. Run migration
    _migrate_state_files(str(core_dir))
    
    # 3. Assertions
    assert state_dir.exists()
    assert (state_dir / "test1.json").exists()
    assert (state_dir / "test2.json.lock").exists()
    assert not file1.exists()
    assert not file2.exists()
    
    # Non-state files should NOT be moved
    assert file3.exists()
    assert not (state_dir / "not_state.py").exists()

def test_resolve_state_path_fallback(tmp_path: Path):
    """
    Test that resolve_state_path prefers 'state/' but falls back to core dir.
    """
    core_dir = tmp_path / "mock_core_fallback"
    core_dir.mkdir()
    state_dir = core_dir / "state"
    state_dir.mkdir()
    
    filename = "persistent_state.json"
    legacy_file = core_dir / filename
    state_file = state_dir / filename
    
    # Case 1: File only in core dir (fallback)
    legacy_file.write_text("legacy", encoding="utf-8")
    resolved = resolve_state_path(filename, str(core_dir))
    assert resolved == str(legacy_file)
    
    # Case 2: File in state dir (preferred)
    state_file.write_text("new state", encoding="utf-8")
    resolved = resolve_state_path(filename, str(core_dir))
    assert resolved == str(state_file)
    
    # Case 3: File in neither (returns expected state path)
    missing_file = "missing.json"
    resolved = resolve_state_path(missing_file, str(core_dir))
    assert resolved == str(state_dir / missing_file)

def test_migration_is_silent_on_second_run(tmp_path: Path):
    """
    Test that running migration twice doesn't cause errors or redundant actions.
    """
    core_dir = tmp_path / "mock_core_twice"
    core_dir.mkdir()
    state_dir = core_dir / "state"
    state_dir.mkdir()
    
    # Run once on empty dir
    _migrate_state_files(str(core_dir))
    assert state_dir.exists()
    
    # Add a file and run again
    new_file = core_dir / "late_arrival.json"
    new_file.write_text("{}", encoding="utf-8")
    _migrate_state_files(str(core_dir))
    
    assert (state_dir / "late_arrival.json").exists()
    assert not new_file.exists()
