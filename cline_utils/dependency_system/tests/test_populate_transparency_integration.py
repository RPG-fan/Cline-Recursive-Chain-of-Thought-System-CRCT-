
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from cline_utils.dependency_system.utils.populate_comments import process_file
from cline_utils.dependency_system.io.transparency_manager import get_transparency_manager, TransparencyManager
from cline_utils.dependency_system.core.key_manager import KeyInfo

def _ki(key: str, path: Path) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=str(path),
        parent_path=str(path.parent),
        tier=int(key[0]) if key[0].isdigit() else 1,
        is_directory=False
    )

def test_process_file_identifies_virtualized_maps_as_refreshed(tmp_path: Path):
    """
    Verify that process_file correctly identifies CONNECTION_MAPs stored in the 
    transparency registry as 'refreshed' rather than 'new'.
    """
    project_root = tmp_path
    source_path = project_root / "source.py"
    
    # 1. Setup a clean file on disk (no CONNECTION_MAPs)
    source_path.write_text("def source_func():\n    pass\n", encoding="utf-8")
    
    # 2. Setup transparency registry with a virtualized CONNECTION_MAP for this file
    registry_path = project_root / "transparency_registry.json"
    manager = TransparencyManager(str(registry_path))
    
    # Mock the global manager instance for the duration of the test
    import cline_utils.dependency_system.io.transparency_manager as tm_mod
    original_manager = tm_mod._manager_instance
    tm_mod._manager_instance = manager
    
    try:
        manager._write_connection_map_metadata(
            str(source_path),
            "def source_func():\n    pass\n",
            [
                {
                    "source_symbol": "source_func",
                    "source_line": 1,
                    "target_key": "1B",
                    "target_symbol": "target_func",
                    "target_line": 10,
                    "dep_char": "d",
                }
            ],
            [
                {
                    "line": 1,
                    "content": "# --- CONNECTION_MAP: 1B(target_func:10) {d} --- source_func [AUTO]",
                }
            ],
        )
        
        # 3. Prepare data for process_file
        source_ki = _ki("1A", source_path)
        target_ki = _ki("1B", project_root / "target.py")
        
        symbol_data = {
            "functions": [{"name": "source_func", "line": 1}],
            "classes": [],
        }
        
        # grid_row for 1A where it depends on 1B
        # Assuming decompress("d") works correctly for this mock setup
        # 'd' in first col (1B) if key_info_list is [target_ki, source_ki]
        key_info_list = [target_ki, source_ki]
        grid_row = "do" # 'd' for 1B, 'o' for 1A (self)
        
        # 4. Run process_file (dry_run=True to avoid disk writes in this test)
        result = process_file(
            file_path=source_path,
            source_key="1A",
            key_info_list=key_info_list,
            grid_row=grid_row,
            tracker_ref="mini_tracker.md",
            entry_from=[],
            exits_to=[],
            symbol_data=symbol_data,
            project_root=project_root,
            dry_run=True,
            verbose=True,
        )
        
        # 5. Assertions
        # If it correctly saw the virtualized map, maps_updated should be 1, and maps_added should be 0.
        assert result["maps_updated"] == 1, f"Expected 1 refreshed map, got {result['maps_updated']}"
        assert result["maps_added"] == 0, f"Expected 0 new maps, got {result['maps_added']}"
        
    finally:
        tm_mod._manager_instance = original_manager

def test_process_file_labels_missing_maps_as_new(tmp_path: Path):
    """
    Verify that process_file correctly identifies truly new CONNECTION_MAPs.
    """
    project_root = tmp_path
    source_path = project_root / "source2.py"
    source_path.write_text("def source_func():\n    pass\n", encoding="utf-8")
    
    registry_path = project_root / "transparency_registry_new.json"
    manager = TransparencyManager(str(registry_path))
    
    import cline_utils.dependency_system.io.transparency_manager as tm_mod
    original_manager = tm_mod._manager_instance
    tm_mod._manager_instance = manager
    
    try:
        # No metadata in registry for source2.py
        
        source_ki = _ki("2A", source_path)
        target_ki = _ki("2B", project_root / "target.py")
        
        symbol_data = {
            "functions": [{"name": "source_func", "line": 1}],
            "classes": [],
        }
        
        key_info_list = [target_ki, source_ki]
        grid_row = "do"
        
        result = process_file(
            file_path=source_path,
            source_key="2A",
            key_info_list=key_info_list,
            grid_row=grid_row,
            tracker_ref="mini_tracker.md",
            entry_from=[],
            exits_to=[],
            symbol_data=symbol_data,
            project_root=project_root,
            dry_run=True,
            verbose=True,
        )
        
        assert result["maps_updated"] == 0
        assert result["maps_added"] == 1
        
    finally:
        tm_mod._manager_instance = original_manager
