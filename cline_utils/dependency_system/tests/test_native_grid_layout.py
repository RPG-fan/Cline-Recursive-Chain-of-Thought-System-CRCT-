from __future__ import annotations

import pytest
from typing import Dict, Set, Tuple

from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.config_manager import ConfigManager
from cline_utils.dependency_system.utils.viz import (
    NativeLayoutConfig,
    build_diagram_model,
    compute_native_layout,
)

def _global_map(n_children: int) -> Dict[str, KeyInfo]:
    gmap = {
        "h:/repo/src": KeyInfo("1A", "h:/repo/src", None, 1, True),
    }
    for i in range(n_children):
        key = f"1A{i+1}"
        path = f"h:/repo/src/file{i+1}.py"
        gmap[path] = KeyInfo(key, path, "h:/repo/src", 1, False)
    return gmap

def _config() -> ConfigManager:
    return ConfigManager()

def _migration_info(global_map: Dict[str, KeyInfo]):
    return {path: (info.key_string, info.key_string) for path, info in global_map.items()}

def test_native_grid_layout_partitions_columns():
    # Test with 10 children, max 4 per column -> should be 3 columns
    n_children = 10
    global_map = _global_map(n_children)
    config = NativeLayoutConfig(max_nodes_per_column=4, enable_hex_stagger=False)
    
    diagram = build_diagram_model(
        focus_keys_list_input=[info.key_string for info in global_map.values()],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links={},
    )
    
    layout = compute_native_layout(diagram, config)
    
    # Filter nodes that are children of 1A
    child_nodes = [layout.nodes[f"1A{i+1}"] for i in range(n_children)]
    
    # Check that children have different X coordinates (indicating multiple columns)
    x_coords = set(n.x for n in child_nodes)
    assert len(x_coords) == 3, f"Expected 3 columns for 10 nodes with max 4/col, got {len(x_coords)}: {x_coords}"

def test_native_hex_stagger_offsets_y():
    # Test with 10 children, max 4 per column, staggered
    n_children = 10
    global_map = _global_map(n_children)
    config = NativeLayoutConfig(max_nodes_per_column=4, enable_hex_stagger=True, hex_stagger_ratio=0.5)
    
    diagram = build_diagram_model(
        focus_keys_list_input=[info.key_string for info in global_map.values()],
        global_path_to_key_info_map=global_map,
        path_migration_info=_migration_info(global_map),
        all_tracker_paths_list=[],
        config_manager_instance=_config(),
        pre_aggregated_links={},
    )
    
    layout = compute_native_layout(diagram, config)
    
    # 10 nodes, 4/col -> 3 columns: Col 0 (4), Col 1 (3), Col 2 (3)
    # Stagger should apply to Col 1 (index 1)
    
    # Sort nodes by X then Y
    nodes_sorted = sorted(
        [layout.nodes[f"1A{i+1}"] for i in range(n_children)],
        key=lambda n: (n.x, n.y)
    )
    
    # Group by X
    columns = {}
    for i in range(n_children):
        n = layout.nodes[f"1A{i+1}"]
        if n.x not in columns: columns[n.x] = []
        columns[n.x].append(n)
    
    sorted_xs = sorted(columns.keys())
    col0_nodes = columns[sorted_xs[0]]
    col1_nodes = columns[sorted_xs[1]]
    
    assert col1_nodes[0].y > col0_nodes[0].y, "Column 1 should be staggered (shifted down)"
    # row_gap is 28, node_height is 40, ratio 0.5 -> expected (40+28)*0.5 = 34
    print(f"Col 0 Y[0]: {col0_nodes[0].y}")
    print(f"Col 1 Y[0]: {col1_nodes[0].y}")
    print(f"Diff: {col1_nodes[0].y - col0_nodes[0].y}")
    assert col1_nodes[0].y - col0_nodes[0].y == pytest.approx(34.0), f"Expected 34px stagger, got {col1_nodes[0].y - col0_nodes[0].y}"
